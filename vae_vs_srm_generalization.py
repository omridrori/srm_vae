# vae_vs_srm_generalization.py
# --- CONFIG (editable) ---
DATA_PATH   = "./all_data.pkl"
SEED        = 1234
TRAIN_RATIO = 0.8
LAG_LIST    =  [-2000, -1000, -500, 0, 100, 200, 300, 500, 1000, 1500, 1800]  # Lags to test (ms)
PCA_DIM     = 50

# Directory to save plots
OUTPUT_DIR  = "shared_space_plots"

# --- Imports ---
import os, pickle, random
from typing import Dict, List
from dataclasses import dataclass

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from brainiak.funcalign import srm as brainiak_srm

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Shared dimension and VAE hyper-parameters
SHARED_K   = 5
VAE_EPOCHS = 500
VAE_LR     = 1e-3
VAE_BETA   = 0.1

# ----------------- helpers -----------------

def set_global_seed(seed=1234):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class SubjectView:
    train: torch.Tensor
    test:  torch.Tensor
    elec_num: int = 0

@dataclass
class LagBatch:
    lag_ms: int
    subjects: List[int]
    subject_views: Dict[int, SubjectView]
    X_train: torch.Tensor
    X_test:  torch.Tensor
    elec_num: Dict[int,int]

# ------------- IO -------------

def load_all_data(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return (np.asarray(obj["electrode_data"]),
            np.asarray(obj["electrode_number"], int),
            np.asarray(obj["word_embeddings"]),
            np.asarray(obj["lags"]).reshape(-1))

# ------------- preprocessing -------------

def _prep_embeddings(Xtr, Xte, pca_dim, seed):
    pca = PCA(n_components=pca_dim, random_state=seed)
    Xtr_p = pca.fit_transform(Xtr)
    Xte_p = pca.transform(Xte)
    return normalize(Xtr_p, axis=1), normalize(Xte_p, axis=1)

def _zscore_train_apply(tr, te):
    mu = tr.mean(0, keepdims=True); sd = tr.std(0, keepdims=True)+1e-8
    return (tr-mu)/sd, (te-mu)/sd

def choose_lag_index(lags_ms, target_ms):
    return int(np.argmin(np.abs(lags_ms-target_ms)))

def time_split_indices(T, ratio):
    n = min(max(1,int(round(T*ratio))), T-1)
    return np.arange(n), np.arange(n, T)

# ------------- VAE-SRM model -------------
class Enc(nn.Module):
    def __init__(self,e_i,k): super().__init__(); self.lin=nn.Linear(e_i,2*k); nn.init.xavier_uniform_(self.lin.weight)
    def forward(self,x): return torch.chunk(self.lin(x),2,-1)
class Dec(nn.Module):
    def __init__(self,k,e_i): super().__init__(); self.lin=nn.Linear(k,e_i,bias=False); nn.init.xavier_uniform_(self.lin.weight)
    def forward(self,z): return self.lin(z)
class SRMVAE(nn.Module):
    def __init__(self,elec_num:Dict[int,int],k:int):
        super().__init__()
        self.encoders=nn.ModuleDict({str(s):Enc(e,k) for s,e in elec_num.items()})
        self.decoders=nn.ModuleDict({str(s):Dec(k,e) for s,e in elec_num.items()})
    @staticmethod
    def _agg(mu_list,lv_list):
        prec=[torch.exp(-lv) for lv in lv_list]; tot=torch.stack(prec,0).sum(0)+1e-8
        mu=torch.stack([m*p for m,p in zip(mu_list,prec)],0).sum(0)/tot
        return mu, -torch.log(tot)
    def encode_group(self,views,split):
        mus,lvs=[],[]
        for sid_str,enc in self.encoders.items():
            sid=int(sid_str)
            if sid in views:
                mu,lv=enc(getattr(views[sid],split))
                mus.append(mu); lvs.append(lv)
        return self._agg(mus,lvs)
    def forward(self,views,beta):
        mu,lv=self.encode_group(views,"train")
        z=mu+torch.exp(0.5*lv)*torch.randn_like(mu)
        rec=sum(F.mse_loss(self.decoders[sid_str](z), views[int(sid_str)].train)
                for sid_str in self.decoders.keys() if int(sid_str) in views)
        kl=-0.5*torch.mean(1+lv-mu.pow(2)-lv.exp())
        return rec+beta*kl
    @torch.no_grad()
    def infer_z(self,views,split):
        return self.encode_group(views,split)[0]

def train_vae(batch,k,epochs,lr,beta):
    dev=torch_device()
    for v in batch.subject_views.values(): v.train,v.test=v.train.to(dev),v.test.to(dev)
    model=SRMVAE(batch.elec_num,k).to(dev)
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    for _ in range(epochs):
        opt.zero_grad(); loss=model(batch.subject_views,beta); loss.backward(); opt.step()
    return model

# ------------- batch builder -------------

def build_batches(Y,elec,X,lags,lag_ms,ratio,test_sid):
    S,_,T,_=Y.shape
    idx=choose_lag_index(lags,lag_ms)
    tr,te=time_split_indices(T,ratio)
    Xtr,Xte=X[tr],X[te]
    Xtr-=Xtr.mean(0,keepdims=True); Xte-=Xtr.mean(0,keepdims=True)
    train_sids=[s for s in range(1,S+1) if s!=test_sid]
    views,elec_dict={},{}
    for sid in train_sids:
        e=int(elec[sid-1]); mat=Y[sid-1,idx,:,:e]
        tr_z, te_z = _zscore_train_apply(mat[tr], mat[te])
        views[sid] = SubjectView(torch.tensor(tr_z, dtype=torch.float32),
                                 torch.tensor(te_z, dtype=torch.float32))
        elec_dict[sid] = e
    batch = LagBatch(int(lags[idx]),
                     train_sids,
                     views,
                     torch.tensor(Xtr, dtype=torch.float32),
                     torch.tensor(Xte, dtype=torch.float32),
                     elec_dict)
    e_test=int(elec[test_sid-1]); mat_t=Y[test_sid-1,idx,:,:e_test]
    tr_z_t, te_z_t = _zscore_train_apply(mat_t[tr], mat_t[te])
    test_view = SubjectView(torch.tensor(tr_z_t, dtype=torch.float32),
                            torch.tensor(te_z_t, dtype=torch.float32),
                            elec_num=e_test)
    return batch,test_view

# ------------- analyses -------------

def vae_generalization(batch,test_view):
    model=train_vae(batch,SHARED_K,VAE_EPOCHS,VAE_LR,VAE_BETA).eval()
    z_train=model.infer_z(batch.subject_views,"train").cpu().numpy()
    Xtr_p,Xte_p=_prep_embeddings(batch.X_train.numpy(),batch.X_test.numpy(),PCA_DIM,SEED)
    enc=LinearRegression().fit(Xtr_p,z_train)
    z_pred=enc.predict(Xte_p)
    Ytr=test_view.train.numpy(); W,_,_,_=np.linalg.lstsq(Ytr,z_train,rcond=None)
    z_true=test_view.test.numpy() @ W
    vx,vy=z_true-z_true.mean(0), z_pred-z_pred.mean(0)
    r=np.sum(vx*vy,0)/(np.sqrt(np.sum(vx**2,0))*np.sqrt(np.sum(vy**2,0))+1e-8)
    return float(np.nanmean(r))

# Classic SRM generalization with safe device transfers
def srm_generalization(batch, test_view):
    train_data = [v.train.cpu().numpy().T for v in batch.subject_views.values()]
    self_tr = test_view.train.cpu().numpy().T
    self_te = test_view.test.cpu().numpy().T
    srm=brainiak_srm.SRM(n_iter=50,features=SHARED_K); srm.fit(train_data)
    shared_train=srm.s_.T
    Xtr_p,Xte_p=_prep_embeddings(batch.X_train.numpy(),batch.X_test.numpy(),PCA_DIM,SEED)
    enc=LinearRegression().fit(Xtr_p,shared_train)
    z_pred=enc.predict(Xte_p)
    W=self_tr @ shared_train @ np.linalg.inv(shared_train.T @ shared_train)
    z_true=(W.T @ self_te).T
    vx,vy=z_true-z_true.mean(0), z_pred-z_pred.mean(0)
    r=np.sum(vx*vy,0)/(np.sqrt(np.sum(vx**2,0))*np.sqrt(np.sum(vy**2,0))+1e-8)
    return float(np.nanmean(r))

# ------------- main -------------
if __name__=="__main__":
    set_global_seed(SEED)
    Y,elec,X,lags=load_all_data(DATA_PATH)
    subjects=list(range(1,len(elec)+1))
    scores_vae={s:{} for s in subjects}; scores_srm={s:{} for s in subjects}

    for sid in subjects:
        print(f"\n>>> Subject {sid}")
        for lag in LAG_LIST:
            batch,test_view=build_batches(Y,elec,X,lags,lag,TRAIN_RATIO,sid)
            r_vae=vae_generalization(batch,test_view)
            r_srm=srm_generalization(batch,test_view)
            scores_vae[sid][lag]=r_vae; scores_srm[sid][lag]=r_srm
            print(f"  lag {lag:4d} | VAE {r_vae:.3f} | SRM {r_srm:.3f}")

    # ---------- plotting per subject ----------
    for sid in subjects:
        plt.figure(figsize=(5.2,3.6))
        lags=LAG_LIST
        plt.plot(lags,[scores_vae[sid][l] for l in lags],'-o',label='VAE',color='dodgerblue')
        plt.plot(lags,[scores_srm[sid][l] for l in lags],'-o',label='SRM',color='darkorange')
        plt.axvline(0,ls="--",color='grey',alpha=0.5)
        plt.title(f'Subject {sid}'); plt.xlabel('lag (ms)'); plt.ylabel('generalization r')
        plt.legend(); plt.tight_layout();
        plt.savefig(f"{OUTPUT_DIR}/subject_{sid}_vae_vs_srm.png", dpi=300)
        plt.close()

    # ---------- combined plot ----------
    colors=plt.cm.tab10(np.linspace(0,1,len(subjects)))
    plt.figure(figsize=(6,4))
    for i,sid in enumerate(subjects):
        plt.plot(LAG_LIST,[scores_vae[sid][l] for l in LAG_LIST],color=colors[i],lw=1.8,label=f'S{sid}')
    plt.axvline(0,ls='--',color='grey',alpha=0.6); plt.title('All subjects – VAE'); plt.xlabel('lag (ms)'); plt.ylabel('r'); plt.legend(ncol=2,fontsize=8); plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/all_subjects_vae_only.png", dpi=300)
    plt.close()

    # ---------- averages ----------
    def mean_sem(score_dict):
        mean,sem=[],[]
        for l in LAG_LIST:
            arr=np.array([score_dict[sid][l] for sid in subjects])
            mean.append(arr.mean()); sem.append(arr.std(ddof=1)/np.sqrt(len(arr)))
        return np.array(mean),np.array(sem)
    m_vae,se_vae=mean_sem(scores_vae); m_srm,se_srm=mean_sem(scores_srm)
    plt.figure(figsize=(6,4))
    plt.plot(LAG_LIST,m_vae,color='dodgerblue',lw=3,label='VAE'); plt.fill_between(LAG_LIST,m_vae-se_vae,m_vae+se_vae,color='dodgerblue',alpha=0.25)
    plt.plot(LAG_LIST,m_srm,color='darkorange',lw=3,label='SRM'); plt.fill_between(LAG_LIST,m_srm-se_srm,m_srm+se_srm,color='darkorange',alpha=0.25)
    plt.axvline(0,ls='--',color='grey',alpha=0.6)
    plt.xlabel('lag (ms)'); plt.ylabel('mean r ± sem'); plt.title('VAE vs classic SRM – average'); plt.legend(); plt.grid(True,linestyle=':',alpha=0.5); plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/averages_vae_vs_srm.png", dpi=300)
    plt.close()

    # ---------- combined subjects plot ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(subjects)))
    # VAE Plot
    axes[0].set_title('All Subjects: VAE Generalization')
    for i, sid in enumerate(subjects):
        lags_sorted = sorted(scores_vae[sid].keys())
        scores_sorted = [scores_vae[sid][lag] for lag in lags_sorted]
        axes[0].plot(lags_sorted, scores_sorted, color=colors[i], lw=2, label=f'Subject {sid}')
    # SRM Plot
    axes[1].set_title('All Subjects: Classic SRM Generalization')
    for i, sid in enumerate(subjects):
        lags_sorted = sorted(scores_srm[sid].keys())
        scores_sorted = [scores_srm[sid][lag] for lag in lags_sorted]
        axes[1].plot(lags_sorted, scores_sorted, color=colors[i], lw=2, label=f'Subject {sid}')
    for ax in axes:
        ax.axvline(0, ls="--", color="grey", alpha=0.6)
        ax.set_xlabel('lag (ms)'); ax.grid(True, linestyle=":", alpha=0.5); ax.legend()
    axes[0].set_ylabel('Generalization Performance (r)'); plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/all_subjects_vae_vs_srm.png", dpi=300)
    plt.close(fig)

    # Create the plot
    plt.figure(figsize=(7, 5))
    plt.plot(LAG_LIST, m_vae, color="dodgerblue", lw=3, label="VAE Generalization (Avg)")
    plt.fill_between(LAG_LIST, m_vae - se_vae, m_vae + se_vae, color="dodgerblue", alpha=0.2)
    plt.plot(LAG_LIST, m_srm, color="darkorange", lw=3, label="Classic SRM Generalization (Avg)")
    plt.fill_between(LAG_LIST, m_srm - se_srm, m_srm + se_srm, color="darkorange", alpha=0.2)
    plt.axvline(0, ls="--", color="grey", alpha=0.6); plt.xlabel("lag (ms)"); plt.ylabel("Avg. Generalization Performance (r)")
    plt.title("Comparison of Average Generalization Performance"); plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/average_vae_vs_srm.png", dpi=300)
    plt.close()