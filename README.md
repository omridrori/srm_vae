## VAE-SRM: Shared Space Encoding (SRM vs SRM-VAE)

This repository compares a classic linear Shared Response Model (SRM) with a non-linear VAE-based SRM (SRM-VAE) on ECoG data, evaluating how well language embeddings linearly predict the learned shared space across temporal lags.

### Data

Download the preprocessed dataset from the Zenodo record: [Zenodo: VAE-SRM ECoG dataset](https://zenodo.org/records/14730569).

After downloading, place the file as `all_data.pkl` in the project root (or update `DATA_PATH` in the script accordingly).

Expected pickle contents (`all_data.pkl`):
- **electrode_data**: numpy array shaped `[S, L, T, Emax]`
- **electrode_number**: numpy array length `S` with per-subject electrode counts
- **word_embeddings**: numpy array shaped `[T, D_emb]`
- **lags**: numpy array length `L` (milliseconds)

You can quickly sanity-check the file:

```python
import pickle
with open("all_data.pkl", "rb") as f:
    d = pickle.load(f)
print(d.keys())
```

### Environment setup

Python 3.10+ recommended. On Windows PowerShell:

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirments.txt
pip install brainiak
```

Notes:
- `brainiak` is required for the SRM baseline and is imported inside the script.
- `torch` (CPU or CUDA) is specified in `requirments.txt`. If you need a specific CUDA build, follow PyTorch install instructions and then run the rest of the installs.

### What the main script does

The file `srm_vs_vae_shared_encoding.py`:
- Loads `all_data.pkl`.
- For each requested lag, computes shared-space encoding performance for:
  - **SRM (brainiak)** baseline
  - **SRM-VAE** (subject-specific encoders/decoders, shared core)
- Plots mean Pearson r across shared dimensions with SEM bands 

### How to run

From the repository root:

```powershell
python srm_vs_vae_shared_encoding.py
```

This will:
- Train the VAE for each lag (defaults: 1000 epochs per lag),
- Fit the SRM baseline for each lag,
- Print progress to the console,
- Display a plot comparing SRM vs VAE curves.

 
 




