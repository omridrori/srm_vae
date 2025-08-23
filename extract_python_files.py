import os
import glob

def extract_python_files_to_txt():
    """
    Extract all Python files (excluding notebooks) in the current directory
    and create a TXT file with filename and content separated by asterisks
    """
    
    # Get all Python files in current directory (excluding notebooks)
    python_files = []
    
    # Find all .py files
    py_files = glob.glob("*.py")
    python_files.extend(py_files)
    
    # Find all .py files in subdirectories
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                python_files.extend([full_path])
    
    # Remove duplicates and sort
    python_files = sorted(list(set(python_files)))
    
    # Create output file
    output_filename = "python_files_content.txt"
    
    with open(output_filename, 'w', encoding='utf-8') as output_file:
        for i, file_path in enumerate(python_files):
            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Write filename
                output_file.write(f"File: {file_path}\n")
                output_file.write("=" * 50 + "\n")
                
                # Write content
                output_file.write(content)
                
                # Add separator (except for last file)
                if i < len(python_files) - 1:
                    output_file.write("\n" + "*" * 80 + "\n\n")
                    
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                output_file.write(f"Error reading file: {e}\n")
                if i < len(python_files) - 1:
                    output_file.write("\n" + "*" * 80 + "\n\n")
    
    print(f"Extracted {len(python_files)} Python files to {output_filename}")
    print("Files processed:")
    for file_path in python_files:
        print(f"  - {file_path}")

if __name__ == "__main__":
    extract_python_files_to_txt() 