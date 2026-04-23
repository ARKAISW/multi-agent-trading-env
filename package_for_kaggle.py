import shutil
import os
from pathlib import Path

def package_project():
    root = Path.cwd()
    output_filename = "mate_env_package"
    
    # Files/Dirs to include
    include_dims = ["env", "agents", "utils", "data", "training", "policy"]
    include_files = ["requirements-space.txt", "app.py"]
    
    # Create temp directory
    temp_dir = root / "temp_kaggle_pkg"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    
    print(f"Packaging files from {root}...")
    
    for item in include_dims:
        src = root / item
        if src.exists():
            shutil.copytree(src, temp_dir / item, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints"))
            
    for item in include_files:
        src = root / item
        if src.exists():
            shutil.copy2(src, temp_dir / item)
            
    # Create Zip
    shutil.make_archive(output_filename, 'zip', temp_dir)
    shutil.rmtree(temp_dir)
    
    print(f"\nDone! Upload '{output_filename}.zip' to Kaggle as a new Dataset.")
    print("Link: https://www.kaggle.com/datasets?new=true")

if __name__ == "__main__":
    package_project()
