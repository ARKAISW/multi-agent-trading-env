import os
import sys
import subprocess
from pathlib import Path

def run_notebook():
    print("============================================================")
    print("  QuantHive — Notebook-based GRPO Training (HF Jobs)")
    print("============================================================")

    # 1. Install necessary execution dependencies
    print("📦 Installing notebook execution tools...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nbconvert", "nbformat", "ipykernel", "huggingface_hub"], check=True)

    # 2. Execute the notebook
    print("🚀 Executing mate_training.ipynb (mimicking Colab flow)...")
    try:
        # We run it with --to notebook --execute --inplace
        subprocess.run([
            "jupyter", "nbconvert", 
            "--to", "notebook", 
            "--execute", 
            "--inplace", 
            "mate_training.ipynb"
        ], check=True)
        print("✅ Notebook execution successful!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Notebook execution failed: {e}")
        # We don't exit here because the notebook might have saved partial results
    
    # 3. Model Storage to HF Hub
    # The notebook saves to ./quanthive-trader-grpo-lora
    model_path = Path("quanthive-trader-grpo-lora")
    if model_path.exists():
        print(f"📡 Found trained model at {model_path}. Pusing to HF Hub...")
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            repo_id = "ARKAISW/quanthive-trader-grpo-lora"
            
            # Use HF_TOKEN if available in environment
            token = os.environ.get("HF_TOKEN")
            
            api.upload_folder(
                folder_path=str(model_path),
                repo_id=repo_id,
                repo_type="model",
                token=token
            )
            print(f"✅ Model successfully pushed to https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"⚠️ Model push failed: {e}")
            print("You can manually push the 'quanthive-trader-grpo-lora' folder later.")
    else:
        print("❌ Trained model folder not found. Check notebook output for errors.")

    print("============================================================")
    print("🏁 Processing Finished.")

if __name__ == "__main__":
    run_notebook()
