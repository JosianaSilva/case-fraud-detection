from huggingface import download_model_from_hf
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODELS_DIR = Path(PROJECT_ROOT) / "models"
HF_USERNAME = os.getenv("HF_USERNAME")
HF_REPO_NAME = os.getenv("HF_REPO_NAME")


def download_model():
    if not HF_REPO_NAME:
        raise ValueError("HF_REPO_NAME environment variable is not set.")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    download_model_from_hf(
        repo_id=f"{HF_USERNAME}/{HF_REPO_NAME}",
        local_dir=MODELS_DIR
    )

if __name__ == "__main__":
    download_model()