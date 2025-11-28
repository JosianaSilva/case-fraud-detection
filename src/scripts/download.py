from .huggingface import download_model_from_hf
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODELS_DIR = Path(PROJECT_ROOT) / "models"
HF_USERNAME = os.getenv("HF_USERNAME")
HF_REPO_NAME = os.getenv("HF_REPO_NAME")


def download_model(hf_repo_name: str = None, hf_username: str = None):
    if not hf_repo_name or not hf_username:
        hf_repo_name = HF_REPO_NAME
        hf_username = HF_USERNAME
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    download_model_from_hf(
        repo_id=f"{hf_username}/{hf_repo_name}",
        local_dir=MODELS_DIR
    )

if __name__ == "__main__":
    download_model()