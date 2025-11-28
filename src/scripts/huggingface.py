import joblib
import tempfile
import shutil
import os
from pathlib import Path
from typing import Dict
from huggingface_hub import HfApi, create_repo, hf_hub_download
from dotenv import load_dotenv

load_dotenv()


def get_hf_api():
    token = os.getenv("HF_TOKEN")
    return HfApi(token=token), token


def create_readme(model_name: str, metrics: Dict = None) -> str:
    metrics_text = ""
    if metrics:
        metrics_text = "\n## Metrics\n"
        for k, v in metrics.items():
            if isinstance(v, float):
                if k in ['accuracy', 'roc_auc']:
                    metrics_text += f"- **{k.replace('_', ' ').title()}**: {v:.4f} ({v*100:.2f}%)\n"
                else:
                    metrics_text += f"- **{k.replace('_', ' ').title()}**: {v:.4f}\n"
    
    return f"""# {model_name}

Credit Card Fraud Detection Model using Logistic Regression.
{metrics_text}
## Deployment Criteria

This model was automatically deployed because it meets the following criteria:
- Accuracy > 90%
- ROC AUC Score > 75%

## Usage
```python
from huggingface import load_model_from_hf
import pandas as pd

model, feature_names, scaler = load_model_from_hf("your-repo-id")

data = pd.DataFrame(...)  
X_scaled = scaler.transform(data[feature_names])

predictions = model.predict(X_scaled)
probabilities = model.predict_proba(X_scaled)[:, 1]
```

## Model Architecture

- **Algorithm**: Logistic Regression (scikit-learn)
- **Features**: Preprocessed transaction data with categorical encoding
- **Preprocessing**: StandardScaler normalization

## Training Details

- **Data Processing**:
  - Temporal features: hour, day_of_week, month
  - Age calculation from date of birth
  - One-Hot Encoding: category, state
  - Target Encoding: merchant, city, job
- **Train/Test Split**: 80/20 with random_state=42
- **Model Parameters**: max_iter=1000, random_state=42

## Files Included

- `model.pkl`: Trained Logistic Regression model
- `scaler.pkl`: StandardScaler for feature normalization
- `feature_names.pkl`: List of feature names for proper data alignment
"""


def upload_model_to_hf(repo_name: str, model_path: Path, feature_names_path: Path, 
                       scaler_path: Path, metrics: Dict = None, 
                       model_name: str = "Fraud Detection Model"):
    api, token = get_hf_api()
    
    create_repo(repo_id=repo_name, token=token, exist_ok=True)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        shutil.copy2(model_path, temp_path / "model.pkl")
        shutil.copy2(feature_names_path, temp_path / "feature_names.pkl")
        shutil.copy2(scaler_path, temp_path / "scaler.pkl")
        
        readme_content = create_readme(model_name, metrics)
        (temp_path / "README.md").write_text(readme_content, encoding='utf-8')
        
        api.upload_folder(
            folder_path=temp_path,
            repo_id=repo_name,
            token=token
        )
    
    return f"https://huggingface.co/{repo_name}"


def load_model_from_hf(repo_id: str):
    token = os.getenv("HF_TOKEN")
    
    model_path = hf_hub_download(repo_id=repo_id, filename="model.pkl", token=token)
    feature_names_path = hf_hub_download(repo_id=repo_id, filename="feature_names.pkl", token=token)
    scaler_path = hf_hub_download(repo_id=repo_id, filename="scaler.pkl", token=token)
    
    return (
        joblib.load(model_path), 
        joblib.load(feature_names_path), 
        joblib.load(scaler_path)
    )


def download_model_from_hf(repo_id: str, local_dir: Path = None):
    if local_dir is None:
        local_dir = Path("models")
    
    local_dir.mkdir(exist_ok=True)
    
    token = os.getenv("HF_TOKEN")
    
    files = ["model.pkl", "feature_names.pkl", "scaler.pkl"]
    
    for filename in files:
        remote_path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
        local_path = local_dir / filename
        shutil.copy2(remote_path, local_path)
    
    print(f"Modelos baixados para {local_dir}")
