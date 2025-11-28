import json
import os
import pandas as pd
import joblib
from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from huggingface import upload_model_to_hf

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

MIN_ACCURACY = 0.90
MIN_ROC_AUC = 0.75


def load_test_data():
    df_test = pd.read_csv(DATA_DIR / "df_test.csv")
    return df_test.drop('is_fraud', axis=1), df_test['is_fraud']


def load_model_components():
    return (
        joblib.load(MODELS_DIR / "model.pkl"),
        joblib.load(MODELS_DIR / "feature_names.pkl"),
        joblib.load(MODELS_DIR / "scaler.pkl")
    )


def calculate_metrics(X_test, y_test, model, feature_names, scaler):
    X_scaled = scaler.transform(X_test[feature_names])
    predictions = model.predict(X_scaled)
    
    return {
        'accuracy': float(accuracy_score(y_test, predictions)),
        'roc_auc': float(roc_auc_score(y_test, predictions)),
        'precision': float(precision_score(y_test, predictions)),
        'recall': float(recall_score(y_test, predictions)),
        'f1_score': float(f1_score(y_test, predictions))
    }


def save_metrics(metrics):
    metrics_file = MODELS_DIR / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics_file


def validate_metrics(metrics):
    accuracy_ok = metrics['accuracy'] > MIN_ACCURACY
    roc_auc_ok = metrics['roc_auc'] > MIN_ROC_AUC
    
    print(f"Accuracy: {metrics['accuracy']:.4f} ({'✓' if accuracy_ok else '✗'})")
    print(f"ROC AUC: {metrics['roc_auc']:.4f} ({'✓' if roc_auc_ok else '✗'})")
    
    return accuracy_ok and roc_auc_ok


def get_repo_config():
    username = os.getenv("HF_USERNAME")
    repo_name = os.getenv("HF_REPO_NAME", "fraud-detection-logistic-regression")
    
    if not username:
        raise ValueError("Configure HF_USERNAME no .env")
    
    return f"{username}/{repo_name}"


def check_model_files():
    required_files = ["model.pkl", "feature_names.pkl", "scaler.pkl"]
    missing = [f for f in required_files if not (MODELS_DIR / f).exists()]
    
    if missing:
        raise FileNotFoundError(f"Arquivos não encontrados: {missing}")


def main():
    try:
        check_model_files()
        
        X_test, y_test = load_test_data()
        model, feature_names, scaler = load_model_components()
        
        metrics = calculate_metrics(X_test, y_test, model, feature_names, scaler)
        save_metrics(metrics)
        
        if not validate_metrics(metrics):
            return False
        
        repo_name = get_repo_config()
        url = upload_model_to_hf(
            repo_name=repo_name,
            model_path=MODELS_DIR / "model.pkl",
            feature_names_path=MODELS_DIR / "feature_names.pkl",
            scaler_path=MODELS_DIR / "scaler.pkl",
            metrics=metrics
        )
        
        print(f"Deploy concluído: {url}")
        return True
        
    except Exception as e:
        print(f"Deploy falhou: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)