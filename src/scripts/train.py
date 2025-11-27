import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split


def preprocess_categorical_features(df):
    """
    Preprocessa as features categóricas do dataset
    """
    df_processed = df.copy()

    df_processed['is_fraud'] = pd.to_numeric(df_processed['is_fraud'])

    # Tratamento de dados temporais
    df_processed['trans_date_trans_time'] = pd.to_datetime(
        df_processed['trans_date_trans_time'], format='%d-%m-%Y %H:%M'
    )
    df_processed['hour'] = df_processed['trans_date_trans_time'].dt.hour
    df_processed['day_of_week'] = df_processed['trans_date_trans_time'].dt.dayofweek
    df_processed['month'] = df_processed['trans_date_trans_time'].dt.month
    df_processed.drop('trans_date_trans_time', axis=1, inplace=True)

    # Data de nascimento -> idade
    df_processed['dob'] = pd.to_datetime(df_processed['dob'], dayfirst=True)
    df_processed['age'] = (pd.to_datetime('today') - df_processed['dob']).dt.days // 365
    df_processed.drop('dob', axis=1, inplace=True)

    # Remove ID da transação
    df_processed.drop('trans_num', axis=1, inplace=True)

    # One-Hot Encoding
    category_dummies = pd.get_dummies(df_processed['category'], prefix='cat')
    state_dummies = pd.get_dummies(df_processed['state'], prefix='state')
    df_processed = pd.concat([df_processed, category_dummies, state_dummies], axis=1)
    df_processed.drop(['category', 'state'], axis=1, inplace=True)

    # Target Encoding para categorias com muitas opções
    for col in ['merchant', 'city', 'job']:
        target_mean = df_processed.groupby(col)['is_fraud'].mean()
        df_processed[f'{col}_target_enc'] = df_processed[col].map(target_mean)

        # Preencher NaN (categorias novas) com média global
        global_mean = df_processed['is_fraud'].mean()
        df_processed[f'{col}_target_enc'].fillna(global_mean, inplace=True)

        df_processed.drop(col, axis=1, inplace=True)

    # Converter colunas booleanas para int
    bool_columns = df_processed.select_dtypes(include=['bool']).columns
    df_processed[bool_columns] = df_processed[bool_columns].astype(int)

    return df_processed


def load_and_preprocess_data(data_path):
    """
    Carrega e preprocessa os dados de treino
    """
    print("Carregando dados de treino")
    df_train = pd.read_csv(data_path)
    
    # Separar features e target
    X = df_train.drop('is_fraud', axis=1)
    y = df_train['is_fraud']
    
    print(f"Dataset carregado com {X.shape[0]} amostras e {X.shape[1]} features")
    print(f"Taxa de fraude: {y.mean():.2%}")
    
    return X, y


def train_model(X_train, y_train):
    """
    Treina o modelo de detecção de fraude
    """
    print("Iniciando treinamento do modelo")
    
    # Padronização das features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Treinamento do modelo
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    print("Modelo treinado com sucesso!")
    
    return model, scaler


def evaluate_model(model, scaler, X_test, y_test):
    """
    Avalia o modelo nos dados de teste
    """
    print("Avaliando modelo")
    
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    auc_score = roc_auc_score(y_test, y_proba)
    print(f"\nAUC Score: {auc_score:.4f}")
    
    return y_pred, y_proba


def save_models(model, scaler, feature_names, models_dir):
    """
    Salva o modelo treinado, o scaler e as feature names
    """
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'model.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    features_path = os.path.join(models_dir, 'feature_names.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_names, features_path)
    
    print(f"Modelo salvo em: {model_path}")
    print(f"Scaler salvo em: {scaler_path}")
    print(f"Feature names salvas em: {features_path}")


def main():
    """
    Função principal do script de treinamento
    """
    data_dir = "data/raw"
    models_dir = "models"
    
    data_path = os.path.join(data_dir, "fraud_data.csv")

    if not os.path.exists(data_path):
        print(f"Erro: Arquivo {data_path} não encontrado!")
        return
    

    try:
        # Carrega dados brutos
        df_raw = pd.read_csv(data_path)
        
        # Preprocessa os dados
        df_processed = preprocess_categorical_features(df_raw)

        X, y = df_processed.drop('is_fraud', axis=1), df_processed['is_fraud']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model, scaler = train_model(X_train, y_train)

        evaluate_model(model, scaler, X_test, y_test)

        # Salvar feature names
        feature_names = X_train.columns.tolist()
        save_models(model, scaler, feature_names, models_dir)
        
        print("\nTreinamento concluído!")
        
    except Exception as e:
        print(f"Erro durante o treinamento: {str(e)}")


if __name__ == "__main__":
    main()
