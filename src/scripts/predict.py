import pandas as pd
import numpy as np
import joblib
import os
from typing import Union, Tuple, Dict, Any


class FraudDetectionPredictor:
    """
    Classe para fazer predições de fraude usando o modelo treinado
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Inicializa o preditor carregando o modelo e scaler
        
        Args:
            models_dir: Diretório onde estão salvos os modelos
        """
        self.models_dir = models_dir
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.load_models()
    
    def load_models(self):
        """
        Carrega o modelo treinado, o scaler e as feature names
        """
        model_path = os.path.join(self.models_dir, 'model.pkl')
        scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
        features_path = os.path.join(self.models_dir, 'feature_names.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado em: {model_path}")
        
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler não encontrado em: {scaler_path}")
            
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Feature names não encontradas em: {features_path}")
        
        print("Carregando modelos")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = joblib.load(features_path)
        print("Modelos carregados com sucesso")
    
    def preprocess_single_transaction(self, transaction_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocessa uma única transação para predição
        
        Args:
            transaction_data: Dicionário com os dados da transação
            
        Returns:
            DataFrame preprocessado
        """
        df = pd.DataFrame([transaction_data])
        return self.preprocess_categorical_features(df)
    
    def preprocess_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocessa as features categóricas (mesmo processo do treinamento)
        """
        df_processed = df.copy()

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

        # Remove ID da transação se existir
        if 'trans_num' in df_processed.columns:
            df_processed.drop('trans_num', axis=1, inplace=True)

        # One-Hot Encoding
        category_dummies = pd.get_dummies(df_processed['category'], prefix='cat')
        state_dummies = pd.get_dummies(df_processed['state'], prefix='state')
        df_processed = pd.concat([df_processed, category_dummies, state_dummies], axis=1)
        df_processed.drop(['category', 'state'], axis=1, inplace=True)

        # Target Encoding para categorias com muitas opções
        # TODO: Usar médias do conjunto de treino salvo
        global_mean = 0.1  # Estimativa 
        
        for col in ['merchant', 'city', 'job']:
            df_processed[f'{col}_target_enc'] = global_mean
            df_processed.drop(col, axis=1, inplace=True)

        bool_columns = df_processed.select_dtypes(include=['bool']).columns
        df_processed[bool_columns] = df_processed[bool_columns].astype(int)

        df_processed = self.align_features_with_training(df_processed)

        return df_processed
    
    def align_features_with_training(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Alinha as features do DataFrame com as features usadas no treinamento
        """
        aligned_df = pd.DataFrame(0, index=df.index, columns=self.feature_names)
        
        for col in df.columns:
            if col in self.feature_names:
                aligned_df[col] = df[col]
        
        return aligned_df
    
    def predict_single(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Faz predição para uma única transação
        
        Args:
            transaction_data: Dicionário com os dados da transação
            
        Returns:
            Dicionário com a predição e probabilidade
        """
        try:
            df_processed = self.preprocess_single_transaction(transaction_data)
            
            # Padronizar features
            X_scaled = self.scaler.transform(df_processed)
            
            prediction = self.model.predict(X_scaled)[0]
            probability = self.model.predict_proba(X_scaled)[0]
            
            result = {
                'is_fraud': int(prediction),
                'fraud_probability': float(probability[1]),
                'no_fraud_probability': float(probability[0]),
                'confidence': float(max(probability))
            }
            
            return result
            
        except Exception as e:
            raise Exception(f"Erro na predição: {str(e)}")
    
    def predict_batch(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Faz predições para um lote de transações
        
        Args:
            transactions_df: DataFrame com múltiplas transações
            
        Returns:
            DataFrame com as predições
        """
        try:
            df_processed = self.preprocess_categorical_features(transactions_df.copy())
            
            X_scaled = self.scaler.transform(df_processed)
            
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
            
            results_df = transactions_df.copy()
            results_df['is_fraud_predicted'] = predictions
            results_df['fraud_probability'] = probabilities[:, 1]
            results_df['no_fraud_probability'] = probabilities[:, 0]
            results_df['confidence'] = np.max(probabilities, axis=1)
            
            return results_df
            
        except Exception as e:
            raise Exception(f"Erro na predição em lote: {str(e)}")
        