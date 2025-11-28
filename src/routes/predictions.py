from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import pandas as pd
import io
from src.scripts.predict import FraudDetectionPredictor

class TransactionData(BaseModel):
    trans_date_trans_time: str = Field(..., description="Data e hora da transação, formato 'YYYY-MM-DD HH:MM'")
    merchant: str = Field(..., description="Nome do comerciante")
    category: str = Field(..., description="Categoria da transação")
    amt: float = Field(..., description="Valor da transação")
    city: str = Field(..., description="Cidade")
    state: str = Field(..., description="Estado")
    lat: float = Field(..., description="Latitude")
    long: float = Field(..., description="Longitude")
    city_pop: int = Field(..., description="População da cidade")
    job: str = Field(..., description="Profissão")
    dob: str = Field(..., description="Data de nascimento, formato 'DD-MM-YYYY'")
    trans_num: Optional[str] = Field(None, description="Número da transação")
    merch_lat: float = Field(..., description="Latitude do comerciante")
    merch_long: float = Field(..., description="Longitude do comerciante")

class PredictionResponse(BaseModel):
    fraud_probability: float = Field(..., description="Probabilidade de ser fraude (0-1)")
    confidence: float = Field(..., description="Confiança da predição (0-1)")
    classification: str = Field(..., description="Classificação textual: 'Fraude' ou 'Não Fraude'")

predictor = FraudDetectionPredictor()

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionData):
    """
    Prediz se uma transação é fraudulenta ou não.
    
    Retorna:
    - fraud_probability: Probabilidade de ser fraude (0-1)
    - confidence: Nível de confiança da predição (0-1)
    - classification: Classificação em texto ("Fraude" ou "Não Fraude")
    """
    try:
        transaction_data = transaction.model_dump()
        
        result = predictor.predict_single(transaction_data)
        
        classification = "Fraude" if result['is_fraud'] == 1 else "Não Fraude"
        
        response = PredictionResponse(
            fraud_probability=result['fraud_probability'],
            confidence=result['confidence'],
            classification=classification
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro na predição: {str(e)}"
        )