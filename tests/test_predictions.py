import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from src.main import app

client = TestClient(app)


@pytest.fixture
def valid_transaction_data():
    """Fixture com dados válidos de transação"""
    return {
        "trans_date_trans_time": "2023-01-15 14:30",
        "merchant": "fraud_Kirlin and Sons",
        "category": "personal_care",
        "amt": 29.84,
        "city": "Malad City",
        "state": "ID",
        "lat": 42.1808,
        "long": -112.2620,
        "city_pop": 2071,
        "job": "Mechanical engineer",
        "dob": "15-03-1988",
        "trans_num": "2da90c7d74bd46a0caf3777415b3ebd3",
        "merch_lat": 43.150704,
        "merch_long": -112.154481
    }


@pytest.fixture
def mock_predictor_response_fraud():
    """Fixture com resposta mockada indicando fraude"""
    return {
        "is_fraud": 1,
        "fraud_probability": 0.92,
        "confidence": 0.85
    }


@pytest.fixture
def mock_predictor_response_not_fraud():
    """Fixture com resposta mockada indicando não fraude"""
    return {
        "is_fraud": 0,
        "fraud_probability": 0.15,
        "confidence": 0.78
    }


class TestPredictEndpoint:
    """Testes para o endpoint /predict"""
    
    @patch('src.routes.predictions.predictor.predict_single')
    def test_predict_fraud_success(self, mock_predict, valid_transaction_data, mock_predictor_response_fraud):
        """Testa predição bem-sucedida de fraude"""
        mock_predict.return_value = mock_predictor_response_fraud
        
        response = client.post("/predict", json=valid_transaction_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["classification"] == "Fraude"
        assert data["fraud_probability"] == 0.92
        assert data["confidence"] == 0.85
        assert mock_predict.called
    
    @patch('src.routes.predictions.predictor.predict_single')
    def test_predict_not_fraud_success(self, mock_predict, valid_transaction_data, mock_predictor_response_not_fraud):
        """Testa predição bem-sucedida de não fraude"""
        mock_predict.return_value = mock_predictor_response_not_fraud
        
        response = client.post("/predict", json=valid_transaction_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["classification"] == "Não Fraude"
        assert data["fraud_probability"] == 0.15
        assert data["confidence"] == 0.78
    
    def test_predict_missing_required_field(self, valid_transaction_data):
        """Testa requisição com campo obrigatório faltando"""
        incomplete_data = valid_transaction_data.copy()
        del incomplete_data["amt"]
        
        response = client.post("/predict", json=incomplete_data)
        
        assert response.status_code == 422
    
    def test_predict_invalid_field_type(self, valid_transaction_data):
        """Testa requisição com tipo de campo inválido"""
        invalid_data = valid_transaction_data.copy()
        invalid_data["amt"] = "invalid_amount"
        
        response = client.post("/predict", json=invalid_data)
        
        assert response.status_code == 422
    
    @patch('src.routes.predictions.predictor.predict_single')
    def test_predict_internal_error(self, mock_predict, valid_transaction_data):
        """Testa tratamento de erro interno"""
        mock_predict.side_effect = Exception("Erro no modelo")
        
        response = client.post("/predict", json=valid_transaction_data)
        
        assert response.status_code == 500
        assert "Erro na predição" in response.json()["detail"]
    
    @patch('src.routes.predictions.predictor.predict_single')
    def test_predict_without_optional_field(self, mock_predict, valid_transaction_data, mock_predictor_response_not_fraud):
        """Testa predição sem campo opcional (trans_num)"""
        mock_predict.return_value = mock_predictor_response_not_fraud
        
        data_without_optional = valid_transaction_data.copy()
        del data_without_optional["trans_num"]
        
        response = client.post("/predict", json=data_without_optional)
        
        assert response.status_code == 200
        data = response.json()
        assert "fraud_probability" in data
        assert "confidence" in data
        assert "classification" in data
    
    @patch('src.routes.predictions.predictor.predict_single')
    def test_predict_response_structure(self, mock_predict, valid_transaction_data, mock_predictor_response_fraud):
        """Testa estrutura da resposta"""
        mock_predict.return_value = mock_predictor_response_fraud
        
        response = client.post("/predict", json=valid_transaction_data)
        
        assert response.status_code == 200
        data = response.json()
        
        required_fields = ["fraud_probability", "confidence", "classification"]
        for field in required_fields:
            assert field in data
        
        assert isinstance(data["fraud_probability"], (int, float))
        assert isinstance(data["confidence"], (int, float))
        assert isinstance(data["classification"], str)
        
        assert 0 <= data["fraud_probability"] <= 1
        assert 0 <= data["confidence"] <= 1
        assert data["classification"] in ["Fraude", "Não Fraude"]

