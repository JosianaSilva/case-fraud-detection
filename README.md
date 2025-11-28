# Case - Detecção de Fraudes

API para predição de fraudes usando modelo de Machine Learning desenvolvida com FastAPI.

## 📁 Estrutura do Projeto

```
case-detecção-de-fraude/
├── data/
│   ├── processed/           # Dados processados para treino
│   └── raw/                 # Dados brutos
├── models/                  # Modelos treinados e métricas
├── notebooks/               # Jupyter notebooks para análise
├── src/
│   ├── main.py             # Aplicação principal FastAPI
│   ├── routes/             # Endpoints da API
│   └── scripts/            # Scripts de treino e deploy
└── requirements.txt        # Dependências Python
```

## 📋 Pré-requisitos

- Python 3.8+
- Docker e Docker Compose (para execução com containers)
- Git

## 🚀 Como começar

### 1. Clonar o repositório

```bash
git clone https://github.com/JosianaSilva/case-fraud-detection.git
cd case-fraud-detection
```

### 2. Opção A: Execução Local

#### Configuração do Ambiente

1. **Criar ambiente virtual:**
```bash
python -m venv env
```

2. **Ativar o ambiente virtual:**
```bash
# Windows
env\Scripts\activate

# Linux/Mac
source env/bin/activate
```

3. **Instalar dependências:**
```bash
pip install -r requirements.txt
```

#### Execução

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### 2. Opção B: Execução com Docker

#### Usando Docker Compose (Recomendado)

```bash
docker-compose up --build
```

#### Usando Docker diretamente

```bash
# Build da imagem
docker build -t fraud-detection .

# Executar container
docker run -p 8000:8000 fraud-detection
```

## 📖 Documentação da API

Após iniciar a aplicação, acesse:

- **Documentação Swagger:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health Check:** http://localhost:8000/health

## 🔍 Exemplos de Uso

### Health Check

```bash
curl -X GET "http://localhost:8000/health"
```

**Resposta:**
```json
{
  "status": "healthy"
}
```

### Predição de Fraude

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "trans_date_trans_time": "15-01-2024 14:30",
       "merchant": "fraud_Rippin, Kub and Mann",
       "category": "misc_net",
       "amt": 4.97,
       "city": "Malvern",
       "state": "AR",
       "lat": 34.9659,
       "long": -92.8092,
       "city_pop": 10563,
       "job": "Mechanical engineer",
       "dob": "09/03/1978",
       "merch_lat": 33.986391,
       "merch_long": -81.200714
     }'
```

**Resposta:**
```json
{
  "fraud_probability": 0.00026010730397907594,
  "confidence": 0.9997398926960209,
  "classification": "Não Fraude"
}
```

## 📊 Campos Obrigatórios

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `trans_date_trans_time` | string | Data e hora da transação |
| `merchant` | string | Nome do comerciante |
| `category` | string | Categoria da transação |
| `amt` | float | Valor da transação |
| `city` | string | Cidade |
| `state` | string | Estado |
| `lat` | float | Latitude |
| `long` | float | Longitude |
| `city_pop` | integer | População da cidade |
| `job` | string | Profissão |
| `dob` | string | Data de nascimento |
| `merch_lat` | float | Latitude do comerciante |
| `merch_long` | float | Longitude do comerciante |

## 🛑 Parar a aplicação

### Docker Compose
```bash
docker-compose down
```

### Aplicação local
Use `Ctrl+C` no terminal onde a aplicação está rodando.