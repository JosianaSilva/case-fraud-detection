# Case - Detecção de fraudes

API para predição de fraudes usando modelo de Machine Learning.

## Configuração do Ambiente

1. Criar ambiente virtual:
```bash
python -m venv env
```

2. Ativar o ambiente virtual:
```bash
# Windows
env\Scripts\activate

# Linux/Mac
source env/bin/activate
```

3. Instalar dependências:
```bash
pip install -r requirements.txt
```

## Execução

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```