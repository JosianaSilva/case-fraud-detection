.PHONY: help train deploy clean setup test

PYTHON = python3
VENV_PATH = env
VENV_PYTHON = $(VENV_PATH)/Scripts/python.exe
MODELS_DIR = models
SRC_DIR = src
SCRIPTS_DIR = $(SRC_DIR)/scripts
METRICS_FILE = models/metrics.json

help:
	@echo "Comandos disponíveis:"
	@echo "  make setup     - Configura ambiente"
	@echo "  make train     - Treina modelo"
	@echo "  make deploy    - Deploy condicional para HF"
	@echo "  make test      - Executa testes"

setup:
	$(PYTHON) -m venv $(VENV_PATH)
	$(VENV_PATH)/Scripts/activate && pip install -r requirements.txt

train:
	@mkdir -p $(MODELS_DIR)
	$(VENV_PYTHON) $(SCRIPTS_DIR)/train.py

deploy:
	$(VENV_PYTHON) $(SCRIPTS_DIR)/deploy.py

test:
	$(VENV_PYTHON) -m pytest test/ -v