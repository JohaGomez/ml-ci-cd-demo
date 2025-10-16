# Makefile
USER_NAME ?= "Johana Gomez"
USER_EMAIL ?= "johana.gomez@universidadean.edu.co"
HF ?= $(HF_TOKEN)
HF_SPACE ?= gomez-joha-2025-ean/Drug-Classification 
# ========== CONFIGURACIÓN BASE ==========
PYTHON := python

# ========== OBJETIVOS ==========
install:
	$(PYTHON) -m pip install -r requirements.txt


format:
	$(PYTHON) -m black *.py

train:
	$(PYTHON) train.py

eval:
	$(PYTHON) evaluate.py

report:
	echo "## Model Metrics" > report.md
	type .\Results\metrics.txt >> report.md
	echo. >> report.md
	echo "## Confusion Matrix Plot" >> report.md
	echo ![Confusion Matrix](./Results/model_results.png) >> report.md

all: install format train eval report

# ======= HUGGING FACE DEPLOY =======

HF ?= $(HF_TOKEN)
HF_SPACE ?= gomez-joha-2025-ean/Drug-Classification

hf-login:
	$(PYTHON) -m pip install -U "huggingface_hub[cli]"
	hf auth login --token $(HF)

push-hub:
	hf upload $(HF_SPACE) ./App --repo-type=space --commit-message="Sync App files"
	hf upload $(HF_SPACE) ./Model /Model --repo-type=space --commit-message="Sync Model"
	hf upload $(HF_SPACE) ./Results /Metrics --repo-type=space --commit-message="Sync Metrics"

deploy: hf-login push-hub
