.PHONY: help venv install data-loading preprocess sequence-generation train-lstm train-transformer train-final pre-failure-score-v2 recalibrate-final finalize-metrics compute-roc-auc explain-shap explain-attention pipeline explain freeze clean

VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# Windows: run `make PYTHON=venv\Scripts\python.exe PIP=venv\Scripts\pip.exe <target>`

help:
	@echo "Setup"
	@echo "  make venv                  create the virtual environment"
	@echo "  make install               install the libraries into venv"
	@echo ""
	@echo "Pipeline (run in this order, or use make pipeline)"
	@echo "  make data-loading          check that the IoT-23 files load"
	@echo "  make preprocess            clean, encode and scale the benign data"
	@echo "  make sequence-generation   build the benign sequences"
	@echo "  make train-lstm            5-fold training, LSTM Autoencoder"
	@echo "  make train-transformer     5-fold training, Transformer Autoencoder"
	@echo "  make train-final           final models on the 70/30 split"
	@echo "  make pre-failure-score-v2  score the malware sequences"
	@echo "  make recalibrate-final     thresholds from the final model benign errors"
	@echo "  make finalize-metrics      precision, recall, F1 and false positive rate"
	@echo "  make compute-roc-auc       ROC AUC and ROC curve"
	@echo "  make explain-shap          SHAP feature importance"
	@echo "  make explain-attention     attention figures"
	@echo ""
	@echo "Shortcuts"
	@echo "  make pipeline              run every step above in order"
	@echo "  make explain               run SHAP and attention only"
	@echo "  make freeze                save the installed versions to requirements.txt"
	@echo "  make clean                 remove venv and cache files"

venv:
	python3 -m venv $(VENV)
	@echo "Virtual environment created. Activate with:"
	@echo "  source $(VENV)/bin/activate   (macOS/Linux)"
	@echo "  $(VENV)\\Scripts\\activate      (Windows)"

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

data-loading:
	$(PYTHON) src/data_loading.py

preprocess:
	$(PYTHON) src/preprocessing.py

sequence-generation:
	$(PYTHON) src/sequence_generation.py

train-lstm:
	$(PYTHON) src/models/lstm_auto_encoder.py

train-transformer:
	$(PYTHON) src/models/transformer_auto_encoder.py

train-final:
	$(PYTHON) src/train_final.py

pre-failure-score-v2:
	$(PYTHON) src/pre_failure_score_v2.py

recalibrate-final:
	$(PYTHON) src/recalibrate_final.py

finalize-metrics:
	$(PYTHON) src/finalize_metrics.py

compute-roc-auc:
	$(PYTHON) src/compute_roc_auc.py

explain-shap:
	$(PYTHON) src/explainability_shap.py

explain-attention:
	$(PYTHON) src/explainability_attention.py

explain: explain-shap explain-attention

pipeline: data-loading preprocess sequence-generation train-lstm train-transformer train-final pre-failure-score-v2 recalibrate-final finalize-metrics compute-roc-auc explain

freeze:
	$(PIP) freeze > requirements.txt

clean:
	rm -rf $(VENV) .pytest_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.pyc" -delete