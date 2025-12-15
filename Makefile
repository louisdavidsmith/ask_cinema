.PHONY: run chat assess clean

DATA_DIR := ml-32m
ZIP_FILE := ml-32m.zip
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

VENV_SENTINEL := $(VENV)/.installed
DATA_SENTINEL := data/.initialized

$(VENV):
	rm -rf $(VENV)
	python -m venv $(VENV)

$(VENV_SENTINEL): requirements.txt | $(VENV)
	$(PIP) install -r requirements.txt
	touch $(VENV_SENTINEL)

$(DATA_DIR):
	wget https://files.grouplens.org/datasets/movielens/$(ZIP_FILE)
	unzip $(ZIP_FILE)

$(DATA_SENTINEL): $(DATA_DIR) $(VENV_SENTINEL)
	$(PYTHON) data/initialize_datastore.py --data-dir $(DATA_DIR)
	touch $(DATA_SENTINEL)

install: $(VENV_SENTINEL) $(DATA_SENTINEL)
	@echo "Install complete."

run: install
	$(VENV)/bin/uvicorn server:app --host 0.0.0.0 --port 8000

chat: install
	$(VENV)/bin/streamlit run chat.py

assess: install
	$(PYTHON) assess_performance.py --all

clean:
	rm -rf $(VENV) $(DATA_SENTINEL)

