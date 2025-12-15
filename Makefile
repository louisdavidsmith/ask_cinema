.PHONY: install run-endpoint run-streamlit

DATA_DIR := ml-32m
ZIP_FILE := ml-32m.zip

install: .venv
	.venv/bin/pip install -r requirements.txt
	test -d $(DATA_DIR) || (wget https://files.grouplens.org/datasets/movielens/$(ZIP_FILE) && unzip $(ZIP_FILE))
	test -d $(DATA_DIR) && python data/initialize_datastore.py --data-dir $(DATA_DIR)

.venv:
	rm -rf .venv
	python -m venv .venv

run-endpoint: install
	.venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000

run-streamlit: install
	.venv/bin/streamlit run chat.py

assess: install
	.venv/bin/python assess_performance.py
