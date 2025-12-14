.PHONY: install run

install:
	rm -r .venv
	rm -r ml-32m*
	python -m venv .venv
	.venv/bin/pip install -r requirements.txt
	wget https://files.grouplens.org/datasets/movielens/ml-32m.zip
	unzip ml-32m.zip
	python data/initalize_datastore.py --data-dir ml-32m

run: install
	.venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000
