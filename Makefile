.PHONY: train run lint test

train:
	python models/train_insurance.py
	python models/train_diabetes.py
	python models/rf_feature_importance.py

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

lint:
	python -m pip install ruff && ruff check .

test:
	pytest -q
