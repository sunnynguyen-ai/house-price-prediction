.PHONY: install train test run clean

install:
	pip install -r requirements.txt

train:
	python src/model_training.py

test:
	python -m pytest tests/ -v

run:
	python app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
