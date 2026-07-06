.PHONY: install load run clean

VENV := .venv
PYTHON := $(VENV)/bin/python
REQUIREMENTS := requirements.txt

install: $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r $(REQUIREMENTS)

$(VENV):
	python -m venv $(VENV)

load:
	@echo "Run 'source $(VENV)/bin/activate' to load the virtualenv"

run: $(VENV)
	$(PYTHON) -m streamlit run app.py

clean:
	rm -rf $(VENV) __pycache__ *.pyc .pytest_cache .mypy_cache
