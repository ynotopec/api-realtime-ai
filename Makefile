PYTHON ?= python3
HOST ?= 0.0.0.0
PORT ?= 8080

.PHONY: install run

install:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip setuptools wheel && pip install -r requirements.txt

run:
	. .venv/bin/activate && SERVER_NAME=$(HOST) SERVER_PORT=$(PORT) $(PYTHON) app.py --host $(HOST) --port $(PORT)
