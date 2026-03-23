SHELL := /bin/bash

PROJECT_NAME := $(notdir $(CURDIR))
VENV_DIR ?= $(HOME)/venv/$(PROJECT_NAME)
UV_PROJECT_ENVIRONMENT := $(VENV_DIR)
RUN_DIR := .run
PROXY_PID_FILE := $(RUN_DIR)/proxy.pid
STATIC_PID_FILE := $(RUN_DIR)/static.pid
PROXY_LOG := $(RUN_DIR)/proxy.log
STATIC_LOG := $(RUN_DIR)/static.log

IP ?= $(or $(word 2,$(MAKECMDGOALS)),127.0.0.1)
PORT ?= $(or $(word 3,$(MAKECMDGOALS)),8000)
STATIC_PORT ?= $(or $(word 4,$(MAKECMDGOALS)),5173)

.PHONY: install start stop upgrade status help

help:
	@echo "make install                  # create ~/venv/$(PROJECT_NAME) and install deps via uv"
	@echo "make start [IP] [PORT] [WEB]  # start proxy + static server in background"
	@echo "make stop                     # stop background processes"
	@echo "make upgrade                  # upgrade dependencies in venv"
	@echo "make status                   # show running/stopped state"

install:
	@mkdir -p "$(RUN_DIR)"
	@command -v uv >/dev/null 2>&1 || python3 -m pip install --user uv
	@UV_PROJECT_ENVIRONMENT="$(UV_PROJECT_ENVIRONMENT)" uv sync --project .
	@cp -n .env.example .env || true
	@echo "Installed in $(VENV_DIR)"

start: install
	@mkdir -p "$(RUN_DIR)"
	@if [ -f "$(PROXY_PID_FILE)" ] && kill -0 $$(cat "$(PROXY_PID_FILE)") 2>/dev/null; then \
		echo "Proxy already running (PID $$(cat $(PROXY_PID_FILE)))."; \
	else \
		set -a; [ -f .env ] && source .env; set +a; \
		HOST="$(IP)" PORT="$(PORT)" nohup "$(VENV_DIR)/bin/python" proxy.py > "$(PROXY_LOG)" 2>&1 & echo $$! > "$(PROXY_PID_FILE)"; \
		echo "Proxy started on $(IP):$(PORT) (PID $$(cat $(PROXY_PID_FILE)))."; \
	fi
	@if [ -f "$(STATIC_PID_FILE)" ] && kill -0 $$(cat "$(STATIC_PID_FILE)") 2>/dev/null; then \
		echo "Static server already running (PID $$(cat $(STATIC_PID_FILE)))."; \
	else \
		nohup "$(VENV_DIR)/bin/python" -m http.server "$(STATIC_PORT)" --bind "$(IP)" > "$(STATIC_LOG)" 2>&1 & echo $$! > "$(STATIC_PID_FILE)"; \
		echo "Static server started at http://$(IP):$(STATIC_PORT) (PID $$(cat $(STATIC_PID_FILE)))."; \
	fi

stop:
	@for pf in "$(PROXY_PID_FILE)" "$(STATIC_PID_FILE)"; do \
		if [ -f "$$pf" ]; then \
			pid=$$(cat "$$pf"); \
			if kill -0 $$pid 2>/dev/null; then kill $$pid && echo "Stopped $$pf ($$pid)."; else echo "Not running: $$pf ($$pid)."; fi; \
			rm -f "$$pf"; \
		else \
			echo "Missing $$pf"; \
		fi; \
	done

upgrade: install
	@UV_PROJECT_ENVIRONMENT="$(UV_PROJECT_ENVIRONMENT)" uv lock --project . --upgrade
	@UV_PROJECT_ENVIRONMENT="$(UV_PROJECT_ENVIRONMENT)" uv sync --project .
	@echo "Dependencies upgraded."

status:
	@for name in proxy static; do \
		pf="$(RUN_DIR)/$$name.pid"; \
		if [ -f "$$pf" ] && kill -0 $$(cat "$$pf") 2>/dev/null; then \
			echo "$$name: running (PID $$(cat $$pf))"; \
		else \
			echo "$$name: stopped"; \
		fi; \
	done

%:
	@:
