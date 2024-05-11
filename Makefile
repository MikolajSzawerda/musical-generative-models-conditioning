.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = conditioning-of-musical-generative-models
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

POETRY_TOOLS_ENV := $(shell cd environ/tools && poetry env info --path)
POETRY_AUDIOCRAFT_ENV := $(shell cd environ/audiocraft && poetry env info --path)
POETRY_MAIN_ENV := $(shell cd environ/main && poetry env info --path)

VSCODE_SETTINGS_DIR := .vscode
VSCODE_SETTINGS_PATH := $(VSCODE_SETTINGS_DIR)/settings.json

install-tools-env:
	@echo "Installing dependencies for the tools environment..."
	@cd environ/tools && poetry install
	@echo "Dependencies installed."

install-audiocraft-env:
	@echo "Installing dependencies for the audiocraft environment..."
	@cd environ/audiocraft && poetry install
	@poetry run pip install torch
	@poetry run python -m ipykernel install --user --name audiocraft_lab --display-name "Python Audiocraft"
	@echo "Dependencies installed."

install-main-env:
	@echo "Installing dependencies for the main environment..."
	@cd environ/main && poetry install
	@echo "Dependencies installed."

vc-env-tools: install-tools-env
	@echo "Setting up VS Code to use tools environment..."
	@mkdir -p $(VSCODE_SETTINGS_DIR)
	@echo '{' > $(VSCODE_SETTINGS_PATH)
	@echo '  "python.defaultInterpreterPath": "$(POETRY_TOOLS_ENV)/bin/python",' >> $(VSCODE_SETTINGS_PATH)
	@echo '  "python.terminal.activateEnvironment": true,' >> $(VSCODE_SETTINGS_PATH)
	@echo '  "python.envFile": "${workspaceFolder}/.env"' >> $(VSCODE_SETTINGS_PATH)
	@echo '}' >> $(VSCODE_SETTINGS_PATH)
	@echo "VS Code is now configured to use the tools environment."

vc-env-audiocraft: install-audiocraft-env
	@echo "Setting up VS Code to use audiocraft environment..."
	@mkdir -p $(VSCODE_SETTINGS_DIR)
	@echo '{' > $(VSCODE_SETTINGS_PATH)
	@echo '  "python.defaultInterpreterPath": "$(POETRY_AUDIOCRAFT_ENV)/bin/python",' >> $(VSCODE_SETTINGS_PATH)
	@echo '  "python.terminal.activateEnvironment": true,' >> $(VSCODE_SETTINGS_PATH)
	@echo '  "python.envFile": "${workspaceFolder}/.env"' >> $(VSCODE_SETTINGS_PATH)
	@echo '}' >> $(VSCODE_SETTINGS_PATH)
	@echo "VS Code is now configured to use the audiocraft environment."

vc-env-main: install-main-env
	@echo "Setting up VS Code to use main environment..."
	@mkdir -p $(VSCODE_SETTINGS_DIR)
	@echo '{' > $(VSCODE_SETTINGS_PATH)
	@echo '  "python.defaultInterpreterPath": "$(POETRY_MAIN_ENV)/bin/python",' >> $(VSCODE_SETTINGS_PATH)
	@echo '  "python.terminal.activateEnvironment": true,' >> $(VSCODE_SETTINGS_PATH)
	@echo '  "python.envFile": "${workspaceFolder}/.env"' >> $(VSCODE_SETTINGS_PATH)
	@echo '}' >> $(VSCODE_SETTINGS_PATH)
	@echo "VS Code is now configured to use the main environment."

sh-env-tools:
	@echo "Setting current shell to use tools environment..."
	@cd environ/tools && poetry shell
	@echo "Shell now uses tools env"

sh-env-audiocraft:
	@echo "Setting current shell to use audiocraft environment..."
	@cd environ/audiocraft && poetry shell
	@echo "Shell now uses audiocraft env"

sh-env-main:
	@echo "Setting current shell to use main environment..."
	@cd environ/main && poetry shell
	@echo "Shell now uses main env"



#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	@bash -c "poetry shell"
	@bash -c "poetry install"

## Make Dataset
data: requirements
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw data/processed

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Upload Data to S3
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

## Download Data from S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif

## Set up python interpreter environment
create_environment:
	@bash -c "poetry shell"
	@echo ">>> Running poetry\n"
	@bash -c "poetry install"

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
