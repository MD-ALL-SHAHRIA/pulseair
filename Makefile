# One command to reproduce the pipeline. This is a thin wrapper over
# `python -m src.reproduce_all`, which runs the project's existing steps in order.
# See `make list` for the full plan.

PYTHON ?= python

.PHONY: all reporting fast setup test list clean-figures

all:            ## Full pipeline, end to end (assumes deps installed; see `make setup`)
	$(PYTHON) -m src.reproduce_all

setup:          ## Install the package and test/reporting dependencies
	$(PYTHON) -m pip install -e ".[test]"
	$(PYTHON) -m pip install matplotlib seaborn python-docx pyyaml joblib

fast:           ## Pipeline without the multi-hour or network steps
	$(PYTHON) -m src.reproduce_all --skip-slow --skip-network --skip-optional

reporting:      ## Recompile results, figures and the thesis from committed metrics only
	$(PYTHON) -m src.reproduce_all --only-reporting

list:           ## Print the ordered pipeline plan and exit
	$(PYTHON) -m src.reproduce_all --list

test:           ## Run the test suites
	$(PYTHON) -m pytest pulsebench/tests src/preprocessing src/reporting/test_generate_figures.py src/reporting/test_build_thesis.py src/models/test_phase_additions.py -q

help:           ## Show these targets
	@grep -E '^[a-z-]+:.*##' Makefile | sed -E 's/:.*## /\t/' | sort
