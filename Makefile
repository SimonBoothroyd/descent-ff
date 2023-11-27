PACKAGE_NAME  := descent-ff
CONDA_ENV_RUN := conda run --no-capture-output --name $(PACKAGE_NAME)

.PHONY: env lint format

env:
	mamba create     --name $(PACKAGE_NAME)
	mamba env update --name $(PACKAGE_NAME) --file environment.yml
	$(CONDA_ENV_RUN) pre-commit install || true

lint:
	$(CONDA_ENV_RUN) pre-commit run --all-files

format:
	$(CONDA_ENV_RUN) ruff format .
