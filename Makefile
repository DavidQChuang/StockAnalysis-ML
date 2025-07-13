lint:
	uvx ruff format --diff src && \
	uvx ruff check src

format:
	uvx ruff format src && \
	uvx ruff check --fix src

test:
	uvx coverage run -m unittest

sync:
	uv sync --all-extras