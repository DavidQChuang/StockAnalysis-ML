lint: lint-src lint-tests

lint-%:
	uvx ruff format --diff $* && \
	uvx ruff check $*


format: format-src format-tests

format-%:
	uvx ruff format $* && \
	uvx ruff check --fix $*

format-unsafe:
	uvx ruff format src && \
	uvx ruff check --fix --unsafe-fixes src


test:
	uvx coverage run -m unittest

sync:
	uv sync --all-extras