pre-commit run --all-files
mypy src
pylama --ignore="E501,E722,C901" --skip="venv/*"
