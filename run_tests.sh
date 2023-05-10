set -e

flake8 ./src/ ./tests/ --count --show-source --statistics 
--ignore=E501,W503
mypy ./src/ ./tests/
pytest
