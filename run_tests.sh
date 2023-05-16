set -e

flake8 ./star_analysis/ ./tests/ --count --show-source --statistics --ignore=E501,W503
pytest
