.PHONY: install install-dev check test run serve

install:
	python3 -m pip install -r requirements.txt

install-dev:
	python3 -m pip install -r requirements-dev.txt

check:
	python3 -m compileall -q app.py config.py pipeline tests
	python3 -m pytest

test:
	python3 -m pytest

run:
	python3 app.py

serve:
	gunicorn --workers 1 --threads 4 --timeout 0 --bind 0.0.0.0:5003 app:app
