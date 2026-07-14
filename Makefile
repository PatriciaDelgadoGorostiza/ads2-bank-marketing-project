.PHONY: install notebook test

install:
	pip install -r requirements.txt

notebook:
	jupyter lab

test:
	pytest
