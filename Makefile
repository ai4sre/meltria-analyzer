.PHONY: init
init:
	poetry install

.PHONY: test
test:
	PYTHONPATH=. poetry run pytest -s -vv tests

.PHONY: docker/build
docker/build:
	docker-compose build
