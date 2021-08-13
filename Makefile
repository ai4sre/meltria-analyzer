.PHONY: init
init:
	poetry install

.PHONY: test
test:
	PYTHONPATH=. poetry run pytest tests

.PHONY: docker/build
docker/build:
	docker-compose build
