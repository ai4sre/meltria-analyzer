.PHONY: init
init:
	poetry install

.PHONY: docker/build
docker/build:
	docker-compose build
