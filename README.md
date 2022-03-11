# meltria-analyzer

This project includes scripts for analyzing data collected from [Meltria](https://github.com/ai4sre/meltria).

## Setup

```shell-session
$ pip install poetry
$ poetry install
$ poetry shell
```

## Directory Layout

```
├── diagnoser	# A package of diagnosis of a failure
├── eval	# Evaluation scripts and utilities
├── meltria	# Meltria utilities
├── notebooks	# Jupyter notebooks
├── tests   	# Python test code
├── tools	# Small util scripts
└── tsdr	# A package of time series reduction
```

## Evaluation

See [doc](./eval/README.md).
