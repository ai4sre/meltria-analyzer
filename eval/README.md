# eval

## Run scripts

```shell-session
$ poetry run eval/bin/eval_diagnoser.py neptune.mode='debug' metrics_files=[(find /datasets/argowf-chaos-hg68n/ -name "*.json" | grep -v network | paste -s -d ',' -)]

# filter
$ poetry run eval/bin/eval_diagnoser.py neptune.mode='debug' metrics_files=[(find /datasets/argowf-chaos-hg68n/ -name "*.json" | grep -v network | grep carts-db | grep pod-cpu-hog | paste -s -d ',' -)]
```

## Setup neptune.ai client

TBD
