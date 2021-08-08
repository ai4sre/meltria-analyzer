import re
from typing import Any, Tuple

CHAOS_TO_CAUSE_METRIC_PREFIX = {
    'pod-cpu-hog': 'cpu_',
    'pod-memory-hog': 'memory_',
    'pod-network-loss': 'network_',
    'pod-network-latency': 'network_',
}


def check_cause_metrics(metrics: list[str], chaos_type: str, chaos_comp: str
                        ) -> Tuple[bool, list[Any]]:
    prefix = CHAOS_TO_CAUSE_METRIC_PREFIX[chaos_type]
    cause_metrics = []
    for metric in metrics:
        if re.match(f"^c-{chaos_comp}_{prefix}.+", metric):
            cause_metrics.append(metric)
    if len(cause_metrics) > 0:
        return True, cause_metrics
    return False, cause_metrics
