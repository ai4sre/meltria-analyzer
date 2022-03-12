from enum import Enum


class MetricType(Enum):
    CONTAINER = 1
    SERVICE = 2
    NODE = 3
    MIDDLEWARE = 4


class MetricNode:
    id: int
    label: str
    comp: str
    comp_type: MetricType
    base_name: str

    # label should be like 'c-orders_cpu_usage_seconds_total'
    def __init__(self, label: str, id: int = 0) -> None:
        self.id = id
        self.label = label
        self.comp, self.base_name = label.split('-', maxsplit=1)[1].split('_', maxsplit=1)
        if label.startswith('c-'):
            self.comp_type = MetricType.CONTAINER
        elif label.startswith('s-'):
            self.comp_type = MetricType.SERVICE
        elif label.startswith('n-'):
            self.comp_type = MetricType.NODE
        elif label.startswith('m-'):
            self.comp_type = MetricType.MIDDLEWARE
        else:
            raise ValueError(f"no prefix: {label}")

    def __eq__(self, other) -> bool:
        return self.id == other.id and self.label == other.label

    def __str__(self) -> str:
        return self.label

    def is_service(self) -> bool:
        return self.comp_type == MetricType.SERVICE

    def is_container(self) -> bool:
        return self.comp_type == MetricType.CONTAINER

    def is_node(self) -> bool:
        return self.comp_type == MetricType.NODE

    def is_middleware(self) -> bool:
        return self.comp_type == MetricType.MIDDLEWARE


def metric_nodes_from_labels(labels: dict[int, str]) -> list[MetricNode]:
    return [MetricNode(v, id=i) for i, v in labels.items()]
