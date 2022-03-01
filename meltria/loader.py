import logging
import os
from concurrent import futures
from multiprocessing import cpu_count
from typing import Optional

import pandas as pd
from tsdr import tsdr


class DatasetRecord:
    """A record of dataset"""
    chaos_comp: str     # chaos-injected component
    chaos_type: str     # injected chaos type
    metrics_file: str   # path of metrics file
    data_df: pd.DataFrame

    def __init__(self, chaos_type: str, chaos_comp: str, metrics_file: str, data_df: pd.DataFrame):
        self.chaos_comp = chaos_comp
        self.chaos_type = chaos_type
        self.metrics_file = metrics_file
        self.data_df = data_df

    def chaos_case(self) -> str:
        return f"{self.chaos_comp}/{self.chaos_type}"

    def chaos_case_file(self) -> str:
        return f"{self.metrics_file} of {self.chaos_case()}"

    def metrics_names(self) -> list[str]:
        return list(self.data_df.columns)


def load_dataset(metrics_files: list[str], exclude_middleware_metrics: bool = False) -> pd.DataFrame:
    """ Load metrics dataset
    """
    dataset = pd.DataFrame()
    with futures.ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        future_list = []
        for metrics_file in metrics_files:
            future_list.append(executor.submit(read_metrics_file, metrics_file, exclude_middleware_metrics))
        for future in futures.as_completed(future_list):
            data_df = future.result()
            if data_df is not None:
                dataset = dataset.append(data_df)
    return dataset.set_index(['chaos_type', 'chaos_comp', 'metrics_file', 'grafana_dashboard_url'])


def read_metrics_file(
    metrics_file: str,
    exclude_middleware_metrics: bool = False,
    logger: logging.Logger = logging.getLogger(),
) -> Optional[pd.DataFrame]:
    try:
        data_df, _, metrics_meta = tsdr.read_metrics_json(
            metrics_file,
            exclude_middlewares=exclude_middleware_metrics,
        )
    except ValueError as e:
        logger.warning(f">> Skip {metrics_file} because of {e}")
        return None
    chaos_type: str = metrics_meta['injected_chaos_type']
    chaos_comp: str = metrics_meta['chaos_injected_component']
    data_df['chaos_type'] = chaos_type
    data_df['chaos_comp'] = chaos_comp
    data_df['metrics_file'] = os.path.basename(metrics_file)
    data_df['grafana_dashboard_url'] = metrics_meta['grafana_dashboard_url']
    return data_df
