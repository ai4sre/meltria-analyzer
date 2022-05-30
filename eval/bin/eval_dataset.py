#!/usr/bin/env python3

import logging
import os
from typing import Any

import hydra
import meltria.loader
import neptune.new as neptune
import pandas as pd
from eval import groundtruth, priorknowledge
from meltria.loader import DatasetRecord
from neptune.new.integrations.python_logger import NeptuneHandler
from omegaconf import DictConfig, OmegaConf
from tsdr import tsdr
from tsdr.outlierdetection.n_sigma_rule import detect_with_n_sigma_rule

# see https://docs.neptune.ai/api-reference/integrations/python-logger
logger = logging.getLogger('eval_dataset')
logger.setLevel(logging.INFO)


def validate_anomalie_range_in_sli(slis: pd.DataFrame, fi_time: int) -> dict[str, Any]:
    """ Evaluate the range of anomalies in SLI metrics
    """
    slis_anomalies_range = slis.apply(
        lambda X: detect_with_n_sigma_rule(X, test_start_time=fi_time).size != 0
    )
    return slis_anomalies_range.to_dict()


def eval_dataset(run: neptune.Run, cfg: DictConfig) -> None:
    """ Evaluate a dataset
    """
    dataset: pd.DataFrame = meltria.loader.load_dataset(
        cfg.metrics_files,
        cfg.exclude_middleware_metrics,
    )[0]
    logger.info("Dataset loading complete")

    sli_anomalies: list[dict[str, Any]] = []
    for (chaos_type, chaos_comp), sub_df in dataset.groupby(level=[0, 1]):
        for (metrics_file, grafana_dashboard_url), data_df in sub_df.groupby(level=[2, 3]):
            record = DatasetRecord(chaos_type, chaos_comp, metrics_file, data_df)

            # evaluate the positions of anomalies in SLI metrics
            slis = data_df.loc[:, data_df.columns.intersection(set(priorknowledge.ROOT_METRIC_LABELS))]
            res = validate_anomalie_range_in_sli(slis, fi_time=cfg.time.fault_inject_time_index)
            sli_anomalies.append(dict({
                'chaos_type': record.chaos_type,
                'chaos_comp': record.chaos_comp,
                'metrics_file': record.metrics_file,
            }, **res))

    sli_df = pd.DataFrame(sli_anomalies).set_index(['chaos_type', 'chaos_comp', 'metrics_file'])

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        logger.info("\n"+sli_df.to_string())


@hydra.main(config_path='../conf/dataset', config_name='config')
def main(cfg: DictConfig) -> None:
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)

    # Setup neptune.ai client
    run: neptune.Run = neptune.init(
        project=os.environ['TSDR_NEPTUNE_PROJECT'],
        api_token=os.environ['TSDR_NEPTUNE_API_TOKEN'],
        mode=cfg.neptune.mode,
    )
    npt_handler = NeptuneHandler(run=run)
    logger.addHandler(npt_handler)

    run['dataset/id'] = cfg.dataset_id
    run['dataset/num_metrics_files'] = len(cfg.metrics_files)
    params = {
        'exclude_middleware_metrics': cfg.exclude_middleware_metrics,
    }

    # Hydra parameters are passed to the Neptune.ai run object
    params.update(OmegaConf.to_container(cfg, resolve=True))

    run['parameters'] = params
    run.wait()  # sync parameters for 'async' neptune mode

    logger.info(OmegaConf.to_yaml(cfg))

    eval_dataset(run, cfg)

    run.stop()


if __name__ == '__main__':
    main()
