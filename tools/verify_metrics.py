#!/usr/bin/env python3

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from enum import Enum

import lib.metrics
import numpy as np
import pandas as pd
import ruptures as rpt
from scipy import interpolate
from statsmodels.tsa import stattools
from tsdr import tsdr

TIME_INTERVAL_SEC = 15


class BkpsStatus(int, Enum):
    NOT_FOUND = 1
    FOUND_OUTSIDE_OF_CHAOS = 2
    FOUND_INSIDE_OF_CHAOS = 3


def detect_bkps(samples: pd.Series, n_bkps=2, model='l2', chaos_duration_min=5, adf_alpha=0.05) -> BkpsStatus:
    samples = samples.interpolate(method="spline", order=3, limit_direction="both")
    return _detect_bkps(samples.to_numpy(), n_bkps, model, chaos_duration_min, adf_alpha)


def _detect_bkps(samples: np.ndarray, n_bkps, model, chaos_duration_min, adf_alpha) -> BkpsStatus:
    """detect breaking points
    1. Check stationality with ADF test
    2. Search breaking poitnts with ruptures binary segmentation
    """

    p_val = stattools.adfuller(samples)[1]
    if p_val <= adf_alpha:
        logging.info(p_val)
        return BkpsStatus.NOT_FOUND

    algo = rpt.Binseg(model=model).fit(samples)
    try:
        result: list[int] = algo.predict(n_bkps=n_bkps)
    except rpt.exceptions.BadSegmentationParameters:
        logging.info('rpt.Binseq.predict BadSegmentationParameters suggests not found bkps')
        return BkpsStatus.NOT_FOUND

    if result[0] == len(samples):  # not found breaking points
        return BkpsStatus.NOT_FOUND
    chaos_plots: int = chaos_duration_min * 60//TIME_INTERVAL_SEC
    chaos_injected_pt: int = len(samples) - chaos_plots
    for bkp in result:
        if bkp < chaos_injected_pt:
            return BkpsStatus.FOUND_OUTSIDE_OF_CHAOS
    return BkpsStatus.FOUND_INSIDE_OF_CHAOS


def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("metricsfiles",
                        nargs='+',
                        help="metrics output JSON file")
    parser.add_argument('--out', help='output file path')
    parser.add_argument('--out-format',
                        choices=['json', 'csv'],
                        default='json',
                        help='output format')
    args = parser.parse_args()

    results = defaultdict(lambda: list())
    for metrics_file in args.metricsfiles:
        data_df, _, metrics_meta = tsdr.read_metrics_json(metrics_file, interporate=False)
        chaos_type: str = metrics_meta['injected_chaos_type']
        chaos_comp: str = metrics_meta['chaos_injected_component']
        dashboard_url: str = metrics_meta['grafana_dashboard_url']
        case: str = f"{chaos_type}:{chaos_comp}"

        logging.info(f">> Running verify_metrics {metrics_file} {case} ...")

        sli = 's-front-end_latency'
        sli_status = detect_bkps(data_df[sli])

        _, cause_metrics = lib.metrics.check_cause_metrics(list(data_df.columns), chaos_type, chaos_comp)
        cause_metrics_series = data_df[cause_metrics]
        cause_metrics_status = {}
        for feature, samples in cause_metrics_series.items():
            status = detect_bkps(samples)
            cause_metrics_status[feature] = status

        service_name = chaos_comp.split('-')[0]
        service_sli = f"s-{service_name}_latency"
        service_sli_status = detect_bkps(data_df[service_sli]) if service_sli in data_df else None

        results[case].append({
            'sli': {sli: sli_status},
            'cause_metrics': cause_metrics_status,
            'service_sli': {service_sli: service_sli_status},
            'dashboard_url': dashboard_url,
        })

    if args.out_format == 'json':
        if args.out is None:
            json.dump(results, sys.stdout, indent=4)
        else:
            with open(args.out, mode='w') as f:
                json.dump(results, f)
    elif args.out_format == 'csv':
        if args.out is not None:
            sys.stdout = open(args.out, 'w', newline='')
        writer = csv.writer(sys.stdout, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # case, no, metric_type, metric_name, sttaus(str), status(int), dashboard_url
        for case, entries in results.items():
            for i, entry in enumerate(entries):
                for metric_type in ['sli', 'service_sli', 'cause_metrics']:
                    for metric_name, status in entry[metric_type].items():
                        writer.writerow([
                            case, str(i), metric_type, metric_name, status.name, status.value, entry['dashboard_url']
                        ])

        sys.stdout = sys.__stdout__
    else:
        pass


if __name__ == '__main__':
    main()
