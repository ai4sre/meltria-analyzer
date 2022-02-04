#!/usr/bin/env python3

import argparse
from pprint import pprint

from tsdr import tsdr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-file",
                        help="metrics JSON file")
    parser.add_argument('series_names',
                        help='the list of series')
    args = parser.parse_args()

    metrics, _, _ = tsdr.read_metrics_json(args.metrics_file)
    series_by_name = {}
    for name in args.series_names.split(","):
        series_by_name[name] = metrics[name].to_list()
    pprint(series_by_name, compact=True)


if __name__ == '__main__':
    main()
