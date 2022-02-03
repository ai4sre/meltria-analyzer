#!/usr/bin/env python3

import argparse

from tsdr import tsdr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_file",
                        help="metrics JSON file")
    parser.add_argument('--series-name',
                        required=True,
                        help='the name of series')
    args = parser.parse_args()

    metrics, _, _ = tsdr.read_metrics_json(args.metrics_file)
    print(metrics[args.series_name].to_list())


if __name__ == '__main__':
    main()
