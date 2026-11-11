"""Original project code: paired simulation-day confidence intervals."""
import argparse
import csv
import json
from pathlib import Path
from experiments.metrics import paired_bootstrap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('summary_csv')
    parser.add_argument('--metric', default='critical_ontime_rate')
    parser.add_argument('--a', required=True)
    parser.add_argument('--b', required=True)
    parser.add_argument('--resamples', type=int, default=2000)
    parser.add_argument('--confidence', type=float, default=.95)
    parser.add_argument('--seed', type=int, default=5301)
    args = parser.parse_args()
    with Path(args.summary_csv).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    result = paired_bootstrap(rows, args.metric, args.a, args.b, confidence=args.confidence,
                              n_resamples=args.resamples, seed=args.seed)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
