"""Original project code: explicit proxy estimates, no model/price assumptions."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def estimate(rows, *, evaluation_days=540, max_retries=1, input_tokens=None, output_tokens=None,
             input_price=None, output_price=None):
    complete = [r for r in rows if r.get('run_status') == 'completed' and r.get('decision_count') not in (None, '')]
    counts = [int(r['decision_count']) for r in complete]
    if not counts:
        raise ValueError('No completed days with decision counts')
    mean_calls = mean(counts)
    call_range = [min(counts) * evaluation_days, max(counts) * evaluation_days * (1 + max_retries)]
    result = {'purpose': 'prebudget_proxy_not_llm_measurement', 'reference_days': len(counts),
              'evaluation_days_per_method': evaluation_days, 'max_retries': max_retries,
              'reference_decisions_per_day_mean': mean_calls,
              'reference_decisions_per_day_min': min(counts), 'reference_decisions_per_day_max': max(counts),
              'estimated_calls_without_retries_at_mean': mean_calls * evaluation_days,
              'proxy_call_envelope': call_range, 'input_tokens_per_call_assumption': input_tokens,
              'output_tokens_per_call_assumption': output_tokens,
              'input_price_per_million_assumption': input_price, 'output_price_per_million_assumption': output_price,
              'estimated_cost_at_mean_no_retry': None, 'proxy_cost_envelope': None,
              'caveat': 'LLM actions can change decisions/day. These are observed rule/optimizer proxies, not upper guarantees. Prices must be supplied and verified by the user; no API has been called.'}
    if all(value is not None for value in (input_tokens, output_tokens, input_price, output_price)):
        if min(input_tokens, output_tokens, input_price, output_price) < 0:
            raise ValueError('Token/price assumptions must be nonnegative')
        cost_call = (input_tokens * input_price + output_tokens * output_price) / 1e6
        result['estimated_cost_at_mean_no_retry'] = cost_call * mean_calls * evaluation_days
        result['proxy_cost_envelope'] = [v * cost_call for v in call_range]
    return result


def main():
    parser = argparse.ArgumentParser(description='Estimate call-volume proxy; price is never assumed.')
    parser.add_argument('summary_csv')
    parser.add_argument('--evaluation-days', type=int, default=540)
    parser.add_argument('--max-retries', type=int, default=1)
    parser.add_argument('--input-tokens-per-call', type=int)
    parser.add_argument('--output-tokens-per-call', type=int)
    parser.add_argument('--input-price-per-million', type=float)
    parser.add_argument('--output-price-per-million', type=float)
    parser.add_argument('--output')
    args = parser.parse_args()
    with Path(args.summary_csv).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    result = estimate(rows, evaluation_days=args.evaluation_days, max_retries=args.max_retries,
                      input_tokens=args.input_tokens_per_call, output_tokens=args.output_tokens_per_call,
                      input_price=args.input_price_per_million, output_price=args.output_price_per_million)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        Path(args.output).write_text(payload, encoding='utf-8')
    print(payload)


if __name__ == '__main__':
    main()
