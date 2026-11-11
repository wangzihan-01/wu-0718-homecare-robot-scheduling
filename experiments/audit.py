"""Audit paired-day artifacts without rerunning or selecting results."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from experiments.metrics import compute_metrics
from experiments.run import load_json, save_json, sha256_json


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def audit_batch(folder):
    folder = Path(folder)
    plan = load_json(folder / 'plan.json')
    rows = list(csv.DictReader((folder / 'summary.csv').open()))
    expected = {(d['scenario'], d['seed'], a) for d in plan['days'] for a in plan['algorithms']}
    observed, issues, groups = set(), [], defaultdict(list)
    for row in rows:
        key = (row['scenario'], int(row['seed']), row['algorithm'])
        if key in observed:
            issues.append(f'duplicate run {key}')
        observed.add(key)
        # Directory basename makes the audit usable after the repo is moved.
        run = folder / 'runs' / Path(row['output_dir']).name
        manifest = load_json(run / 'manifest.json')
        trajectory = load_json(run / 'trajectory.json')
        cfg = load_json(run / 'config.json')
        shared = load_json(folder / 'trajectories' / f'{key[0]}_{key[1]}.json')
        if sha256_json(cfg) != manifest['config_sha256']:
            issues.append(f'config hash mismatch {key}')
        if sha256_json(trajectory) != manifest['trajectory_sha256'] or trajectory != shared:
            issues.append(f'trajectory mismatch {key}')
        groups[key[:2]].append(manifest['trajectory_sha256'])
        if row['run_status'] != 'completed':
            # Keep failure rows; do not audit partial-day metrics as full days.
            if manifest.get('evaluation_valid') is not False or not (run / 'error.json').exists():
                issues.append(f'failure not documented {key}')
            continue
        tasks = load_json(run / 'tasks.json')
        stats = load_json(run / 'stats.json')
        decisions = read_jsonl(run / 'decision_log.jsonl')
        actions = read_jsonl(run / 'action_log.jsonl')
        events = read_jsonl(run / 'event_log.jsonl')
        metrics = load_json(run / 'metrics.json')
        if {t['id'] for t in tasks} != {t['id'] for t in trajectory['tasks']}:
            issues.append(f'demand missing from denominator {key}')
        if len(actions) != len(decisions):
            issues.append(f'action count mismatch {key}')
        previous = 0
        for d in decisions:
            if d['time_s'] != previous or d['next_time_s'] <= previous or d['info']['elapsed_s'] != d['next_time_s']-previous:
                issues.append(f'time progression invalid {key}')
                break
            previous = d['next_time_s']
        if previous != cfg['horizon_s'] or sum(stats['mode_time_s'].values()) != cfg['horizon_s']:
            issues.append(f'full day not accounted {key}')
        if not 0 <= stats['min_energy_wh'] <= cfg['model']['capacity_wh']:
            issues.append(f'battery outside bounds {key}')
        for t in tasks:
            if t['remaining_service_s'] < 0 or (t['completed_at_s'] is not None and t['remaining_service_s'] != 0):
                issues.append(f'inconsistent service accounting {key} {t["id"]}')
            if t['response_at_s'] is not None and not t['release_time_s'] <= t['response_at_s'] <= cfg['horizon_s']:
                issues.append(f'invalid response timestamp {key} {t["id"]}')
        recomputed = compute_metrics(tasks, cfg['horizon_s'], cfg['model']['critical_priority'], decisions, stats, events)
        for field, value in recomputed.items():
            saved = metrics.get(field)
            equal = math.isclose(value, saved, abs_tol=1e-9) if isinstance(value, (int,float)) and isinstance(saved, (int,float)) else value == saved
            if not equal:
                issues.append(f'metric mismatch {key} {field}')
    if observed != expected:
        issues.append(f'missing/unexpected pairs: missing={sorted(expected-observed)} unexpected={sorted(observed-expected)}')
    for day, hashes in groups.items():
        if len(hashes) != len(plan['algorithms']) or len(set(hashes)) != 1:
            issues.append(f'unpaired trajectory {day}')
    result = {'purpose': 'artifact_integrity_check_not_research_inference', 'batch': folder.name,
              'expected_runs': len(expected), 'saved_runs': len(rows), 'paired_days': len(groups),
              'completed': sum(r['run_status'] == 'completed' for r in rows),
              'failed_or_not_run': sum(r['run_status'] != 'completed' for r in rows),
              'checks': ['unique_complete_pairs', 'identical_hidden_trajectories', 'config_hashes',
                         'full_demand_denominators', 'positive_contiguous_time', 'full_day_activity_accounting',
                         'battery_bounds', 'task_and_response_timestamps', 'recomputed_all_metrics'],
              'passed': not issues, 'issues': issues}
    save_json(folder / 'audit.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('batch')
    result = audit_batch(parser.parse_args().batch)
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
