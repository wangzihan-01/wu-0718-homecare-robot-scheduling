"""Original project code: declared, paired scenario batches; formal execution opt-in."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from uuid import uuid4

from experiments.run import ROOT, load_json, merge_config, provenance, run_one, save_csv, save_json, sha256_json


def expand_scenarios(spec):
    if 'scenarios' in spec:
        return spec['scenarios']
    matrix = spec['matrix']
    result = {}
    for load, emergency, capacity in product(matrix['task_rate_per_hour'],
                                              matrix['emergency_rate_per_hour'], matrix['capacity_wh']):
        name = f'load{load:g}_emergency{emergency:g}_battery{capacity:g}'
        result[name] = {'scenario': {'task_rate_per_hour': load, 'emergency_rate_per_hour': emergency},
                        'model': {'capacity_wh': capacity, 'initial_energy_wh': capacity}}
    return result


def expand_days(spec):
    scenarios = expand_scenarios(spec)
    days = [(name, seed) for name in scenarios for seed in spec['seeds']]
    subset = spec.get('common_subset')
    if subset is not None:
        selected = {(item['scenario'], item['seed']) for item in subset}
        if not selected or not selected.issubset(set(days)) or len(selected) != len(subset):
            raise ValueError('common_subset must contain unique valid scenario/seed days')
        days = [day for day in days if day in selected]
    return scenarios, days


def run_batch(spec, output='results', *, allow_formal=False):
    formal = 'matrix' in spec
    scenarios, days = expand_days(spec)
    base_path = Path(spec['base_config'])
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    base = load_json(base_path)
    if formal and not allow_formal:
        raise ValueError('formal evaluation is opt-in; use --execute only after freezing parameters and budget')
    if formal and 'llm' in spec['algorithms']:
        raise ValueError('formal LLM is not configured: no live transport is implemented; mock/replay cannot qualify. No batch started.')
    if 'learned' in spec['algorithms']:
        # Fail before any days are generated or baselines run if a model is absent.
        from planners.learned import LearnedPlanner
        for overrides in scenarios.values():
            config = merge_config(base, overrides)
            learned = LearnedPlanner(config['planners'].get('learned', {}))
            if learned.test_only:
                raise ValueError('Test-only learned model cannot enter a research batch. No batch started.')
    batch_id = f'{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{spec["name"]}_{uuid4().hex[:8]}'
    folder = Path(output) / batch_id
    folder.mkdir(parents=True, exist_ok=False)
    save_json(folder / 'declaration.json', spec)
    save_json(folder / 'base_config.json', base)
    version = provenance()
    save_json(folder / 'version.json', version)
    save_json(folder / 'plan.json', {'days': [{'scenario': s, 'seed': n} for s, n in days],
                                   'algorithms': spec['algorithms'], 'expected_runs': len(days) * len(spec['algorithms']),
                                   'status': 'declared_before_execution'})
    from env.scenario import generate_trajectory
    rows, workload = [], []
    for scenario, seed in days:
        config = merge_config(base, scenarios[scenario])
        config['experiment']['purpose'] = spec['purpose']
        # This object is generated exactly once per scenario/seed, deep-copied by
        # each runner. Model algorithms never receive this future trajectory.
        trajectory = generate_trajectory(config, seed)
        day_path = folder / 'trajectories' / f'{scenario}_{seed}.json'
        day_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(day_path, trajectory)
        ordinary = [t for t in trajectory['tasks'] if t['type'] != 'emergency']
        emergencies = [t for t in trajectory['tasks'] if t['type'] == 'emergency']
        workload.append({'scenario': scenario, 'seed': seed,
                         'ordinary_task_count': len(ordinary), 'emergency_count': len(emergencies),
                         'ordinary_service_s': sum(t['service_duration_s'] for t in ordinary),
                         'service_demand_to_horizon_ratio': sum(t['service_duration_s'] for t in ordinary) / config['horizon_s'],
                         'trajectory_sha256': sha256_json(trajectory)})
        for algorithm in spec['algorithms']:
            result = run_one(config, seed, algorithm, scenario=scenario, output_root=folder / 'runs',
                             trajectory=trajectory, version=version)
            rows.append(result)
            save_csv(folder / 'summary.csv', rows)
            print(f'{len(rows)}/{len(days)*len(spec["algorithms"])} {scenario} seed={seed} {algorithm}: {result["run_status"]}', flush=True)
    save_csv(folder / 'workload.csv', workload)
    save_json(folder / 'completion.json', {'batch_id': batch_id, 'runs': len(rows),
                                         'completed': sum(r['run_status'] == 'completed' for r in rows),
                                         'failed_or_not_run': sum(r['run_status'] != 'completed' for r in rows),
                                         'purpose': spec['purpose']})
    from experiments.plot import plot_summary
    plot_summary(folder / 'summary.csv', folder / 'smoke_overview.png')
    print(f'Batch artifacts: {folder.resolve()}')
    return folder, rows


def main():
    parser = argparse.ArgumentParser(description='Smoke batch or dry-run formal plan; all days are paired.')
    parser.add_argument('--config', default='configs/smoke.json')
    parser.add_argument('--output', default='results')
    parser.add_argument('--execute', action='store_true', help='Explicitly enable formal matrix after parameter/budget freeze')
    args = parser.parse_args()
    spec = load_json(args.config)
    if 'matrix' in spec and not args.execute:
        _, days = expand_days(spec)
        print(f'DRY RUN ONLY: {len(days)} days/method x {len(spec["algorithms"])} methods = {len(days)*len(spec["algorithms"])} runs.')
        print('Formal evaluation has NOT run. Configure LLM and freeze parameters, budget and common subset first.')
        return 0
    folder, rows = run_batch(spec, args.output, allow_formal=args.execute)
    return 0 if all(r['run_status'] == 'completed' for r in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
