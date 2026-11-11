"""Original public-state demonstrations and CPU behavior cloning; execution is opt-in."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

from experiments.run import ROOT, load_json, merge_config, provenance, save_json, save_jsonl, sha256_json
from planners.learned import FEATURES, SCHEMA, feature_rows, fit_weights, probabilities


def validate_spec(spec):
    groups = [set(spec[k]) for k in ('training_seeds', 'validation_seeds', 'reserved_evaluation_seeds')]
    if (any(not g for g in groups) or any(type(s) is not int for g in groups for s in g)
            or any(groups[i] & groups[j] for i in range(3) for j in range(i))):
        raise ValueError('training, validation and evaluation seeds must be nonempty and disjoint')
    for k in ('training_seeds', 'validation_seeds', 'reserved_evaluation_seeds'):
        if len(spec[k]) != len(set(spec[k])):
            raise ValueError('duplicate seeds')


def collect(spec, output):
    """Called only after explicit CLI --execute. All failures remain in the dataset folder."""
    validate_spec(spec)
    from env.simulator import HomeCareEnv
    from env.scenario import generate_trajectory
    from planners.ortools_planner import ORToolsPlanner
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output/'spec.json', spec)
    save_json(output/'version.json', provenance())
    base = load_json(ROOT/spec['base_config'])
    save_json(output/'base_config.json', base)
    records, days = [], []
    for scenario, overrides in spec['scenarios'].items():
        for seed in spec['training_seeds'] + spec['validation_seeds']:
            config = merge_config(base, overrides)
            config['planners']['ortools'].update(spec['teacher_overrides'])
            day_id = f'{scenario}_{seed}'
            folder = output/day_id
            folder.mkdir()
            day = {'day_id': day_id, 'scenario': scenario, 'seed': seed, 'status': 'running',
                   'split': 'train' if seed in spec['training_seeds'] else 'validation'}
            counts, pending, decisions = Counter(), [], []
            try:
                trajectory = generate_trajectory(config, seed)
                save_json(folder/'trajectory.json', trajectory)
                save_json(folder/'config.json', config)
                day['trajectory_sha256'] = sha256_json(trajectory)
                env = HomeCareEnv(config, trajectory)
                state = env.reset(seed)
                teacher = ORToolsPlanner(config['planners']['ortools'])
                done = False
                while not done:
                    if len(decisions) >= config['experiment']['max_decisions']:
                        raise RuntimeError('teacher max_decisions reached')
                    tasks, rows = feature_rows(state, spec['scales'])
                    action = teacher.decide(deepcopy(state))
                    info_teacher = deepcopy(teacher.last_info)
                    ids = [t['id'] for t in tasks]
                    snapshot_time = state['time_s']
                    snapshot_sha = sha256_json(state)
                    new_state, _, done, info = env.step(action)
                    reason = None
                    if info_teacher.get('status') != 'OPTIMAL' or info_teacher.get('fallback'):
                        reason = 'non_optimal_or_fallback_teacher'
                    elif len(ids) < 2:
                        reason = 'fewer_than_two_choices'
                    elif action.get('type') != 'serve' or action.get('task_id') not in ids:
                        reason = 'not_ordinary_task_choice'
                    elif (info.get('executed_action') != action or info.get('invalid_action')
                          or info.get('override_reason')):
                        reason = 'safety_changed_suggestion'
                    elif len(pending) >= spec['max_examples_per_day']:
                        reason = 'predeclared_day_cap'
                    if reason:
                        counts[reason] += 1
                    else:
                        pending.append({'day_id': day_id, 'seed': seed, 'split': day['split'],
                                        'time_s': snapshot_time, 'state_sha256': snapshot_sha,
                                        'candidate_ids': ids, 'features': rows,
                                        'tie_break_keys': [[t['deadline_s'], t['release_time_s'], t['id']]
                                                           for t in tasks],
                                        'target': ids.index(action['task_id']), 'teacher_status': 'OPTIMAL'})
                    decisions.append({'time_s': snapshot_time, 'requested_action': action,
                                      'planner_info': info_teacher, 'info': info, 'skip_reason': reason})
                    state = new_state
                save_json(folder/'tasks.json', env.export_tasks())
                save_jsonl(folder/'event_log.jsonl', env.event_log)
                day['status'] = 'completed'
                records.extend(pending)
            except Exception:
                day['status'] = 'failed'
                save_json(folder/'error.json', {'traceback': traceback.format_exc()})
            finally:
                save_jsonl(folder/'decisions.jsonl', decisions)
                day.update(eligible_examples=len(pending), skipped=dict(counts))
                days.append(day)
                save_json(output/'days.json', days)
                save_jsonl(output/'examples.jsonl', records)
    manifest = {'schema': SCHEMA, 'feature_names': FEATURES, 'scales': spec['scales'],
                'spec_sha256': sha256_json(spec), 'status': 'completed' if all(
                    d['status'] == 'completed' for d in days) else 'failed',
                'days': len(days), 'examples': len(records),
                'examples_sha256': hashlib.sha256((output/'examples.jsonl').read_bytes()).hexdigest(),
                'teacher': 'rolling_ortools_OPTIMAL_only_not_full_day_optimum', 'test_only': False}
    save_json(output/'dataset.json', manifest)
    return manifest


def fit_dataset(spec, dataset, model_path):
    validate_spec(spec)
    dataset, model_path = Path(dataset), Path(model_path)
    if model_path.exists():
        raise FileExistsError('model already exists; choose another path, never overwrite silently')
    manifest = load_json(dataset/'dataset.json')
    raw = (dataset/'examples.jsonl').read_bytes()
    if (manifest['status'] != 'completed' or manifest['schema'] != SCHEMA
            or manifest['spec_sha256'] != sha256_json(spec)
            or manifest['feature_names'] != FEATURES or manifest['scales'] != spec['scales']
            or manifest['examples_sha256'] != hashlib.sha256(raw).hexdigest()):
        raise ValueError('dataset is incomplete, modified or from another spec')
    rows = [json.loads(line) for line in raw.decode().splitlines() if line]
    if any(e['seed'] not in spec['training_seeds']+spec['validation_seeds']
           or e['teacher_status'] != 'OPTIMAL' for e in rows):
        raise ValueError('unexpected seed or teacher provenance')
    train = [e for e in rows if e['seed'] in spec['training_seeds']]
    val = [e for e in rows if e['seed'] in spec['validation_seeds']]
    if not train or not val:
        raise ValueError('both train and validation need eligible examples; inspect coverage first')
    started = time.perf_counter()
    weights, losses = fit_weights(train, **spec['training'])
    from planners.learned import validate_examples
    validate_examples(val)
    scores = {}
    for e in val:
        probs = probabilities(e['features'], weights)
        keys = e.get('tie_break_keys')
        if not manifest['test_only'] and (not keys or len(keys) != len(probs)):
            raise ValueError('real validation examples require the same tie keys as inference')
        chosen = min(range(len(probs)), key=lambda i: (-probs[i], keys[i] if keys else [i]))
        scores.setdefault(e['day_id'], []).append(chosen == e['target'])
    model = {'schema': SCHEMA, 'feature_names': FEATURES, 'scales': spec['scales'], 'weights': weights,
             'training_examples': len(train), 'validation_examples': len(val),
             'training_seeds': spec['training_seeds'], 'validation_seeds': spec['validation_seeds'],
             'dataset_sha256': manifest['examples_sha256'], 'dataset_path': str(dataset.resolve()),
             'test_only': bool(manifest['test_only']), 'training_config': spec['training'],
             'training_wall_s': time.perf_counter()-started,
             'validation_teacher_agreement_by_day': {k: sum(v)/len(v) for k,v in scores.items()},
             'training_losses': losses, 'version': provenance(),
             'interpretation': 'Teacher imitation on held-out days, not measured care or scheduling superiority'}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects a concurrent model writer too.
    with model_path.open('x') as f:
        json.dump(model, f, ensure_ascii=False, indent=2, allow_nan=False)
    return model


def main():
    parser = argparse.ArgumentParser(description='Prepare teacher data or fit a lightweight learned dispatcher')
    parser.add_argument('stage', choices=['collect','fit'])
    parser.add_argument('--config', default='configs/imitation_training.json')
    parser.add_argument('--dataset')
    parser.add_argument('--model', default='models/homecare_imitation.json')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    spec = load_json(args.config)
    validate_spec(spec)
    if not args.execute:
        print('PREVIEW ONLY: no simulation or training started. Pass --execute after explicitly resuming.')
        print(f"Teacher days: {len(spec['scenarios'])*(len(spec['training_seeds'])+len(spec['validation_seeds']))}")
        return 0
    if args.stage == 'collect':
        folder = Path('results/imitation') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        result = collect(spec, folder)
        print(f'Dataset: {folder.resolve()} ({result["status"]}, {result["examples"]} examples)')
        return 0 if result['status'] == 'completed' else 1
    if not args.dataset:
        parser.error('--dataset is required for fitting')
    fit_dataset(spec, args.dataset, args.model)
    print(f'Model: {Path(args.model).resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
