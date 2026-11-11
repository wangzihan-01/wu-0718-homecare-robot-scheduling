"""Original project code: reproducible single-day runner and artifact retention."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import time
import traceback
from uuid import uuid4

from experiments.metrics import compute_metrics

ROOT = Path(__file__).resolve().parents[1]


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def save_jsonl(path, rows):
    with Path(path).open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def save_csv(path, rows, fields=None):
    fields = fields or sorted({key for row in rows for key in row})
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def merge_config(base, overrides):
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def sha256_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def provenance():
    def git(*args):
        try:
            return subprocess.check_output(['git', *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    files = {}
    for directory in ('env', 'planners', 'experiments', 'configs'):
        for path in sorted((ROOT / directory).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                files[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for filename in ('requirements.txt', 'requirements-lock.txt', 'docs/INTERFACE.md'):
        path = ROOT / filename
        if path.exists():
            files[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'git_commit': git('rev-parse', 'HEAD'), 'git_status': git('status', '--porcelain'),
            'source_sha256': files, 'python': platform.python_version(), 'platform': platform.platform(),
            'dependencies': {d.metadata['Name']: d.version for d in metadata.distributions()},
            'decision_time_injected_into_simulation': False}


def run_one(config: dict, seed: int, algorithm: str, *, scenario: str = 'custom',
            output_root: str | Path = 'results', trajectory: dict | None = None,
            version: dict | None = None) -> dict:
    """Retain a directory and summary row even on planner/environment failure."""
    run_id = f'{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{scenario}_{algorithm}_{seed}_{uuid4().hex[:8]}'
    folder = Path(output_root) / run_id
    folder.mkdir(parents=True, exist_ok=False)
    run_version = deepcopy(version) if version is not None else provenance()
    manifest = {'run_id': run_id, 'seed': seed, 'algorithm': algorithm, 'scenario': scenario,
                'purpose': config['experiment']['purpose'], 'run_status': 'running',
                'git_commit': run_version.get('git_commit'), 'config_sha256': sha256_json(config),
                'output_dir': str(folder.resolve())}
    save_json(folder / 'config.json', config)
    save_json(folder / 'version.json', run_version)
    save_json(folder / 'manifest.json', manifest)
    decisions, tasks, events, stats = [], [], [], {}
    environment = planner = None
    metrics = {}
    error = None
    try:
        from env.scenario import generate_trajectory
        from env.simulator import HomeCareEnv
        from planners import make_planner
        if trajectory is None:
            trajectory = generate_trajectory(config, seed)
        trajectory = deepcopy(trajectory)
        save_json(folder / 'trajectory.json', trajectory)
        manifest['trajectory_sha256'] = sha256_json(trajectory)
        environment = HomeCareEnv(config, trajectory)
        state = environment.reset(seed)
        planner = make_planner(algorithm, config['planners'].get(algorithm, {}))
        if algorithm == 'learned':
            manifest['model_sha256'] = planner.artifact_sha256
            if planner.test_only:
                manifest.update(test_only=True, purpose='learned_adapter_test_only')
        done = False
        while not done:
            if len(decisions) >= config['experiment']['max_decisions']:
                raise RuntimeError('max_decisions reached: retained as failed, not a completed day')
            started = time.perf_counter()
            record = {'decision_index': len(decisions), 'time_s': state['time_s'],
                      'state_sha256': sha256_json(state)}
            try:
                action = planner.decide(deepcopy(state))
            except Exception:
                record.update(wall_time_s=time.perf_counter() - started,
                              planner_info=deepcopy(getattr(planner, 'last_info', {})),
                              error=traceback.format_exc())
                decisions.append(record)
                raise
            record.update(requested_action=deepcopy(action), wall_time_s=time.perf_counter() - started,
                          planner_info=deepcopy(getattr(planner, 'last_info', {})))
            decisions.append(record)
            state, reward, done, info = environment.step(action)
            record.update(info=deepcopy(info), next_time_s=state['time_s'], reward=reward, done=done)
        tasks = environment.export_tasks()
        events = environment.event_log
        stats = environment.stats
        metrics = compute_metrics(tasks, config['horizon_s'], config['model']['critical_priority'],
                                  decisions, stats, events)
        manifest.update(run_status='completed', final_time_s=state['time_s'], evaluation_valid=True)
        if algorithm == 'learned' and planner.test_only:
            manifest['evaluation_valid'] = False
        if algorithm == 'llm' and config['planners']['llm'].get('mode') in {'mock', 'replay'}:
            # Replay of mock records is also a test, not fresh external-model evidence.
            manifest.update(test_only=True, purpose='llm_adapter_test_only', evaluation_valid=False)
    except Exception as exc:
        error = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        manifest.update(run_status='not_run' if type(exc).__name__ in {'LLMNotConfigured', 'LearnedNotConfigured'} else 'failed',
                        evaluation_valid=False, error_type=type(exc).__name__, error_message=str(exc))
        if environment is not None:
            # reset may fail before runtime fields exist; preserve the original
            # error and whatever snapshots are actually available.
            if hasattr(environment, 'tasks'):
                tasks = environment.export_tasks()
            events = getattr(environment, 'event_log', [])
            stats = getattr(environment, 'stats', {})
        # A software failure is not a completed simulation day: retain artifacts but
        # do not fabricate end-of-day task rates from a truncated execution.
        save_json(folder / 'error.json', error)
    finally:
        save_csv(folder / 'tasks.csv', tasks, fields=[
            'id','type','release_time_s','earliest_start_s','deadline_s','priority','location',
            'service_duration_s','preemptible','remaining_service_s','status','completed_at_s',
            'response_at_s','failure_reason'])
        save_json(folder / 'tasks.json', tasks)
        save_jsonl(folder / 'event_log.jsonl', events)
        save_jsonl(folder / 'decision_log.jsonl', decisions)
        save_jsonl(folder / 'action_log.jsonl', getattr(environment, 'action_log', []))
        save_jsonl(folder / 'llm_calls.jsonl', getattr(planner, 'logs', []))
        save_json(folder / 'stats.json', stats)
        save_json(folder / 'manifest.json', manifest)
        summary = {**manifest, **metrics}
        save_json(folder / 'metrics.json', summary)
        save_csv(folder / 'summary.csv', [summary])
    return summary


def main():
    parser = argparse.ArgumentParser(description='One day; no paid model transport is provided.')
    parser.add_argument('--config', default='configs/base.json')
    parser.add_argument('--algorithm', choices=['fifo','edf','greedy','ortools','llm','learned'], default='edf')
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--scenario', default='custom')
    parser.add_argument('--output', default='results/single')
    parser.add_argument('--trajectory', help='Replay a previously exported complete trajectory JSON')
    args = parser.parse_args()
    summary = run_one(load_json(args.config), args.seed, args.algorithm, scenario=args.scenario,
                      output_root=args.output, trajectory=load_json(args.trajectory) if args.trajectory else None)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary['run_status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
