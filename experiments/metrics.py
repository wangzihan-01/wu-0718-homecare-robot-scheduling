"""Original project code: day-level metrics; no copied third-party implementation."""
from __future__ import annotations

import math
import random
from statistics import mean
from typing import Iterable


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def quantile(values: Iterable[float], probability: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def compute_metrics(tasks: list[dict], horizon_s: int, critical_priority: int,
                    decisions: list[dict], stats: dict, event_log: list[dict] | None = None) -> dict:
    """Every released ordinary task is counted, including tasks after robot failure.

    Incomplete overdue work receives terminal-censored tardiness, never an invented
    completion. Responded emergencies still retain a separate incomplete count.
    """
    tasks = [t for t in tasks if t['release_time_s'] < horizon_s]
    ordinary = [t for t in tasks if t['type'] != 'emergency']
    emergencies = [t for t in tasks if t['type'] == 'emergency']

    def completed(t):
        return t.get('completed_at_s') is not None and t['completed_at_s'] <= horizon_s

    def tardiness(t):
        end = t['completed_at_s'] if completed(t) else horizon_s
        return max(0, end - t['deadline_s'])

    critical = [t for t in ordinary if t['priority'] >= critical_priority and t['deadline_s'] <= horizon_s]
    critical_ontime = sum(completed(t) and t['completed_at_s'] <= t['deadline_s'] for t in critical)
    incomplete = sum(not completed(t) for t in ordinary)
    responses = [t['response_at_s'] - t['release_time_s'] for t in emergencies
                 if t.get('response_at_s') is not None and t['response_at_s'] <= horizon_s]
    times = [d['wall_time_s'] for d in decisions if d.get('wall_time_s') is not None]
    invalid = sum(bool(d.get('info', {}).get('invalid_action')) for d in decisions)
    intervened = sum(bool(d.get('info', {}).get('override_reason')) or
                     any(e.get('type') == 'safety_override' for e in d.get('info', {}).get('events', []))
                     for d in decisions)
    fallbacks = sum(bool(d.get('planner_info', {}).get('fallback')) for d in decisions)
    return {
        'task_count': len(ordinary),
        'completed_tasks': len(ordinary) - incomplete,
        'unfinished_tasks': incomplete,
        'unfinished_task_rate': ratio(incomplete, len(ordinary)),
        'critical_task_count': len(critical),
        'critical_ontime_count': critical_ontime,
        'critical_ontime_rate': ratio(critical_ontime, len(critical)),
        'total_tardiness_s': sum(tardiness(t) for t in ordinary),
        'weighted_tardiness_priority_s': sum(tardiness(t) * t['priority'] for t in ordinary),
        'unfinished_overdue_count': sum(not completed(t) and t['deadline_s'] < horizon_s for t in ordinary),
        'emergency_count': len(emergencies),
        'emergency_responded_count': len(responses),
        'emergency_unresponded_count': len(emergencies) - len(responses),
        'emergency_unresponded_rate': ratio(len(emergencies) - len(responses), len(emergencies)),
        'emergency_unfinished_count': sum(not completed(t) for t in emergencies),
        'emergency_response_mean_s': mean(responses) if responses else None,
        'emergency_response_median_s': quantile(responses, 0.5),
        'emergency_response_p95_s': quantile(responses, 0.95),
        'charge_count': stats.get('charge_count', 0),
        'charge_time_s': stats.get('charge_time_s', 0),
        'min_energy_wh': stats.get('min_energy_wh'),
        'battery_violations': stats.get('battery_violations', 0),
        'robot_failed': bool(stats.get('robot_failed', False)),
        'energy_consumed_wh': stats.get('energy_consumed_wh'),
        'travel_distance_m': stats.get('travel_distance_m'),
        'moving_time_s': stats.get('mode_time_s', {}).get('moving', 0),
        'service_time_s': stats.get('mode_time_s', {}).get('serving', 0),
        'waiting_time_s': sum(stats.get('mode_time_s', {}).get(k, 0) for k in ('waiting', 'idle')),
        'failed_time_s': stats.get('mode_time_s', {}).get('failed', 0),
        'service_time_fraction': ratio(stats.get('mode_time_s', {}).get('serving', 0), horizon_s),
        'decision_count': len(decisions),
        'decision_wall_mean_s': mean(times) if times else None,
        'decision_wall_p95_s': quantile(times, 0.95),
        'decision_wall_total_s': sum(times),
        'invalid_action_count': invalid,
        'invalid_action_rate': ratio(invalid, len(decisions)),
        'safety_intervention_decision_count': intervened,
        'safety_intervention_rate': ratio(intervened, len(decisions)),
        'safety_override_event_count': sum(e.get('type') == 'safety_override' for e in event_log or []),
        'planner_fallback_count': fallbacks,
        'planner_fallback_rate': ratio(fallbacks, len(decisions)),
    }


def paired_bootstrap(rows: list[dict], metric: str, algorithm_a: str, algorithm_b: str,
                     *, confidence: float = 0.95, n_resamples: int = 2000, seed: int = 5301) -> dict:
    """Paired differences; resample entire simulation days within scenario strata.

    No task-level pseudo-replication. Duplicate algorithm/scenario/seed rows are
    rejected: choosing a repeated run must be explicit before statistical analysis.
    Missing/invalid pairs are reported; formal analyses should inspect these first.
    """
    if not 0 < confidence < 1 or n_resamples < 1:
        raise ValueError('confidence must be in (0,1); n_resamples must be positive')
    indexed: dict[tuple, dict] = {}
    for row in rows:
        if row['algorithm'] not in {algorithm_a, algorithm_b}:
            continue
        key = (row['scenario'], int(row['seed']), row['algorithm'])
        if key in indexed:
            raise ValueError(f'duplicate day: {key}')
        indexed[key] = row
    days = sorted({(k[0], k[1]) for k in indexed})
    strata: dict[str, list[float]] = {}
    excluded = []
    for scenario, day_seed in days:
        a = indexed.get((scenario, day_seed, algorithm_a))
        b = indexed.get((scenario, day_seed, algorithm_b))
        if (a is None or b is None or a.get('run_status') != 'completed' or b.get('run_status') != 'completed'
                or a.get(metric) in (None, '') or b.get(metric) in (None, '')):
            excluded.append({'scenario': scenario, 'seed': day_seed})
            continue
        strata.setdefault(scenario, []).append(float(a[metric]) - float(b[metric]))
    differences = [value for values in strata.values() for value in values]
    result = {'metric': metric, 'contrast': f'{algorithm_a} - {algorithm_b}', 'confidence': confidence,
              'n_resamples': n_resamples, 'bootstrap_seed': seed, 'paired_days': len(differences),
              'excluded_days': excluded, 'resampling_unit': 'simulation_day_within_scenario',
              'estimate': None, 'ci_low': None, 'ci_high': None}
    if differences:
        rng = random.Random(seed)
        samples = [mean([rng.choice(values) for values in strata.values() for _ in values])
                   for _ in range(n_resamples)]
        result.update(estimate=mean(differences), ci_low=quantile(samples, (1 - confidence) / 2),
                      ci_high=quantile(samples, (1 + confidence) / 2))
    return result
