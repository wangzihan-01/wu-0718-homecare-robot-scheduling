"""Original project implementation; independently generated Poisson arrival streams."""
import copy
import math
import random


def generate_trajectory(config, seed):
    """Generate a full hidden day once, using local RNGs independent of planners."""
    c = config['scenario']
    horizon = config['horizon_s']
    nodes = sorted({n for a, b, _ in config['model']['edges'] for n in (a, b)} - {config['model']['dock']})
    tasks = []
    for emergency, rate, stream in [(False, c['task_rate_per_hour'], 11), (True, c['emergency_rate_per_hour'], 29)]:
        rng = random.Random(f'{seed}:{stream}')
        if rate < 0:
            raise ValueError('arrival rate must be nonnegative')
        if not rate:
            continue
        t, number = 0.0, 0
        while True:
            t += rng.expovariate(rate / 3600)
            release = math.ceil(t)
            if release >= horizon:
                break
            number += 1
            k = rng.randrange(len(c['types']))
            duration = c['emergency_service_s'] if emergency else rng.randint(*c['service_range_s'])
            earliest = release if emergency else release + rng.randint(*c['earliest_delay_range_s'])
            deadline = release + c['emergency_deadline_s'] if emergency else earliest + duration + rng.randint(*c['deadline_slack_range_s'])
            tasks.append(dict(id=f'{"E" if emergency else "T"}{number:04d}', type='emergency' if emergency else c['types'][k],
                              release_time_s=release, earliest_start_s=earliest, deadline_s=deadline,
                              priority=rng.randint(4, 5) if emergency else c['priorities'][k],
                              location=rng.choice(nodes), service_duration_s=duration, preemptible=True))
    return {'seed': seed, 'tasks': sorted(tasks, key=lambda x: (x['release_time_s'], x['id']))}


def validate_trajectory(trajectory, config):
    tasks = copy.deepcopy(trajectory['tasks'])
    nodes = {n for a, b, _ in config['model']['edges'] for n in (a, b)}
    ids = set()
    for task in tasks:
        if not isinstance(task['id'], str) or task['id'] in ids:
            raise ValueError('task ids must be unique strings')
        ids.add(task['id'])
        for field in ('release_time_s', 'earliest_start_s', 'deadline_s', 'service_duration_s', 'priority'):
            if type(task[field]) is not int:
                raise ValueError(f'{field} must be integer')
        if not 0 <= task['release_time_s'] < config['horizon_s'] or task['earliest_start_s'] < task['release_time_s']:
            raise ValueError('invalid release or earliest start')
        if task['service_duration_s'] <= 0 or task['priority'] <= 0 or task['location'] not in nodes:
            raise ValueError('invalid service, priority, or location')
        if type(task['preemptible']) is not bool:
            raise ValueError('preemptible must be bool')
        if task['type'] not in config['scenario']['types'] + ['emergency']:
            raise ValueError('unknown task type')
    return sorted(tasks, key=lambda x: (x['release_time_s'], x['id']))
