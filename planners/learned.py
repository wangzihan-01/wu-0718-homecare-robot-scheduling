"""Original CPU-only imitation dispatcher; no upstream code or checkpoints copied."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .base import BasePlanner, candidates, idle_action, locked_action

SCHEMA = 'homecare_imitation_linear_v1'
FEATURES = ['priority', 'slack', 'travel', 'service', 'age', 'earliest_wait',
            'energy_fraction', 'priority_urgency']
SCALE_KEYS = {'priority', 'slack_s', 'travel_s', 'service_s', 'age_s', 'wait_s'}


class LearnedNotConfigured(RuntimeError):
    pass


def validate_scales(scales):
    if set(scales) != SCALE_KEYS or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                                       for v in scales.values()):
        raise ValueError('feature scales must be the documented positive finite values')


def feature_rows(state, scales):
    """Read only named public fields; no environment, RNG or future trajectory access."""
    validate_scales(scales)
    tasks = sorted(candidates(state), key=lambda t: t['id'])
    now, m = state['time_s'], state['public_model']
    energy = max(float(state['battery']['energy_wh']), 1e-9)
    rows = []
    for t in tasks:
        loc = t['location']
        travel = m['travel_time_from_position_s'][loc]
        service = t['remaining_service_s']
        wait = max(0, t['earliest_start_s'] - now - travel)
        slack = t['deadline_s'] - now - travel - wait - service
        priority = t['priority'] / scales['priority']
        task_energy = ((m['distance_from_position_m'][loc] + m['distances_m'][loc][m['dock']])
                       * m['move_wh_m'] + service * m['service_wh_s']
                       + wait * m['idle_wh_s'] + m['reserve_wh'])
        row = [math.tanh(priority), math.tanh(slack / scales['slack_s']),
               math.tanh(travel / scales['travel_s']), math.tanh(service / scales['service_s']),
               math.tanh((now - t['release_time_s']) / scales['age_s']),
               math.tanh(wait / scales['wait_s']), math.tanh(task_energy / energy),
               math.tanh(priority * -slack / scales['slack_s'])]
        if not all(math.isfinite(v) for v in row):
            raise ValueError('non-finite public features')
        rows.append(row)
    return tasks, rows


def probabilities(rows, weights):
    logits = [sum(a*b for a, b in zip(row, weights)) for row in rows]
    if not logits or not all(math.isfinite(v) for v in logits):
        raise ValueError('empty or non-finite scores')
    top = max(logits)
    exps = [math.exp(v-top) for v in logits]
    total = sum(exps)
    return [v / total for v in exps]


def validate_examples(examples):
    if not examples:
        raise ValueError('no eligible examples; do not manufacture a trained model')
    for e in examples:
        rows, target = e['features'], e['target']
        if (len(rows) < 2 or type(target) is not int or not 0 <= target < len(rows)
                or any(len(row) != len(FEATURES) or any(type(v) not in (int, float)
                   or not math.isfinite(v) or abs(v) > 1 for v in row) for row in rows)):
            raise ValueError('invalid grouped training example')


def fit_weights(examples, *, epochs, learning_rate, l2, seed):
    """Grouped softmax behavior cloning, learned weights rather than hand-coded scores."""
    import random
    validate_examples(examples)
    if (type(epochs) is not int or epochs < 1 or not math.isfinite(learning_rate)
            or learning_rate <= 0 or not math.isfinite(l2) or l2 < 0):
        raise ValueError('invalid training hyperparameters')
    weights = [0.0] * len(FEATURES)
    rng = random.Random(seed)
    order = list(range(len(examples)))
    losses = []
    for _ in range(epochs):
        rng.shuffle(order)
        for index in order:
            e = examples[index]
            probs = probabilities(e['features'], weights)
            for j in range(len(weights)):
                gradient = sum((p - (i == e['target'])) * row[j]
                               for i, (p, row) in enumerate(zip(probs, e['features'])))
                weights[j] -= learning_rate * (gradient + l2 * weights[j])
        if not all(math.isfinite(v) for v in weights):
            raise ValueError('training diverged')
        losses.append(sum(-math.log(max(probabilities(e['features'], weights)[e['target']], 1e-15))
                          for e in examples) / len(examples))
    return weights, losses


class LearnedPlanner(BasePlanner):
    def __init__(self, config=None):
        super().__init__(config)
        path = Path(self.config.get('model_path') or '')
        if not self.config.get('model_path') or not path.is_file():
            raise LearnedNotConfigured('Learned policy NOT RUN: train and configure a model artifact first')
        raw = path.read_bytes()
        model = json.loads(raw)
        if (model.get('schema') != SCHEMA or model.get('feature_names') != FEATURES
                or type(model.get('training_examples')) is not int or model['training_examples'] < 1):
            raise ValueError('not a compatible trained imitation model')
        if model.get('test_only', False) and not self.config.get('allow_test_model', False):
            raise LearnedNotConfigured('Test-only model cannot be used as a research result')
        self.test_only = model.get('test_only', False)
        self.weights, self.scales = model['weights'], model['scales']
        validate_scales(self.scales)
        if len(self.weights) != len(FEATURES) or any(type(v) not in (int, float) or not math.isfinite(v)
                                                  for v in self.weights):
            raise ValueError('invalid model weights')
        self.artifact_sha256 = hashlib.sha256(raw).hexdigest()

    def decide(self, state):
        self.last_info = {'planner': 'learned', 'model_sha256': self.artifact_sha256,
                          'fallback': False, 'test_only': self.test_only}
        locked = locked_action(state)
        if locked:
            self.last_info['shared_policy'] = 'locked_activity'
            return locked
        tasks, rows = feature_rows(state, self.scales)
        if not tasks:
            self.last_info['shared_policy'] = 'idle_or_charge'
            return idle_action(state)
        scores = [sum(a*b for a, b in zip(row, self.weights)) for row in rows]
        if not all(math.isfinite(v) for v in scores):
            raise ValueError('non-finite model inference')
        chosen = min(range(len(tasks)), key=lambda i: (-scores[i], tasks[i]['deadline_s'],
                                                      tasks[i]['release_time_s'], tasks[i]['id']))
        self.last_info.update(candidate_ids=[t['id'] for t in tasks], scores=scores)
        return {'type': 'serve', 'task_id': tasks[chosen]['id']}
