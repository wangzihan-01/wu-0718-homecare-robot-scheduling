"""Credential-free LLM protocol, mock transport and replay; no live API client.

A future paid provider adapter must be explicitly enabled outside this module.
"""
import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path
from .base import BasePlanner
from .rules import EDFPlanner


class LLMNotConfigured(RuntimeError):
    pass


def parse_action(raw):
    if not isinstance(raw, str):
        raise ValueError('model output must be a JSON string')
    action = json.loads(raw)
    if not isinstance(action, dict):
        raise ValueError('action must be an object')
    kind = action.get('type')
    keys = {'serve': {'type', 'task_id'}, 'charge': {'type'},
            'wait': {'type', 'duration_s'}, 'continue': {'type'}}
    if kind not in keys or set(action) != keys[kind]:
        raise ValueError('unknown action or extra/missing fields')
    if kind == 'serve' and not isinstance(action['task_id'], str):
        raise ValueError('task_id must be a string')
    if kind == 'wait' and (type(action['duration_s']) is not int or action['duration_s'] <= 0):
        raise ValueError('duration_s must be a positive integer')
    return action


class LLMPlanner(BasePlanner):
    def __init__(self, config=None, transport=None):
        super().__init__(config)
        self.transport = transport
        self.logs = []
        self.mode = self.config.get('mode', 'disabled')
        if self.mode == 'disabled' or not self.config.get('model'):
            raise LLMNotConfigured('LLM NOT RUN: configure a model and explicit mock/replay mode; no paid API adapter is installed')
        if self.mode not in ('mock', 'replay'):
            raise LLMNotConfigured('Live API calls are not implemented or authorized; only mock/replay are supported')
        if self.mode == 'mock' and transport is None:
            raise LLMNotConfigured('mock mode requires an injected test transport')
        version = self.config['prompt_version']
        if version != 'homecare-v1':
            raise ValueError('unknown prompt version')
        self.prompt = (Path(__file__).parent / 'prompts' / f'{version}.txt').read_text()
        self.cache_dir = Path(self.config['cache_dir'])
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def decide(self, state):
        from env.safety import validate_action
        request = {'prompt_version': self.config['prompt_version'], 'prompt': self.prompt,
                   'state': deepcopy(state), 'model': self.config['model'],
                   'temperature': self.config['temperature']}
        encoded = json.dumps(request, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
        key = hashlib.sha256(encoded.encode()).hexdigest()
        cache_file = self.cache_dir / f'{key}.json'
        record = {'planner': 'llm', 'test_only': True, 'experiment_eligible': False,
                  'mode': self.mode, 'cache_key': key, 'input': request, 'attempts': [],
                  'timeout_s': self.config['timeout_s'], 'max_retries': self.config['max_retries'],
                  'fallback': False, 'usage': None, 'cost': None, 'currency': None}
        started = time.perf_counter()
        action = None
        for attempt in range(self.config['max_retries'] + 1):
            attempt_started = time.perf_counter()
            detail = {'attempt': attempt, 'output': None, 'usage': None, 'error': None, 'cache_hit': False}
            try:
                if cache_file.exists():
                    response = json.loads(cache_file.read_text())
                    detail['cache_hit'] = True
                    if response.get('test_only') is not True:
                        raise ValueError('This adapter accepts explicitly test-only cache records')
                elif self.mode == 'replay':
                    raise FileNotFoundError('replay cache miss: no live call permitted')
                else:
                    # Transport contract: must enforce timeout itself. The elapsed-time
                    # check also rejects a slow result; it cannot cancel arbitrary Python.
                    response = self.transport(deepcopy(request), timeout_s=self.config['timeout_s'])
                    if not isinstance(response, dict):
                        raise ValueError('transport must return a dict with output and optional usage')
                    response = {**response, 'test_only': True}
                detail['output'] = response.get('output')
                detail['usage'] = response.get('usage')
                if not detail['cache_hit'] and time.perf_counter() - attempt_started > self.config['timeout_s']:
                    raise TimeoutError('transport returned after configured deadline')
                proposed = parse_action(detail['output'])
                reason = validate_action(state, proposed)
                if reason:
                    raise ValueError('unsafe output: ' + reason)
                if proposed not in state['valid_actions']:
                    raise ValueError('output not in prompted valid_actions')
                action = proposed
                record['usage'] = detail['usage']
                if not detail['cache_hit']:
                    # Replace a private temporary file atomically; no credentials stored.
                    temporary = cache_file.with_suffix('.tmp')
                    temporary.write_text(json.dumps(response, ensure_ascii=False, indent=2))
                    temporary.replace(cache_file)
            except Exception as exc:
                detail['error'] = f'{type(exc).__name__}: {exc}'
            detail['elapsed_s'] = time.perf_counter() - attempt_started
            record['attempts'].append(detail)
            if action is not None:
                break
            if self.mode == 'replay':
                break
        if action is None:
            action = EDFPlanner().decide(state)
            record.update(fallback=True, fallback_policy='feasible_edf', error=record['attempts'][-1]['error'])
        record.update(action=deepcopy(action), elapsed_s=time.perf_counter() - started)
        self.logs.append(record)
        self.last_info = record
        return action
