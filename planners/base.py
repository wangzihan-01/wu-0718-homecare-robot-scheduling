"""Original project code; no third-party implementation copied."""
from abc import ABC, abstractmethod
from copy import deepcopy


class BasePlanner(ABC):
    def __init__(self, config=None):
        self.config = deepcopy(config or {})
        self.last_info = {}

    @abstractmethod
    def decide(self, state):
        """Return one JSON action from an independent public state snapshot."""
        raise NotImplementedError


def candidates(state):
    allowed = {a['task_id'] for a in state['valid_actions'] if a['type'] == 'serve'}
    return [t for t in state['pending_tasks'] if t['id'] in allowed and t['type'] != 'emergency']


def idle_action(state):
    # Finish a charge instead of generating repeated tiny charge cycles.
    if state['mode'] in ('charging', 'moving') and state['active_task_id'] is None:
        return {'type': 'continue'}
    charge = {'type': 'charge'}
    m = state['public_model']
    if charge in state['valid_actions'] and state['battery']['energy_wh'] < state['battery']['capacity_wh'] * m['charge_target_fraction'] - 1e-8:
        return charge
    return {'type': 'wait', 'duration_s': m['default_wait_s']}


def locked_action(state):
    active = next((t for t in state['pending_tasks'] if t['id'] == state['active_task_id']), None)
    if active and (active['type'] == 'emergency' or (state['mode'] == 'serving' and not active['preemptible'])):
        return {'type': 'continue'}
    return None
