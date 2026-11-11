"""Deterministic rules over public, energy-feasible task snapshots."""
from .base import BasePlanner, candidates, idle_action, locked_action


class RulePlanner(BasePlanner):
    name = 'rule'

    def key(self, task, state):
        raise NotImplementedError

    def decide(self, state):
        self.last_info = {'planner': self.name, 'fallback': False}
        locked = locked_action(state)
        if locked:
            return locked
        available = candidates(state)
        if not available:
            return idle_action(state)
        task = min(available, key=lambda t: self.key(t, state))
        return {'type': 'serve', 'task_id': task['id']}


class FIFOPlanner(RulePlanner):
    name = 'fifo'

    def key(self, task, state):
        return task['release_time_s'], task['id']


class EDFPlanner(RulePlanner):
    name = 'edf'

    def key(self, task, state):
        return task['deadline_s'], task['release_time_s'], task['id']


class GreedyPlanner(RulePlanner):
    name = 'greedy'

    def key(self, task, state):
        c = self.config
        distance = state['public_model']['distance_from_position_m'][task['location']]
        score = c['priority_weight'] * task['priority'] / c['priority_scale'] - c['distance_weight'] * distance / c['distance_scale_m']
        return -score, task['release_time_s'], task['id']
