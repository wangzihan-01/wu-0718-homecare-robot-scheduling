"""Shared public-state feasibility checks; original code, no private simulator access."""
import math


def task_energy(state, task):
    m = state['public_model']
    travel = m['travel_time_from_position_s'][task['location']]
    wait = max(0, task['earliest_start_s'] - state['time_s'] - travel)
    return (m['distance_from_position_m'][task['location']] * m['move_wh_m']
            + task['remaining_service_s'] * m['service_wh_s'] + wait * m['idle_wh_s']
            + m['distances_m'][task['location']][m['dock']] * m['move_wh_m'] + m['reserve_wh'])


def charge_feasible(state):
    m = state['public_model']
    return state['battery']['energy_wh'] + 1e-9 >= m['distance_from_position_m'][m['dock']] * m['move_wh_m']


def validate_action(state, action):
    if not isinstance(action, dict) or not isinstance(action.get('type'), str):
        return 'invalid_action_schema'
    kind = action['type']
    fields = {'serve': {'type', 'task_id'}, 'charge': {'type'}, 'continue': {'type'}, 'wait': {'type', 'duration_s'}}
    if kind not in fields or set(action) != fields[kind]:
        return 'invalid_action_schema'
    if kind == 'wait' and (type(action['duration_s']) is not int or not 1 <= action['duration_s'] <= state['public_model']['max_wait_s']):
        return 'invalid_wait_duration'
    if kind == 'serve' and not isinstance(action['task_id'], str):
        return 'invalid_task_id'
    if state['mode'] in ('failed', 'finished'):
        return None if kind in ('continue', 'wait') else 'robot_unavailable'
    current = next((t for t in state['pending_tasks'] if t['id'] == state['active_task_id']), None)
    if current and state['mode'] == 'serving' and not current['preemptible']:
        if kind != 'continue' and not (kind == 'serve' and action['task_id'] == current['id']):
            return 'nonpreemptible_service'
    if kind == 'serve':
        task = next((t for t in state['pending_tasks'] if t['id'] == action['task_id']), None)
        if task is None or task['status'] == 'unresponded':
            return 'unknown_or_unavailable_task'
        if task_energy(state, task) > state['battery']['energy_wh'] + 1e-9:
            return 'insufficient_task_return_energy'
    if kind == 'charge' and not charge_feasible(state):
        return 'unreachable_dock'
    return None


def fallback_action(state):
    for task in sorted(state['pending_tasks'], key=lambda t: (t['deadline_s'], t['release_time_s'], t['id'])):
        action = {'type': 'serve', 'task_id': task['id']}
        if validate_action(state, action) is None:
            return action
    if state['active_task_id'] and state['mode'] == 'serving':
        return {'type': 'continue'}
    if state['battery']['energy_wh'] < state['battery']['capacity_wh'] * state['public_model']['charge_target_fraction'] - 1e-9 and validate_action(state, {'type': 'charge'}) is None:
        return {'type': 'charge'}
    return {'type': 'wait', 'duration_s': state['public_model']['default_wait_s']}


def valid_actions(state):
    actions = [{'type': 'serve', 'task_id': t['id']} for t in state['pending_tasks']]
    actions += [{'type': 'charge'}, {'type': 'wait', 'duration_s': state['public_model']['default_wait_s']}, {'type': 'continue'}]
    return [a for a in actions if validate_action(state, a) is None]
