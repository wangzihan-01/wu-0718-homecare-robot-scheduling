"""Original single-sortie rolling CP-SAT model; OR-Tools is Apache-2.0.

Uses the published OR-Tools API, not copied example implementation.
"""
import math
import time
from ortools.sat.python import cp_model
from .base import BasePlanner, candidates, idle_action, locked_action
from .rules import EDFPlanner


class ORToolsPlanner(BasePlanner):
    def decide(self, state):
        started = time.perf_counter()
        self.last_info = {'planner': 'ortools', 'status': 'NOT_SOLVED', 'objective': None,
                          'best_bound': None, 'fallback': False, 'time_limit_s': self.config['time_limit_s'],
                          'model': 'single_sortie_v1'}
        locked = locked_action(state)
        if locked:
            self.last_info.update(status='LOCKED', wall_time_s=time.perf_counter() - started)
            return locked
        c, m, now = self.config, state['public_model'], state['time_s']
        end = min(state['horizon_s'], now + c['horizon_s'])
        tasks = sorted(candidates(state), key=lambda t: (t['deadline_s'], t['release_time_s'], t['id']))[:c['max_tasks']]
        self.last_info['candidate_ids'] = [t['id'] for t in tasks]
        if not tasks or end <= now:
            self.last_info.update(status='NO_CANDIDATES', wall_time_s=time.perf_counter() - started)
            return idle_action(state)
        n = len(tasks)
        model = cp_model.CpModel()
        selected = {i: model.new_bool_var(f'use_{i}') for i in range(1, n + 1)}
        start = {i: model.new_int_var(0, end, f'start_{i}') for i in selected}
        completion = {i: model.new_int_var(0, end, f'completion_{i}') for i in selected}
        wait = {i: model.new_int_var(0, end - now, f'wait_{i}') for i in selected}
        late = {i: model.new_int_var(0, max(end, end - tasks[i-1]['deadline_s']), f'late_{i}') for i in selected}
        arcs, x, travel, distance = [], {}, {}, {}
        empty = model.new_bool_var('empty')
        arcs.append((0, 0, empty))
        model.add(sum(selected.values()) == 0).only_enforce_if(empty)
        model.add(sum(selected.values()) >= 1).only_enforce_if(empty.Not())
        for i in selected:
            arcs.append((i, i, selected[i].Not()))
            task = tasks[i-1]
            model.add(start[i] >= max(now, task['earliest_start_s'])).only_enforce_if(selected[i])
            model.add(completion[i] == start[i] + task['remaining_service_s']).only_enforce_if(selected[i])
            model.add(late[i] >= completion[i] - task['deadline_s']).only_enforce_if(selected[i])
            for variable in (start[i], completion[i], wait[i], late[i]):
                model.add(variable == 0).only_enforce_if(selected[i].Not())
        for i in range(n + 1):
            for j in range(n + 1):
                if i == j:
                    continue
                a = x[i, j] = model.new_bool_var(f'arc_{i}_{j}')
                arcs.append((i, j, a))
                if i == 0:
                    loc = tasks[j-1]['location']
                    travel[i, j] = m['travel_time_from_position_s'][loc]
                    distance[i, j] = m['distance_from_position_m'][loc]
                    model.add(start[j] >= now + travel[i, j]).only_enforce_if(a)
                    model.add(wait[j] == start[j] - now - travel[i, j]).only_enforce_if(a)
                elif j == 0:
                    loc = tasks[i-1]['location']
                    travel[i, j] = m['travel_times_s'][loc][m['dock']]
                    distance[i, j] = m['distances_m'][loc][m['dock']]
                    model.add(completion[i] + travel[i, j] <= end).only_enforce_if(a)
                else:
                    origin, dest = tasks[i-1]['location'], tasks[j-1]['location']
                    travel[i, j] = m['travel_times_s'][origin][dest]
                    distance[i, j] = m['distances_m'][origin][dest]
                    model.add(start[j] >= completion[i] + travel[i, j]).only_enforce_if(a)
                    model.add(wait[j] == start[j] - completion[i] - travel[i, j]).only_enforce_if(a)
        model.add_circuit(arcs)
        scale = c['energy_scale']
        movement = sum(math.ceil(distance[k] * m['move_wh_m'] * scale) * a for k, a in x.items())
        service = sum(math.ceil(tasks[i-1]['remaining_service_s'] * m['service_wh_s'] * scale) * selected[i] for i in selected)
        waiting = sum(wait.values()) * math.ceil(m['idle_wh_s'] * scale)
        budget = math.floor((state['battery']['energy_wh'] - m['reserve_wh']) * scale)
        model.add(movement + service + waiting <= budget)
        # Integer objective; fixed task order and fixed solver seed resolve search reproducibly
        # absent machine-dependent wall-clock truncation. Log incumbent and bound.
        reward = sum(tasks[i-1]['priority'] * c['completion_reward'] * selected[i] for i in selected)
        tardiness = sum(tasks[i-1]['priority'] * c['tardiness_weight'] * late[i] for i in selected)
        route_time = sum(travel[k] * a for k, a in x.items())
        # Tiny secondary terms explicitly prefer earlier fixed-index first decisions.
        primary = reward - tardiness - route_time
        model.maximize(primary * (n + 1) - sum(j * x[0, j] for j in selected))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = c['time_limit_s']
        solver.parameters.num_search_workers = c['num_workers']
        solver.parameters.random_seed = c['seed']
        status = solver.solve(model)
        self.last_info.update(status=solver.status_name(status), solver_wall_time_s=solver.wall_time,
                              wall_time_s=time.perf_counter() - started,
                              num_workers=c['num_workers'], solver_seed=c['seed'])
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            self.last_info.update(objective=solver.objective_value, best_bound=solver.best_objective_bound)
            route = []
            previous = 0
            while True:
                successor = next((j for j in range(n + 1) if j != previous and (previous, j) in x and solver.value(x[previous, j])), 0)
                if successor == 0:
                    break
                route.append(successor)
                previous = successor
            self.last_info['planned_route'] = [tasks[i-1]['id'] for i in route]
            self.last_info['planned_schedule'] = [{'task_id': tasks[i-1]['id'], 'start_s': solver.value(start[i]), 'completion_s': solver.value(completion[i])} for i in route]
            self.last_info['planned_energy_wh'] = solver.value(movement + service + waiting) / scale
            if route and status == cp_model.OPTIMAL:
                return {'type': 'serve', 'task_id': tasks[route[0]-1]['id']}
            reason = 'time_limit_with_incumbent' if status == cp_model.FEASIBLE else 'empty_sortie'
        else:
            self.last_info['best_bound'] = solver.best_objective_bound if status == cp_model.UNKNOWN else None
            reason = 'solver_' + solver.status_name(status).lower()
        self.last_info.update(fallback=True, fallback_reason=reason, fallback_policy='feasible_edf')
        return EDFPlanner().decide(state)
