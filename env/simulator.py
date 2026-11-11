"""Original event-driven home-care scheduling simulator, public protocol 0.1."""
import copy
import math
import networkx as nx
from env.scenario import generate_trajectory, validate_trajectory
from env.safety import task_energy, charge_feasible, validate_action, fallback_action, valid_actions

EPS = 1e-9


class HomeCareEnv:
    def __init__(self, config, trajectory=None):
        self.config = copy.deepcopy(config)
        self._supplied_trajectory = copy.deepcopy(trajectory)
        self.m = self.config['model']
        self._validate_config()
        self.graph = nx.Graph()
        for a, b, length in sorted(self.m['edges']):
            self.graph.add_edge(a, b, weight=length)
        if not nx.is_connected(self.graph):
            raise ValueError('map must be connected')
        self.paths = dict(nx.all_pairs_dijkstra_path(self.graph, weight='weight'))
        self.distances = dict(nx.all_pairs_dijkstra_path_length(self.graph, weight='weight'))
        self.times = {a: {b: sum(math.ceil(self.graph[u][v]['weight'] / self.m['speed_m_s']) for u, v in zip(path, path[1:])) for b, path in targets.items()} for a, targets in self.paths.items()}

    def _validate_config(self):
        m = self.m
        if type(self.config['horizon_s']) is not int or self.config['horizon_s'] <= 0:
            raise ValueError('positive integer horizon required')
        for key in ('speed_m_s', 'capacity_wh', 'charge_wh_s'):
            if not math.isfinite(m[key]) or m[key] <= 0:
                raise ValueError(f'{key} must be positive and finite')
        for key in ('initial_energy_wh', 'move_wh_m', 'service_wh_s', 'idle_wh_s', 'reserve_wh'):
            if not math.isfinite(m[key]) or m[key] < 0:
                raise ValueError(f'{key} must be nonnegative and finite')
        if m['initial_energy_wh'] > m['capacity_wh'] or not 0 <= m['low_battery_fraction'] < m['charge_target_fraction'] <= 1:
            raise ValueError('invalid battery bounds')
        if (type(m['default_wait_s']) is not int or type(m['max_wait_s']) is not int
                or not 1 <= m['default_wait_s'] <= m['max_wait_s']):
            raise ValueError('invalid wait configuration')
        if any(not math.isfinite(d) or d <= 0 for _, _, d in m['edges']):
            raise ValueError('edge distances must be positive and finite')
        nodes = {n for a, b, _ in m['edges'] for n in (a, b)}
        if m['dock'] not in nodes or m['start_node'] not in nodes:
            raise ValueError('dock/start must be map nodes')

    def reset(self, seed):
        self._trajectory = copy.deepcopy(self._supplied_trajectory) if self._supplied_trajectory is not None else generate_trajectory(self.config, seed)
        self._future = validate_trajectory(self._trajectory, self.config)
        self._index = 0
        self.tasks = {}
        self.time = 0
        self.horizon = self.config['horizon_s']
        self.energy = float(self.m['initial_energy_wh'])
        self.position = dict(node=self.m['start_node'], edge=None, offset_m=0.0, remaining_distance_m=0.0)
        self.mode = 'idle'
        self.active = None
        self.intent = None
        self._segments = []
        self._segment = None
        self._remaining = 0
        self.done = False
        self.event_log = []
        self.action_log = []
        self.stats = dict(charge_count=0, charge_time_s=0, min_energy_wh=self.energy, battery_violations=0, robot_failed=False)
        self.stats.update(energy_consumed_wh=0.0, travel_distance_m=0.0,
                          mode_time_s={k: 0 for k in ('idle', 'moving', 'serving', 'waiting', 'charging', 'failed')})
        self._release()
        self._enforce_safety('reset')
        return self._state()

    def _log(self, kind, **kwargs):
        self.event_log.append(dict(time_s=self.time, type=kind, **kwargs))

    def _route(self, target):
        """Choose physical shortest-distance route, including either end of current edge."""
        p = self.position
        if p['edge'] is None:
            origin = p['node']
            prefix = []
            path = self.paths[origin][target]
        else:
            a, b = p['edge']
            length = self.graph[a][b]['weight']
            options = [(p['offset_m'] + self.distances[a][target], a, p['offset_m']),
                       (length - p['offset_m'] + self.distances[b][target], b, length - p['offset_m'])]
            _, origin, partial = min(options, key=lambda x: (round(x[0], 9), x[1]))
            prefix = [] if partial < EPS else [(a, b, p['offset_m'], 0.0 if origin == a else float(length), math.ceil(partial / self.m['speed_m_s']))]
            path = self.paths[origin][target]
        for a, b in zip(path, path[1:]):
            length = float(self.graph[a][b]['weight'])
            prefix.append((a, b, 0.0, length, math.ceil(length / self.m['speed_m_s'])))
        return prefix

    def _public_model(self):
        model = copy.deepcopy(self.m)
        model['distances_m'] = copy.deepcopy(self.distances)
        model['travel_times_s'] = copy.deepcopy(self.times)
        routes = {node: self._route(node) for node in sorted(self.graph.nodes)}
        model['distance_from_position_m'] = {n: sum(abs(s[3] - s[2]) for s in route) for n, route in routes.items()}
        model['travel_time_from_position_s'] = {n: sum(s[4] for s in route) for n, route in routes.items()}
        return model

    def _state(self):
        state = dict(schema_version='0.1', time_s=self.time, horizon_s=self.horizon,
                     position=copy.deepcopy(self.position), battery=dict(energy_wh=self.energy, capacity_wh=self.m['capacity_wh']),
                     mode=self.mode, active_task_id=self.active,
                     pending_tasks=[copy.deepcopy(t) for t in self.tasks.values() if t['status'] not in ('completed', 'unresponded')],
                     current_events=[copy.deepcopy(t) for t in self.tasks.values() if t['type'] == 'emergency'],
                     public_model=self._public_model())
        state['valid_actions'] = valid_actions(state)
        return state

    def export_tasks(self):
        return copy.deepcopy(list(self.tasks.values()))

    def _release(self):
        while self._index < len(self._future) and self._future[self._index]['release_time_s'] <= self.time:
            task = copy.deepcopy(self._future[self._index])
            task.update(remaining_service_s=task['service_duration_s'], status='pending', completed_at_s=None, response_at_s=None, failure_reason=None)
            self.tasks[task['id']] = task
            self._index += 1
            self._log('task_released', task_id=task['id'], task_type=task['type'])
            if self.stats['robot_failed'] and task['type'] == 'emergency':
                task.update(status='unresponded', failure_reason='robot_failed')

    def _interrupt(self):
        if self.active is not None:
            task = self.tasks[self.active]
            if task['status'] == 'active':
                task['status'] = 'pending'
                self._log('task_preempted', task_id=self.active, remaining_service_s=task['remaining_service_s'], position=copy.deepcopy(self.position))
        self.active = None
        self._segments = []
        self._segment = None
        self.intent = None

    def _start_next_segment(self):
        if self._segments:
            a, b, start, end, duration = self._segments.pop(0)
            self._segment = dict(a=a, b=b, start=start, end=end, duration=duration, elapsed=0)
            self.position = dict(node=None, edge=[a, b], offset_m=start, remaining_distance_m=abs(end-start))
            self.mode = 'moving'
            self._remaining = duration
        else:
            self._segment = None
            self._arrive()

    def _arrive(self):
        if self.intent == 'charge':
            target = self.m['capacity_wh'] * self.m['charge_target_fraction']
            if self.energy < target - EPS:
                self.mode = 'charging'
                self.stats['charge_count'] += 1
                self._remaining = max(1, math.ceil((target - self.energy) / self.m['charge_wh_s']))
                self._log('charge_started')
            else:
                self.intent = None
                self._start_wait(self.m['default_wait_s'])
        elif self.active:
            task = self.tasks[self.active]
            if task['type'] == 'emergency' and task['response_at_s'] is None:
                task['response_at_s'] = self.time
                self._log('emergency_response', task_id=self.active, response_time_s=self.time-task['release_time_s'])
            if self.time < task['earliest_start_s']:
                self.mode = 'waiting'
                self._remaining = task['earliest_start_s'] - self.time
            else:
                self.mode = 'serving'
                self._remaining = task['remaining_service_s']
        else:
            self.mode = 'idle'

    def _start_wait(self, duration):
        self.mode = 'waiting'
        self._remaining = duration

    def _apply(self, action):
        if self.mode in ('failed', 'finished'):
            return
        kind = action['type']
        if kind == 'continue' or (kind == 'serve' and action['task_id'] == self.active):
            if self.mode == 'idle':
                self._start_wait(self.m['default_wait_s'])
            return
        if kind == 'charge' and self.intent == 'charge':
            return
        self._interrupt()
        if kind == 'serve':
            self.active = action['task_id']
            self.tasks[self.active]['status'] = 'active'
            self.intent = 'serve'
            self._segments = self._route(self.tasks[self.active]['location'])
            self._log('task_selected', task_id=self.active)
            self._start_next_segment()
        elif kind == 'charge':
            self.intent = 'charge'
            self._segments = self._route(self.m['dock'])
            self._start_next_segment()
        else:
            self._start_wait(action['duration_s'])

    def _fail(self, reason):
        if self.stats['robot_failed']:
            return
        self._interrupt()
        self.mode = 'failed'
        self.stats['robot_failed'] = True
        for task in self.tasks.values():
            if task['type'] == 'emergency' and task['status'] != 'completed':
                task['status'] = 'unresponded'
                task['failure_reason'] = reason
        self._log('robot_failed', reason=reason)

    def _enforce_safety(self, phase):
        if self.mode in ('failed', 'finished'):
            return None
        state = self._state()
        if not charge_feasible(state):
            self._fail('unreachable_dock')
            return 'unreachable_dock'
        if self.energy <= EPS and self.position['node'] != self.m['dock']:
            self._fail('battery_depleted')
            return 'battery_depleted'
        current = self.tasks.get(self.active)
        emergencies = sorted((t for t in self.tasks.values() if t['type'] == 'emergency' and t['status'] in ('pending', 'active')), key=lambda t: (-t['priority'], t['release_time_s'], t['id']))
        if current and current['type'] == 'emergency':
            return None  # Queue discipline: an active emergency finishes first.
        if current and self.mode == 'serving' and not current['preemptible']:
            if emergencies:
                self._log('safety_deferred', phase=phase, reason='nonpreemptible_service', task_id=current['id'])
            return None
        rejected = False
        for task in emergencies:
            if task_energy(state, task) <= self.energy + EPS:
                action = dict(type='serve', task_id=task['id'])
                self._apply(action)
                self._log('safety_override', phase=phase, reason='forced_emergency', action=action)
                return 'forced_emergency'
            task['status'] = 'unresponded'
            task['failure_reason'] = 'energy_infeasible_emergency'
            rejected = True
            self._log('emergency_unresponded', task_id=task['id'], reason=task['failure_reason'])
        return_energy = state['public_model']['distance_from_position_m'][self.m['dock']] * self.m['move_wh_m']
        threshold = max(self.m['capacity_wh'] * self.m['low_battery_fraction'], return_energy + self.m['reserve_wh'])
        # One-second lookahead prevents crossing the return boundary when the
        # residual margin is smaller than a discrete powered tick. A moving
        # second may both consume energy and increase distance to the dock.
        guard = self._energy_rate() * (2 if self.mode == 'moving' else 1)
        if rejected or self.energy <= threshold + guard + EPS:
            if self.intent != 'charge':
                reason = 'energy_infeasible_emergency' if rejected else 'low_battery_or_return_reserve'
                self._apply({'type': 'charge'})
                self._log('safety_override', phase=phase, reason=reason, action={'type': 'charge'})
                return reason
        return None

    def _current_action(self):
        if self.active:
            return {'type': 'serve', 'task_id': self.active}
        if self.intent == 'charge':
            return {'type': 'charge'}
        if self.mode == 'waiting':
            return {'type': 'wait', 'duration_s': self._remaining}
        return {'type': 'continue'}

    def _energy_rate(self):
        if self.mode == 'moving':
            seg = self._segment
            return abs(seg['end'] - seg['start']) / seg['duration'] * self.m['move_wh_m']
        if self.mode == 'serving':
            return self.m['service_wh_s']
        if self.mode in ('waiting', 'idle'):
            return self.m['idle_wh_s']
        return 0.0

    def _advance(self, dt):
        rate = self._energy_rate()
        self.stats['mode_time_s'][self.mode] += dt
        if self.mode == 'moving':
            seg = self._segment
            self.stats['travel_distance_m'] += abs(seg['end']-seg['start']) / seg['duration'] * dt
            seg['elapsed'] += dt
            self.position['offset_m'] = seg['start'] + (seg['end']-seg['start']) * seg['elapsed']/seg['duration']
            self.position['remaining_distance_m'] = abs(seg['end']-self.position['offset_m'])
        elif self.mode == 'serving':
            self.tasks[self.active]['remaining_service_s'] -= dt
        elif self.mode == 'charging':
            self.energy = min(self.m['capacity_wh'] * self.m['charge_target_fraction'], self.energy + dt*self.m['charge_wh_s'])
            self.stats['charge_time_s'] += dt
        consumption = rate * dt
        if consumption > self.energy + EPS:
            self.stats['battery_violations'] += 1
            raise RuntimeError('energy conservation violated: unpowered activity')
        self.stats['energy_consumed_wh'] += consumption
        self.energy = max(0.0, self.energy - consumption)
        self.stats['min_energy_wh'] = min(self.stats['min_energy_wh'], self.energy)
        self.time += dt
        self._remaining -= dt

    def _settle(self):
        if self.mode in ('failed', 'finished'):
            return
        if self._remaining <= 0:
            if self.mode == 'moving':
                seg = self._segment
                node = seg['a'] if abs(seg['end']) < EPS else seg['b']
                self.position = dict(node=node, edge=None, offset_m=0.0, remaining_distance_m=0.0)
                self._log('node_arrival', node=node)
                self._start_next_segment()
            elif self.mode == 'serving':
                task = self.tasks[self.active]
                task.update(status='completed', completed_at_s=self.time, remaining_service_s=0)
                self._log('task_completed', task_id=self.active)
                self.active = None
                self.intent = None
                self.mode = 'idle'
            elif self.mode == 'charging':
                self._log('charge_completed')
                self.intent = None
                self.mode = 'idle'
            elif self.mode == 'waiting':
                if self.active:
                    self._arrive()
                else:
                    self.mode = 'idle'

    def step(self, action):
        if self.done:
            return self._state(), 0.0, True, dict(elapsed_s=0, events=[], requested_action=copy.deepcopy(action), executed_action={'type':'continue'}, invalid_action='episode_finished', override_reason=None, termination_reason='horizon')
        start, log_start = self.time, len(self.event_log)
        state = self._state()
        invalid = validate_action(state, action)
        actual = fallback_action(state) if invalid else copy.deepcopy(action)
        override = None
        # Keep active emergency commitments; every method shares this enforced layer.
        current = self.tasks.get(self.active)
        if current and current['type'] == 'emergency' and actual != {'type': 'continue'} and actual != {'type':'serve', 'task_id':self.active}:
            actual = {'type': 'continue'}
            override = 'active_emergency_commitment'
            self._log('safety_override', phase='action', reason=override, action=actual)
        self._apply(actual)
        safety_reason = self._enforce_safety('action')
        if safety_reason:
            override = safety_reason
            actual = self._current_action()
        rate = self._energy_rate()
        if rate > 0 and self.energy + EPS < rate:
            self._fail('insufficient_energy_for_one_second')
            override = 'insufficient_energy_for_one_second'
            actual = {'type': 'continue'}
        limits = [self.horizon - self.time]
        if self._index < len(self._future):
            limits.append(self._future[self._index]['release_time_s'] - self.time)
        if self.mode != 'failed':
            limits.append(max(1, self._remaining))
            rate = self._energy_rate()
            if rate > 0:
                limits.append(max(1, math.floor((self.energy + EPS) / rate)))
                if self.intent != 'charge' and not (self.active and (self.tasks[self.active]['type']=='emergency' or not self.tasks[self.active]['preemptible'])):
                    # Conservative return energy upper bound grows by at most distance travelled.
                    ret = self._public_model()['distance_from_position_m'][self.m['dock']] * self.m['move_wh_m']
                    bound_rate = rate * (2 if self.mode == 'moving' else 1)
                    thresholds = [(self.energy-self.m['capacity_wh']*self.m['low_battery_fraction'])/rate,
                                  (self.energy-ret-self.m['reserve_wh'])/bound_rate]
                    for x in thresholds:
                        if x > 0:
                            limits.append(max(1, math.floor(x)))
        dt = max(1, min(limits))
        self._advance(dt)
        self._settle()
        self._release()
        if self.time >= self.horizon:
            self.done = True
            if self.mode != 'failed':
                self.mode = 'finished'
            self._log('horizon_reached')
        else:
            self._enforce_safety('event')
        info = dict(elapsed_s=self.time-start, events=copy.deepcopy(self.event_log[log_start:]), requested_action=copy.deepcopy(action), executed_action=actual,
                    invalid_action=invalid, override_reason=override, termination_reason=('robot_failed' if self.stats['robot_failed'] else 'horizon') if self.done else None)
        self.action_log.append(copy.deepcopy(info))
        return self._state(), 0.0, self.done, info
