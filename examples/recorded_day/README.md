# Recorded emergency scenario

This page shows a pair of development runs on one synthetic 12-hour household day, using seed 101 and a 40 Wh battery. Both planners received the same [task and emergency arrivals](trajectory.json) and [configuration](config.json).

| Method | Critical ordinary tasks on time | Unfinished ordinary tasks | Emergencies responded to | Median response | Mean decision time | Planner fallbacks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EDF | 12/12 | 1/53 | 7/7 | 24 s | 0.0752 ms | 0 |
| Rolling optimization | 12/12 | 1/53 | 7/7 | 24 s | 0.57 ms | 4 |

Emergency response time runs from an event's arrival to the robot's arrival on site. Both methods share the simulator's emergency handling rules. Decision time was measured on the original computer. This single day illustrates the workflow and cannot support general performance conclusions.

- EDF [metrics](edf/metrics.json), [tasks](edf/tasks.json), [decisions](edf/decision_log.jsonl), [events](edf/event_log.jsonl)
- Rolling optimization [metrics](ortools/metrics.json), [tasks](ortools/tasks.json), [decisions](ortools/decision_log.jsonl), [events](ortools/event_log.jsonl)
- [Source and input checksums](provenance.json)
- [Instructions for checking the records or rerunning the scenario](../../README.md#test-the-recorded-example)
