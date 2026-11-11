# How AI Can Manage and Schedule Long-Horizon Home Care Robot Tasks

Research track project for WU course 0718, Agentic AI in Strategy & Innovation.

Team members are **Zihan Wang** and **Teng Ye**.

We study how one home care robot can manage a day's workload as service requests arrive. At each decision point, a scheduler can choose the next task, wait, send the robot to charge or continue its current action. Deadlines, priorities, travel and battery limits shape those choices. The service operations question is when more advanced scheduling improves timely completion of critical tasks and how much decision time it adds.

## Open the example

[View the recorded comparison and decision logs](examples/recorded_day/README.md) directly in your browser. A deadline rule and a rolling optimizer faced the same synthetic task and emergency arrivals in a single development scenario.

## Project status

| Method | What is available |
| --- | --- |
| Earliest-deadline-first (EDF) | Runnable rule baseline that selects among feasible tasks by deadline |
| Rolling optimization | Runnable OR-Tools planner that replans from currently known tasks and records solver status and fallback decisions |
| Learned task selection | Implemented imitation-learning pipeline. It is designed to learn from optimizer demonstrations and rank feasible ordinary tasks. Training and held-out evaluation are pending |

All schedulers use the same observable information and share the same energy checks and emergency response rules. Their decisions can affect the robot's location, available energy and service backlog when an emergency arrives. The repository also includes an offline LLM adapter for mock and replay tests. No live LLM results are included. Findings from the simulator apply to the modeled household and its assumptions.

## Test the recorded example

With Python 3.13, run this from the repository root:

```bash
python3.13 examples/inspect_day.py
```

The script uses the standard library to verify the shared trajectory, check that both decision logs cover the full day, recompute the metrics and print a short comparison. It reads the saved records.

## Run the same scenario locally

After cloning or downloading the repository, create an environment and install the pinned dependencies from the repository root:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
```

Then run either baseline against the bundled trajectory:

```bash
.venv/bin/python -m experiments.run --config examples/recorded_day/config.json --trajectory examples/recorded_day/trajectory.json --seed 101 --scenario emergency --algorithm edf
.venv/bin/python -m experiments.run --config examples/recorded_day/config.json --trajectory examples/recorded_day/trajectory.json --seed 101 --scenario emergency --algorithm ortools
```

Each command writes a new folder under `results/single/`. Start with `metrics.json` for service outcomes and `decision_log.jsonl` for individual decisions. The optimization planner uses a wall-clock time limit, so solver outcomes and decision timing can vary across computers. On Windows, use `py -3.13` for the system Python commands and `.venv\Scripts\python.exe` for the virtual environment commands.

## Source and dependencies

The simulator and scheduling code were developed for this project. [Dependency notices](docs/DEPENDENCIES.md) identify the libraries and their license texts. The project code has no separate reuse license at this stage.
