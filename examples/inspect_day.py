"""Verify and inspect the bundled synthetic day without starting a simulation."""
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.metrics import compute_metrics

EXAMPLE = Path(__file__).resolve().parent / "recorded_day"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def digest(obj):
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    provenance = read_json(EXAMPLE / "provenance.json")
    trajectory = read_json(EXAMPLE / "trajectory.json")
    config = read_json(EXAMPLE / "config.json")
    check(digest(trajectory) == provenance["trajectory_sha256"], "Trajectory hash differs")
    check(digest(config) == provenance["configuration_sha256"], "Configuration hash differs")
    task_ids = {task["id"] for task in trajectory["tasks"]}
    print("Recorded synthetic day, emergency scenario, seed 101")
    for method in provenance["methods"]:
        folder = EXAMPLE / method
        saved = read_json(folder / "metrics.json")
        tasks = read_json(folder / "tasks.json")
        stats = read_json(folder / "stats.json")
        decisions = read_jsonl(folder / "decision_log.jsonl")
        events = read_jsonl(folder / "event_log.jsonl")
        check(saved["algorithm"] == method and saved["run_status"] == "completed", f"{method}: incorrect run record")
        check(saved["trajectory_sha256"] == digest(trajectory), f"{method}: trajectory mismatch")
        check(saved["config_sha256"] == digest(config), f"{method}: configuration mismatch")
        check({t["id"] for t in tasks} == task_ids, f"{method}: task set differs")
        clock = 0
        for decision in decisions:
            check(decision["time_s"] == clock and decision["next_time_s"] > clock,
                  f"{method}: non-contiguous decision log")
            clock = decision["next_time_s"]
        check(clock == config["horizon_s"], f"{method}: incomplete day")
        metrics = compute_metrics(tasks, clock, config["model"]["critical_priority"], decisions, stats, events)
        for key, value in metrics.items():
            actual = saved[key]
            matches = math.isclose(actual, value, rel_tol=1e-10, abs_tol=1e-12) if (
                type(actual) in (int, float) and type(value) in (int, float)
            ) else actual == value
            check(matches, f"{method}: recomputed metric differs for {key}")
        print(f"{method:8} critical on time {metrics['critical_ontime_count']}/{metrics['critical_task_count']}, "
              f"unfinished {metrics['unfinished_tasks']}/{metrics['task_count']}, "
              f"emergency responses {metrics['emergency_responded_count']}/{metrics['emergency_count']}, "
              f"fallbacks {metrics['planner_fallback_count']}")
    print("PASS: shared inputs, full day logs and recomputed metrics for both methods")


if __name__ == "__main__":
    main()
