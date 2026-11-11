"""One-command future CPU pilot; default only previews, never trains implicitly."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import traceback

from experiments.imitation import collect, fit_dataset, validate_spec
from experiments.run import ROOT, load_json, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', default='configs/imitation_training.json')
    parser.add_argument('--evaluation', default='configs/course_ai_quick.json')
    parser.add_argument('--output', default='results/learned_pilot')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    training, evaluation = load_json(args.training), load_json(args.evaluation)
    validate_spec(training)
    if (not set(evaluation['seeds']) <= set(training['reserved_evaluation_seeds'])
            or evaluation['scenarios'] != training['scenarios']
            or evaluation['algorithms'] != ['edf','ortools','learned']):
        raise ValueError('pilot must use the declared matched scenarios and held-out evaluation seeds')
    teacher_days = len(training['scenarios']) * (len(training['training_seeds']) + len(training['validation_seeds']))
    evaluation_days = len(evaluation['scenarios']) * len(evaluation['seeds']) * len(evaluation['algorithms'])
    if evaluation_days != evaluation['expected_runs']:
        raise ValueError('evaluation count differs from its declaration')
    if not args.execute:
        print(f'PREVIEW ONLY: {teacher_days} teacher days + {evaluation_days} evaluation days; no simulation or training started.')
        print('After explicitly resuming, use the same command with --execute. Runtime is not measured.')
        return 0
    folder = Path(args.output)/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder.mkdir(parents=True, exist_ok=False)
    stage = 'teacher_collection'
    save_json(folder/'training_spec.json', training)
    save_json(folder/'evaluation_spec.json', evaluation)
    try:
        result = collect(training, folder/'teacher_data')
        if result['status'] != 'completed':
            raise RuntimeError('teacher collection failed; see retained day/error logs')
        stage = 'model_fit'
        model = folder/'model.json'
        fit_dataset(training, folder/'teacher_data', model)
        # Derive both evaluation baselines and the student from the SAME base used by the teacher.
        base = load_json(ROOT/training['base_config'])
        base['planners']['ortools'].update(training['teacher_overrides'])
        base['planners']['learned'] = {'model_path': str(model.resolve()), 'allow_test_model': False}
        base_path = folder/'evaluation_base.json'
        save_json(base_path, base)
        evaluation['base_config'] = str(base_path.resolve())
        stage = 'evaluation'
        from experiments.batch import run_batch
        batch, rows = run_batch(evaluation, folder/'evaluation')
        if any(row['run_status'] != 'completed' for row in rows):
            raise RuntimeError('some evaluation runs failed; all rows were retained')
        save_json(folder/'completion.json', {'status':'completed', 'evaluation_batch':str(batch.resolve())})
    except Exception:
        save_json(folder/'error.json', {'status':'failed', 'stage':stage,'traceback':traceback.format_exc()})
        print(f'FAILED at {stage}; artifacts retained at {folder.resolve()}')
        return 1
    print(f'Pilot artifacts: {folder.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
