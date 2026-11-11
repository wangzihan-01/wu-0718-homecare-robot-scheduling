"""Original project code: diagnostic plots of complete simulation-day rows."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_summary(summary_csv, output):
    with Path(summary_csv).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    algorithms = list(dict.fromkeys(r['algorithm'] for r in rows))
    scenarios = list(dict.fromkeys(r['scenario'] for r in rows))
    completed = [r for r in rows if r.get('run_status') == 'completed']
    measures = [('critical_ontime_rate', 'Critical ordinary tasks on time', 'Proportion'),
                ('unfinished_task_rate', 'Unfinished ordinary tasks', 'Proportion'),
                ('emergency_response_median_s', 'Emergency response (day median)', 'Seconds; absent events = NA'),
                ('decision_wall_mean_s', 'Decision wall time (day mean)', 'Milliseconds')]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), constrained_layout=True)
    width = 0.8 / max(1, len(algorithms))
    colors = plt.get_cmap('tab10').colors
    for ax, (metric, title, ylabel) in zip(axes.flat, measures):
        for ai, algorithm in enumerate(algorithms):
            for si, scenario in enumerate(scenarios):
                values = [float(r[metric]) for r in completed
                          if r['algorithm'] == algorithm and r['scenario'] == scenario and r.get(metric) not in (None, '')]
                if metric == 'decision_wall_mean_s':
                    values = [v * 1000 for v in values]
                x = si - 0.4 + width / 2 + ai * width
                if values:
                    ax.bar(x, mean(values), width=width * 0.9, color=colors[ai], alpha=.55,
                           label=algorithm if si == 0 else None)
                    offsets = [(i - (len(values)-1)/2) * width * .14 for i in range(len(values))]
                    ax.scatter([x + d for d in offsets], values, color=colors[ai], s=16, zorder=3)
                else:
                    ax.text(x, 0.02, 'NA', transform=ax.get_xaxis_transform(), ha='center', fontsize=7, rotation=90)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(scenarios)), scenarios, rotation=12)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=.2)
        if metric.endswith('_rate'):
            ax.set_ylim(0, 1.05)
    handles = [plt.Line2D([0], [0], color=colors[i], linewidth=7, label=a) for i, a in enumerate(algorithms)]
    fig.legend(handles=handles, loc='outside lower center', ncol=max(1, len(algorithms)), frameon=False)
    fig.suptitle('Smoke pipeline validation only — no superiority inference\n'
                 f'Bars: day means; dots: individual development days. Completed {len(completed)}/{len(rows)} runs.', fontsize=13)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    fig.savefig(output.with_suffix('.pdf'))
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('summary_csv')
    parser.add_argument('--output', default='results/smoke_overview.png')
    args = parser.parse_args()
    print(plot_summary(args.summary_csv, args.output).resolve())


if __name__ == '__main__':
    main()
