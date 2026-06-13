#!/usr/bin/env python3
"""Export TensorBoard training curves to PNG for the thesis."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LOG_DIRS = {
    'Baseline (SR)': 'runs/baseline',
    'SelectiveNet': 'runs/selective_net',
    'Evidential (EDL)': 'runs/evidential',
    'Hybrid (3D-ResNet-EDL)': 'runs/hybrid',
}

METRIC_GROUPS = {
    'loss': ['Train/Epoch_Loss', 'Val/Loss'],
    'auc': ['Val/AUC'],
    'sens_at_spec': ['Val/Sens@80%Spec', 'Val/sensitivity_at_target_spec'],
}


def load_scalars(log_dir: Path, tag: str):
    ea = event_accumulator.EventAccumulator(str(log_dir))
    ea.Reload()
    if tag not in ea.Tags().get('scalars', []):
        return None, None
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def export_curves(log_dirs: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for group_name, tags in METRIC_GROUPS.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False

        for model_name, log_path in log_dirs.items():
            log_dir = Path(log_path)
            if not log_dir.exists():
                continue
            for tag in tags:
                steps, values = load_scalars(log_dir, tag)
                if steps:
                    ax.plot(steps, values, label=f'{model_name} ({tag})')
                    plotted = True
                    break

        if not plotted:
            plt.close(fig)
            continue

        ax.set_xlabel('Epoch')
        ax.set_ylabel(group_name)
        ax.set_title(f'Training curves: {group_name}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = output_dir / f'training_curve_{group_name}.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {out}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='results/training_curves')
    args = parser.parse_args()
    export_curves(LOG_DIRS, Path(args.output_dir))


if __name__ == '__main__':
    main()
