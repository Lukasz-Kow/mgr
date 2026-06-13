#!/usr/bin/env python3
"""Run all supplementary thesis artifact generators."""

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    ['scripts/generate_dataset_stats.py'],
    ['scripts/export_training_curves.py'],
    ['scripts/generate_reliability_diagram.py'],
    ['scripts/generate_gradcam.py', '--model', 'hybrid', '--max-samples', '3'],
    ['scripts/generate_gradcam.py', '--model', 'baseline', '--max-samples', '2'],
]


def main():
    root = Path(__file__).resolve().parent.parent
    python = sys.executable

    for cmd_args in SCRIPTS:
        cmd = [python] + cmd_args
        print(f'\n>>> {" ".join(cmd)}')
        result = subprocess.run(cmd, cwd=root)
        if result.returncode != 0:
            print(f'Warning: {cmd_args[0]} exited with code {result.returncode}')

    print('\nThesis artifacts generation complete.')


if __name__ == '__main__':
    main()
