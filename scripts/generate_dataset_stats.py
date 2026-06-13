#!/usr/bin/env python3
"""Generate dataset statistics table and figures for the thesis."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    root = Path(__file__).resolve().parent.parent
    inventory_path = root / 'Data baseline' / 'metadata' / 'inventory_report.json'
    metadata_path = root / 'data_metadata_adni.csv'
    out_dir = root / 'results' / 'dataset_stats'
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(inventory_path) as f:
        inventory = json.load(f)

    df = pd.read_csv(metadata_path)

    rows = []
    for split in ['train', 'val', 'test']:
        sub = df[df['split'] == split]
        cn = (sub['label'] == 0).sum()
        mci = (sub['label'] == 1).sum()
        baseline = (sub['cohort'] == 'baseline').sum() if 'cohort' in sub.columns else 0
        adni2 = (sub['cohort'] == 'ADNI2').sum() if 'cohort' in sub.columns else 0
        rows.append({
            'Split': split,
            'N': len(sub),
            'CN': cn,
            'MCI': mci,
            'CN%': f'{100 * cn / len(sub):.1f}' if len(sub) else '0',
            'MCI%': f'{100 * mci / len(sub):.1f}' if len(sub) else '0',
            'Baseline cohort': baseline,
            'ADNI2 cohort': adni2,
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / 'dataset_split_summary.csv', index=False)

    overview = pd.DataFrame([{
        'Total subjects': inventory.get('unique_disk_subjects', len(df)),
        'Baseline subjects': inventory.get('baseline_subjects', ''),
        'ADNI2 subjects': inventory.get('adni2_subjects', ''),
        'Overlap (both cohorts)': inventory.get('overlap_subjects', ''),
        'Unlabeled on disk': inventory.get('unlabeled_on_disk', 0),
        'Total CN': (df['label'] == 0).sum(),
        'Total MCI': (df['label'] == 1).sum(),
    }])
    overview.to_csv(out_dir / 'dataset_overview.csv', index=False)

    # Class distribution bar chart
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    labels = ['CN', 'MCI']
    counts = [(df['label'] == 0).sum(), (df['label'] == 1).sum()]
    axes[0].bar(labels, counts, color=['#4C72B0', '#DD8452'])
    axes[0].set_title('Class distribution (full dataset)')
    axes[0].set_ylabel('Subjects')
    for i, v in enumerate(counts):
        axes[0].text(i, v + 5, str(v), ha='center')

    split_names = summary['Split'].tolist()
    cn_vals = summary['CN'].tolist()
    mci_vals = summary['MCI'].tolist()
    x = range(len(split_names))
    w = 0.35
    axes[1].bar([i - w / 2 for i in x], cn_vals, w, label='CN', color='#4C72B0')
    axes[1].bar([i + w / 2 for i in x], mci_vals, w, label='MCI', color='#DD8452')
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(split_names)
    axes[1].set_title('Class distribution per split')
    axes[1].set_ylabel('Subjects')
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(out_dir / 'dataset_distribution.png', dpi=150, bbox_inches='tight')
    fig.savefig(out_dir / 'dataset_distribution.pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"Dataset stats saved to {out_dir}")
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
