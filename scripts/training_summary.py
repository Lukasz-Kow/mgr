#!/usr/bin/env python3

"""Print and save training summary from checkpoint val_metrics."""



import sys

import csv

import argparse

from pathlib import Path



import torch



sys.path.insert(0, str(Path(__file__).resolve().parent.parent))



from src.training.eval_utils import load_evaluation_config



CHECKPOINTS_PHASE1 = {

    'Baseline': 'checkpoints/baseline/best_model.pth',

    'SelectiveNet': 'checkpoints/selective_net/best_model.pt',

    'Evidential (EDL)': 'checkpoints/evidential/best_model.pt',

    'Hybrid (3D-ResNet-EDL)': 'checkpoints/hybrid/best_model.pt',

}



CHECKPOINTS_PHASE2 = {
    'Baseline (MONAI)': 'checkpoints/baseline_monai/best_model.pth',
    'SelectiveNet (MONAI)': 'checkpoints/selective_net_monai/best_model.pt',
    'Evidential (MONAI)': 'checkpoints/evidential_monai/best_model.pt',
    'Hybrid (MONAI)': 'checkpoints/hybrid_monai/best_model.pt',
}





def _backbone_from_ckpt(ckpt) -> str:

    if ckpt is None:

        return ''

    cfg = ckpt.get('config', {})

    bb = cfg.get('model', {}).get('backbone', {})

    if bb.get('type') == 'monai':

        return f"MONAI+{bb.get('pretrained', 'medicalnet')}"

    return 'simple CNN'





def load_ckpt(path: Path):

    if not path.exists():

        return None

    return torch.load(path, map_location='cpu', weights_only=False)





def summarize_checkpoints(checkpoints: dict, target_spec: float) -> list:

    rows = []

    for name, rel_path in checkpoints.items():

        path = Path(rel_path)

        ckpt = load_ckpt(path)

        if ckpt is None:

            print(f"  [MISSING] {name}: {path}")

            rows.append({

                'Model': name,

                'Backbone': '',

                'Status': 'MISSING',

                'Epoch': '',

                f'Sens@{target_spec:.0%}Spec (val)': '',

                'Actual Spec (val)': '',

                'Balanced Acc (val)': '',

                'AUC (val)': '',

                'Checkpoint': str(path),

            })

            continue



        vm = ckpt.get('val_metrics', {})

        ms = vm.get('metrics_at_target_spec', {})

        row = {

            'Model': name,

            'Backbone': _backbone_from_ckpt(ckpt),

            'Status': 'OK',

            'Epoch': ckpt.get('epoch', ''),

            f'Sens@{target_spec:.0%}Spec (val)': f"{ms.get('sensitivity', vm.get('sensitivity_at_target_spec', 0)):.4f}",

            'Actual Spec (val)': f"{ms.get('actual_specificity', vm.get('specificity_at_target_spec', 0)):.4f}",

            'Balanced Acc (val)': f"{vm.get('balanced_accuracy', 0):.4f}",

            'AUC (val)': f"{vm.get('auc', 0):.4f}",

            'Checkpoint': str(path),

        }

        rows.append(row)

        print(

            f"  {name} [{row['Backbone']}]: epoch={row['Epoch']} "

            f"Sens@{target_spec:.0%}Spec={row[f'Sens@{target_spec:.0%}Spec (val)']} "

            f"Spec={row['Actual Spec (val)']} "

            f"BalAcc={row['Balanced Acc (val)']} AUC={row['AUC (val)']}"

        )

    return rows





def main():

    parser = argparse.ArgumentParser(description='Training summary from checkpoints')

    parser.add_argument('--phase2', action='store_true', help='Include Phase 2 MONAI checkpoints')

    args = parser.parse_args()



    eval_cfg = load_evaluation_config()

    target_spec = eval_cfg.get('target_specificity', 0.80)



    rows = []

    print("=" * 90)

    print(f"  TRAINING SUMMARY Phase 1 (Sens@{target_spec:.0%}Spec on validation)")

    print("=" * 90)

    rows.extend(summarize_checkpoints(CHECKPOINTS_PHASE1, target_spec))



    if args.phase2:

        print("\n" + "=" * 90)

        print(f"  TRAINING SUMMARY Phase 2 — MONAI (Sens@{target_spec:.0%}Spec on validation)")

        print("=" * 90)

        rows.extend(summarize_checkpoints(CHECKPOINTS_PHASE2, target_spec))



    results_dir = Path('results')

    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / 'training_summary.csv'

    txt_path = results_dir / 'training_summary.txt'



    if rows:

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:

            writer = csv.DictWriter(f, fieldnames=rows[0].keys())

            writer.writeheader()

            writer.writerows(rows)



    with open(txt_path, 'w', encoding='utf-8') as f:

        f.write(f"Training summary (Sens@{target_spec:.0%}Spec on validation)\n")

        f.write("=" * 60 + "\n")

        for row in rows:

            f.write(f"{row['Model']}: {row}\n")



    print("=" * 90)

    print(f"Saved: {csv_path}")

    print(f"Saved: {txt_path}")

    return 0





if __name__ == '__main__':

    sys.exit(main())

