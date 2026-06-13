import torch
from pathlib import Path

models = [
    ('Baseline (SR)', 'checkpoints/baseline/best_model.pth'),
    ('SelectiveNet', 'checkpoints/selective_net/best_model.pt'),
    ('Evidential (EDL)', 'checkpoints/evidential/best_model.pt'),
    ('Hybrid (3D-ResNet-EDL)', 'checkpoints/hybrid/best_model.pt'),
]
for name, p in models:
    path = Path(p)
    if not path.exists():
        print(f'{name}: BRAK')
        continue
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    vm = ckpt.get('val_metrics', {})
    epoch = ckpt.get('epoch', '?')
    sens = vm.get('sensitivity_at_target_spec', vm.get('val_sensitivity_at_target_spec', '?'))
    auc = vm.get('auc', '?')
    print(f'{name}: OK | epoch={epoch} | Sens@80%Spec={sens} | AUC={auc}')
