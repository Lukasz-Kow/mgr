#!/usr/bin/env python3
"""
Training script for SelectiveNet model.
"""

import sys
import os
import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import MCIDataModule
from src.models.backbone import get_backbone
from src.models.selective_net import SelectiveNet, SelectiveNetLoss
from src.evaluation.metrics import MetricsTracker

def train():
    parser = argparse.ArgumentParser(description='Train SelectiveNet Model')
    parser.add_argument('--config', type=str, default='configs/selectivenet_config.yaml', help='Path to config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    with open('configs/data_config.yaml', 'r') as f:
        data_cfg = yaml.safe_load(f)

    torch.manual_seed(config['seed'])
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

    dm = MCIDataModule(
        metadata_csv=config['data']['metadata_csv'],
        preprocessor_config=data_cfg['preprocessing'],
        batch_size=config['training']['batch_size'],
        num_workers=data_cfg['dataloader']['num_workers'],
        augmentation_config=data_cfg
    )

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    backbone = get_backbone(config['model']['backbone'], force_3d=True)
    model = SelectiveNet(
        backbone=backbone,
        num_classes=config['model']['classifier']['num_classes'],
        dropout=config['model']['classifier']['dropout'],
        selection_dropout=config['model']['classifier']['selection_dropout']
    ).to(device)

    criterion = SelectiveNetLoss(
        target_coverage=config['selective_net']['target_coverage'],
        aux_weight=config['selective_net']['aux_weight'],
        coverage_penalty=config['selective_net']['coverage_penalty']
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config['training']['scheduler']['patience'], factor=config['training']['scheduler']['factor'])

    checkpoint_dir = Path(config['checkpoint']['dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    early_stop_counter = 0

    print(f"\nStarting SelectiveNet training...")
    
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        for images, labels, _ in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            pred_logits, selection_probs, aux_logits = model(images, return_selection=True, return_auxiliary=True)
            loss, loss_dict = criterion(pred_logits, selection_probs, aux_logits, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'cov': loss_dict['coverage']})
        
        # Validation
        model.eval()
        val_loss = 0.0
        tracker = MetricsTracker(num_classes=config['model']['classifier']['num_classes'])
        
        with torch.no_grad():
            for images, labels, _ in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                images, labels = images.to(device), labels.to(device)
                pred_logits, selection_probs, aux_logits = model(images, return_selection=True, return_auxiliary=True)
                loss, _ = criterion(pred_logits, selection_probs, aux_logits, labels)
                val_loss += loss.item()
                
                probs = torch.softmax(pred_logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                tracker.update(preds, labels, confidences=selection_probs, probabilities=probs)

        avg_val_loss = val_loss / len(val_loader)
        val_metrics = tracker.compute_all_metrics()
        
        print(f"Epoch {epoch+1}: Val Loss={avg_val_loss:.4f}, Val Acc={val_metrics['accuracy']:.4f}, Cov={val_metrics['abstention_rate']:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({'model_state_dict': model.state_dict()}, checkpoint_dir / 'best_model.pt')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= config['training']['early_stopping']['patience']:
                break

    print("\n✅ SelectiveNet training finished!")

if __name__ == '__main__':
    train()
