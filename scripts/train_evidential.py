#!/usr/bin/env python3
"""
Training script for Evidential Deep Learning (EDL) model.
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
from src.models.baseline_softmax import BaselineSoftmaxModel # Use as container for EDL head if not using Hybrid class
from src.models.evidential_layer import EvidentialLayer, EvidentialLoss
from src.evaluation.metrics import MetricsTracker

# Lightweight container for EDL
class EDLModel(nn.Module):
    def __init__(self, backbone, num_classes=2):
        super().__init__()
        self.backbone = backbone
        self.evidential_head = EvidentialLayer(backbone.feature_dim, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.evidential_head(features)

def train():
    parser = argparse.ArgumentParser(description='Train Evidential Model')
    parser.add_argument('--config', type=str, default='configs/evidential_config.yaml', help='Path to config file')
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
        augmentation_config=data_cfg,
        cache_dir='cache/evidential'
    )

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    backbone = get_backbone(config['model']['backbone'], force_3d=True)
    model = EDLModel(backbone, num_classes=config['model']['classifier']['num_classes']).to(device)

    criterion = EvidentialLoss(
        num_classes=config['model']['classifier']['num_classes'],
        kl_weight=config['evidential']['kl_weight'],
        kl_anneal_start=config['evidential']['kl_anneal_start'],
        kl_anneal_end=config['evidential']['kl_anneal_end']
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config['training']['scheduler']['patience'], factor=config['training']['scheduler']['factor'])

    checkpoint_dir = Path(config['checkpoint']['dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    early_stop_counter = 0

    # Mixed Precision
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    use_amp = (scaler is not None)
    if use_amp:
        print("⚡ Mixed Precision (AMP) enabled")

    # Validate every N epochs
    validate_every_n = config['training'].get('validate_every_n', 1)

    print(f"\nStarting EDL training...")
    
    for epoch in range(config['training']['epochs']):
        model.train()
        criterion.set_epoch(epoch)
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        for images, labels, _ in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                alpha = model(images)
                loss, loss_dict = criterion(alpha, labels)
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'unc': loss_dict['mean_epistemic_unc']})
        
        avg_train_loss = train_loss / len(train_loader)

        # Skip validation on intermediate epochs if configured
        if (epoch + 1) % validate_every_n != 0 and (epoch + 1) != config['training']['epochs']:
            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} (validation skipped)")
            continue

        # Validation
        model.eval()
        val_loss = 0.0
        tracker = MetricsTracker(num_classes=config['model']['classifier']['num_classes'])
        
        with torch.no_grad():
            for images, labels, _ in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                images, labels = images.to(device), labels.to(device)
                alpha = model(images)
                loss, _ = criterion(alpha, labels)
                val_loss += loss.item()
                
                strength = alpha.sum(dim=1, keepdim=True)
                probs = alpha / strength
                preds = torch.argmax(probs, dim=1)
                epi_unc = 2.0 / strength.squeeze()
                tracker.update(preds, labels, confidences=1.0-epi_unc, probabilities=probs)

        avg_val_loss = val_loss / len(val_loader)
        val_metrics = tracker.compute_all_metrics()
        
        print(f"Epoch {epoch+1}: Val Loss={avg_val_loss:.4f}, Val Acc={val_metrics['accuracy']:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({'model_state_dict': model.state_dict()}, checkpoint_dir / 'best_model.pt')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= config['training']['early_stopping']['patience']:
                print(f"\n🛑 STOP: Early stopping triggered at epoch {epoch+1}.")
                print(f"   Metric 'val_loss' stopped improving for {config['training']['early_stopping']['patience']} epochs.")
                stop_reason = "Early Stopping"
                break
        
        if epoch + 1 == config['training']['epochs']:
            stop_reason = "Limit of Epochs reached"

    print("\n" + "="*60)
    print(f"✅ EVIDENTIAL TRAINING FINISHED")
    print(f"   Reason: {stop_reason}")
    print(f"   Best Val Loss: {best_val_loss:.4f}")
    print(f"   Final Epoch: {epoch+1}")
    print("="*60)
    print(f"Model saved to: {checkpoint_dir / 'best_model.pt'}")

if __name__ == '__main__':
    train()
