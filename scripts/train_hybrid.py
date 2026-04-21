#!/usr/bin/env python3
"""
Training script for Hybrid 3D-ResNet-EDL model.
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
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import MCIDataModule
from src.models.backbone import get_backbone
from src.models.hybrid_model import HybridEvidentialModel
from src.models.evidential_layer import EvidentialLoss
from src.evaluation.metrics import MetricsTracker

def train():
    parser = argparse.ArgumentParser(description='Train Hybrid 3D-ResNet-EDL Model')
    parser.add_argument('--config', type=str, default='configs/hybrid_config.yaml', help='Path to config file')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data config for preprocessing settings
    with open('configs/data_config.yaml', 'r') as f:
        data_cfg = yaml.safe_load(f)

    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # DataModule
    dm = MCIDataModule(
        metadata_csv=config['data']['metadata_csv'],
        preprocessor_config=data_cfg['preprocessing'],
        batch_size=config['training']['batch_size'],
        num_workers=data_cfg['dataloader']['num_workers'],
        augmentation_config=data_cfg
    )

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    # Model
    backbone = get_backbone(config['model']['backbone'], force_3d=True)
    model = HybridEvidentialModel(
        backbone=backbone,
        num_classes=config['model']['classifier']['num_classes'],
        dropout=config['model']['classifier']['dropout']
    ).to(device)

    # Loss
    criterion = EvidentialLoss(
        num_classes=config['model']['classifier']['num_classes'],
        kl_weight=config['evidential']['kl_weight'],
        kl_anneal_start=config['evidential']['kl_anneal_start'],
        kl_anneal_end=config['evidential']['kl_anneal_end']
    )

    # Optimizer
    if config['training']['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])

    # Scheduler
    if config['training']['scheduler']['type'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['scheduler']['t_max'], eta_min=config['training']['scheduler']['min_lr'])
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config['training']['scheduler']['factor'], patience=config['training']['scheduler']['patience'])

    # Checkpoint dir
    checkpoint_dir = Path(config['checkpoint']['dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_loss = float('inf')
    early_stop_counter = 0

    print(f"\nStarting Hybrid model training for {config['training']['epochs']} epochs...")
    
    for epoch in range(config['training']['epochs']):
        # Train
        model.train()
        criterion.set_epoch(epoch)
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} [Train]")
        for images, labels, _ in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            alpha = model(images)
            loss, loss_dict = criterion(alpha, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'kl': loss_dict['kl_coeff']})
        
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        tracker = MetricsTracker(num_classes=config['model']['classifier']['num_classes'])
        
        with torch.no_grad():
            for images, labels, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} [Val]"):
                images, labels = images.to(device), labels.to(device)
                alpha = model(images)
                loss, _ = criterion(alpha, labels)
                val_loss += loss.item()
                
                # For metrics
                strength = alpha.sum(dim=1, keepdim=True)
                probs = alpha / strength
                preds = torch.argmax(probs, dim=1)
                # Uncertainty for rejection sweep
                epi_unc = config['model']['classifier']['num_classes'] / strength.squeeze()
                
                tracker.update(preds, labels, confidences=1-epi_unc, probabilities=probs)

        avg_val_loss = val_loss / len(val_loader)
        val_metrics = tracker.compute_all_metrics()
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_metrics['accuracy']:.4f}")

        # Scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()

        # Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'metrics': val_metrics
            }, checkpoint_dir / 'best_model.pt')
            print(f"✓ Saved best model to {checkpoint_dir / 'best_model.pt'}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if config['training']['early_stopping']['enabled'] and early_stop_counter >= config['training']['early_stopping']['patience']:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    print("\n✅ Hybrid model training finished!")

if __name__ == '__main__':
    train()
