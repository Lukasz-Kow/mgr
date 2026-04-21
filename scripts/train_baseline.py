#!/usr/bin/env python3
"""
Train baseline model (Softmax Response with threshold rejection).

Usage:
    python scripts/train_baseline.py
    python scripts/train_baseline.py --config configs/baseline_config.yaml
"""

import sys
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import MCIDataModule
from src.models.backbone import get_backbone
from src.models import BaselineSoftmaxModel
from src.evaluation.metrics import MetricsTracker


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer, log_interval=10):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [TRAIN]')
    for batch_idx, (images, labels, _) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Forward
        logits = model(images)
        loss = criterion(logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Logging
        running_loss += loss.item()
        
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            if writer:
                global_step = epoch * len(dataloader) + batch_idx
                writer.add_scalar('Train/Loss', avg_loss, global_step)
    
    return running_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    tracker = MetricsTracker(num_classes=2)
    
    for images, labels, _ in tqdm(dataloader, desc='Validating'):
        images, labels = images.to(device), labels.to(device)
        
        # Forward
        logits = model(images)
        loss = criterion(logits, labels)
        
        # Predictions
        predictions, confidences, probabilities = model.predict_with_confidence(images)
        
        # Track metrics
        tracker.update(predictions, labels, confidences, probabilities)
        running_loss += loss.item()
    
    # Compute metrics
    metrics = tracker.compute_all_metrics()
    metrics['loss'] = running_loss / len(dataloader)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train Baseline Model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/baseline_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--data-config',
        type=str,
        default='configs/data_config.yaml',
        help='Path to data config file'
    )
    args = parser.parse_args()
    
    # Load configs
    config = load_config(args.config)
    data_config = load_config(args.data_config)
    
    # Set device and seed
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config['seed'])
    
    print("="*60)
    print("BASELINE MODEL TRAINING")
    print("="*60)
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print("="*60)
    
    # Create data module
    print("\n📂 Loading data...")
    data_module = MCIDataModule(
        metadata_csv=config['data']['metadata_csv'],
        preprocessor_config=data_config['preprocessing'],
        batch_size=config['training']['batch_size'],
        num_workers=data_config['dataloader']['num_workers'],
        augmentation_config=data_config.get('augmentation')
    )
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    # Create model
    print("\n🏗️  Building model...")
    backbone = get_backbone(config['model']['backbone'], force_3d=config['model']['backbone'].get('use_3d', True))
    
    model = BaselineSoftmaxModel(
        backbone=backbone,
        num_classes=config['model']['classifier']['num_classes'],
        dropout=config['model']['classifier']['dropout']
    ).to(device)
    
    arch_name = config['model']['backbone'].get('arch_3d', 'resnet3d_18') if config['model']['backbone'].get('use_3d', True) else config['model']['backbone'].get('arch_2d', 'resnet18')
    print(f"Model architecture: {arch_name} ({'3D' if config['model']['backbone'].get('use_3d', True) else '2D'})")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function
    if config['training']['use_class_weights']:
        class_weights = data_module.get_class_weights().to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using class weights: {class_weights.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if config['training']['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    elif config['training']['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['training']['optimizer']}")
    
    # Scheduler
    if config['training']['scheduler']['type'] == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=config['training']['scheduler']['patience'],
            factor=config['training']['scheduler']['factor'],
            min_lr=config['training']['scheduler']['min_lr']
        )
    else:
        scheduler = None
    
    # TensorBoard
    log_dir = config['logging']['tensorboard_dir']
    writer = SummaryWriter(log_dir)
    print(f"\n📊 TensorBoard logs: {log_dir}")
    
    # Checkpoint directory
    checkpoint_dir = Path(config['checkpoint']['dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"💾 Checkpoints: {checkpoint_dir}")
    
    # Training loop
    print("\n🚀 Starting training...")
    print("="*60)
    
    best_val_metric = float('inf')
    patience_counter = 0
    
    for epoch in range(1, config['training']['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['training']['epochs']}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, writer, config['logging']['log_interval']
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Log
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}")
        print(f"Val AUC: {val_metrics.get('auc', 0):.4f}")
        
        # TensorBoard
        writer.add_scalar('Train/Epoch_Loss', train_loss, epoch)
        writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
        writer.add_scalar('Val/Accuracy', val_metrics['accuracy'], epoch)
        writer.add_scalar('Val/F1', val_metrics['f1'], epoch)
        if 'auc' in val_metrics:
            writer.add_scalar('Val/AUC', val_metrics['auc'], epoch)
        
        # Scheduler step
        if scheduler:
            scheduler.step(val_metrics['loss'])
            writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Checkpointing
        val_metric = val_metrics[config['checkpoint']['monitor'].replace('val_', '')]
        is_best = val_metric < best_val_metric
        
        if is_best:
            best_val_metric = val_metric
            patience_counter = 0
            
            if config['checkpoint']['save_best_only']:
                checkpoint_path = checkpoint_dir / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'config': config
                }, checkpoint_path)
                print(f"✅ Saved best model: {checkpoint_path}")
        else:
            patience_counter += 1
        
        # Early stopping
        if config['training']['early_stopping']['enabled']:
            if patience_counter >= config['training']['early_stopping']['patience']:
                print(f"\n⚠️  Early stopping triggered after {epoch} epochs")
                break
    
    print("\n" + "="*60)
    print("✅ Training complete!")
    print("="*60)
    print(f"Best {config['checkpoint']['monitor']}: {best_val_metric:.4f}")
    print(f"Model saved to: {checkpoint_dir / 'best_model.pth'}")
    
    writer.close()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
