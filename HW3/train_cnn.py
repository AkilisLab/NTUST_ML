#!/usr/bin/env python3
"""
CNN Training Script
Supports MNIST and Adult datasets with configurable CNN depth and architecture.
"""

import argparse
import json
import os
import time
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from models.cnn import CNN, create_cnn_for_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train CNN on specified dataset')
    
    # Dataset and I/O
    parser.add_argument('--dataset', choices=['mnist', 'adult'], required=True,
                       help='Dataset to train on')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--val-size', type=float, default=0.1,
                       help='Validation split size (for adult dataset)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Limit number of training samples (for quick testing)')
    
    # CNN Architecture
    parser.add_argument('--num-conv-layers', type=int, default=3,
                       help='Number of convolutional layers')
    parser.add_argument('--base-channels', type=int, default=32,
                       help='Base number of channels in first conv layer')
    parser.add_argument('--channel-multiplier', type=float, default=2.0,
                       help='Multiply channels by this factor each layer')
    parser.add_argument('--kernel-size', type=int, default=3,
                       help='Convolution kernel size')
    parser.add_argument('--pool-size', type=int, default=2,
                       help='Max pooling size')
    parser.add_argument('--fc-sizes', type=int, nargs='*', default=None,
                       help='Fully connected layer sizes (e.g., --fc-sizes 512 256)')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout probability')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--optimizer', choices=['adam', 'sgd', 'adamw'], default='adam',
                       help='Optimizer choice')
    parser.add_argument('--scheduler', choices=['none', 'step', 'cosine'], default='step',
                       help='Learning rate scheduler')
    
    # Class balancing (for Adult dataset)
    parser.add_argument('--class-weight', action='store_true',
                       help='Use class weights for imbalanced datasets')
    parser.add_argument('--weighted-sampler', action='store_true',
                       help='Use weighted random sampler')
    
    # Device and reproducibility
    parser.add_argument('--device', default='auto',
                       help='Device to use (cpu, cuda, mps, or auto)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Output
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save-every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--plot', action='store_true',
                       help='Generate training plots')
    
    # Preprocessing options (for Adult dataset)
    parser.add_argument('--one-hot-only', action='store_true',
                       help='Use only one-hot encoding for Adult dataset')
    parser.add_argument('--feature-selection', choices=['pearson', 'none'], default='none',
                       help='Feature selection method')
    parser.add_argument('--feature-threshold', type=float, default=0.02,
                       help='Feature selection threshold')
    
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    """Determine the best available device"""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        return torch.device(device_arg)


def load_dataset(args):
    """Load dataset using existing data loaders"""
    if args.dataset == 'mnist':
        from data.mnist_loader import get_mnist_loaders
        train_loader, val_loader, test_loader = get_mnist_loaders(
            batch_size=args.batch_size,
            max_samples=args.max_samples
        )
        num_classes = 10
        input_shape = (28, 28)
        
    elif args.dataset == 'adult':
        from data.adult_loader import get_adult_loaders
        
        # Configure preprocessing options
        loader_kwargs = {
            'batch_size': args.batch_size,
            'val_size': args.val_size,
            'max_samples': args.max_samples,
            'one_hot_only': args.one_hot_only,
        }
        
        if args.feature_selection != 'none':
            loader_kwargs['feature_selection'] = args.feature_selection
            loader_kwargs['feature_threshold'] = args.feature_threshold
        
        train_loader, val_loader, test_loader = get_adult_loaders(**loader_kwargs)
        num_classes = 2
        
        # Get actual input dimension from data
        sample_batch, _ = next(iter(train_loader))
        input_shape = (sample_batch.shape[1],)
        
    return train_loader, val_loader, test_loader, num_classes, input_shape


def create_model(args, input_shape, num_classes):
    """Create CNN model with specified architecture"""
    model = CNN(
        input_shape=input_shape,
        num_classes=num_classes,
        num_conv_layers=args.num_conv_layers,
        base_channels=args.base_channels,
        channel_multiplier=args.channel_multiplier,
        kernel_size=args.kernel_size,
        pool_size=args.pool_size,
        fc_sizes=args.fc_sizes,
        dropout=args.dropout
    )
    
    return model


def setup_training(model, args, train_loader, device):
    """Setup optimizer, loss function, and scheduler"""
    # Optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:  # sgd
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    
    # Loss function
    if model.output_type == 'binary':
        if args.class_weight and args.dataset == 'adult':
            # Calculate class weights for Adult dataset
            all_targets = []
            for _, targets in train_loader:
                all_targets.extend(targets.numpy())
            
            pos_count = sum(all_targets)
            neg_count = len(all_targets) - pos_count
            pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()
    else:
        if args.class_weight:
            # Calculate class weights
            all_targets = []
            for _, targets in train_loader:
                all_targets.extend(targets.numpy())
            
            unique_classes, counts = np.unique(all_targets, return_counts=True)
            total_samples = len(all_targets)
            class_weights = torch.tensor([total_samples / (len(unique_classes) * count) for count in counts], 
                                       dtype=torch.float32).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
    
    # Scheduler
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs//3, gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None
    
    return optimizer, criterion, scheduler


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        if model.output_type == 'binary':
            target = target.float().unsqueeze(1)
            loss = criterion(output, target)
            pred = (torch.sigmoid(output) > 0.5).float()
            correct += (pred == target).sum().item()
        else:
            loss = criterion(output, target)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        total += target.size(0)
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            if model.output_type == 'binary':
                target = target.float().unsqueeze(1)
                loss = criterion(output, target)
                pred = (torch.sigmoid(output) > 0.5).float()
                correct += (pred == target).sum().item()
            else:
                loss = criterion(output, target)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
            
            running_loss += loss.item()
            total += target.size(0)
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, val_acc, args, filename):
    """Save model checkpoint"""
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'meta': {
            'dataset': args.dataset,
            'input_shape': model.input_shape,
            'num_classes': model.num_classes,
            'num_conv_layers': args.num_conv_layers,
            'base_channels': args.base_channels,
            'channel_multiplier': args.channel_multiplier,
            'kernel_size': args.kernel_size,
            'pool_size': args.pool_size,
            'fc_sizes': args.fc_sizes,
            'dropout': args.dropout,
            'output_type': model.output_type,
        }
    }
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    torch.save(checkpoint, os.path.join(args.checkpoint_dir, filename))


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    train_loader, val_loader, test_loader, num_classes, input_shape = load_dataset(args)
    print(f"Input shape: {input_shape}, Number of classes: {num_classes}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    model = create_model(args, input_shape, num_classes)
    model.to(device)
    print(f"Created CNN with {model.get_num_parameters():,} parameters")
    print(f"Architecture: {args.num_conv_layers} conv layers, base channels: {args.base_channels}")
    
    # Setup training
    optimizer, criterion, scheduler = setup_training(model, args, train_loader, device)
    print(f"Optimizer: {args.optimizer}, Scheduler: {args.scheduler}")
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print progress
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:3d}/{args.epochs} ({epoch_time:.1f}s) - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, val_acc, args, 'best_cnn.pth')
        
        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, val_acc, args, f'cnn_epoch_{epoch}.pth')
    
    # Save final model
    save_checkpoint(model, optimizer, args.epochs, train_losses[-1], val_losses[-1], val_accs[-1], args, 'last_cnn.pth')
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'total_time': total_time,
        'args': vars(args)
    }
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(os.path.join(args.checkpoint_dir, 'cnn_training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Generate plots
    if args.plot:
        plot_path = os.path.join(args.checkpoint_dir, 'cnn_training_curves.png')
        plot_training_history(train_losses, val_losses, train_accs, val_accs, plot_path)
        print(f"Training curves saved to {plot_path}")
    
    print(f"Checkpoints saved to {args.checkpoint_dir}/")


if __name__ == '__main__':
    main()
