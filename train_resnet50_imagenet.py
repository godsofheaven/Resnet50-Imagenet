"""
Train ResNet-50 from scratch on full ImageNet (ILSVRC-2012)

This script trains a ResNet-50 model from scratch on the full ImageNet dataset
with modern training techniques including:
- Mixed precision training
- Label smoothing
- MixUp and CutMix augmentation
- Exponential Moving Average (EMA)
- Warmup + Cosine LR scheduling
- Gradient clipping

Expected ImageNet directory structure:
    IMAGENET_DIR/
        train/
            n01440764/  (class folder)
                n01440764_*.JPEG
            n01443537/
            ...
        val/
            n01440764/
            ...

Usage:
    # Set ImageNet path
    export IMAGENET_DIR=/path/to/imagenet
    
    # Run training
    python train_resnet50_imagenet.py
    
    # Or specify path in code (line 68)
"""

import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50
import math
from typing import Tuple
from copy import deepcopy
from time import time
from pathlib import Path
import json
from dataclasses import dataclass
from tqdm.auto import tqdm

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    # Dataset
    num_classes: int = 1000  # Full ImageNet has 1000 classes
    img_size: int = 224      # Standard ImageNet image size
    
    # Training
    epochs: int = 100        # Standard ImageNet: 90-100 epochs
    batch_size: int = 256    # Adjust based on your GPU memory
    num_workers: int = 16    # Data loading workers
    
    # Optimizer
    base_lr: float = 0.1     # Base LR for batch_size=256
    momentum: float = 0.9
    weight_decay: float = 1e-4
    nesterov: bool = True
    
    # Scheduler
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Regularization
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    mixup_prob: float = 0.5  # Apply mixup/cutmix 50% of time
    grad_clip_norm: float = 5.0
    
    # EMA
    ema_decay: float = 0.9998
    
    # Paths
    data_dir: str = "/path/to/imagenet"  # UPDATE THIS!
    checkpoint_dir: str = "./checkpoints_imagenet"
    
    # Reproducibility
    seed: int = 42


# ============================================================================
# Utility Classes and Functions
# ============================================================================

class ModelEMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.9998):
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.num_updates += 1
        # Adjust decay based on number of updates
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k], alpha=1.0 - d)


class SmoothCrossEntropyLoss(nn.Module):
    """Cross entropy loss with label smoothing."""
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        num_classes = pred.size(-1)
        log_probs = torch.log_softmax(pred, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing."""
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, 
                 base_lr: float, min_lr: float):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
    
    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * self.current_epoch / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Computes the accuracy over the k top predictions."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def rand_bbox(W: int, H: int, lam: float) -> Tuple[int, int, int, int]:
    """Generate random bounding box for CutMix."""
    cut_rat = math.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def mixup_cutmix_data(x, y, alpha_mixup: float, alpha_cutmix: float, prob: float):
    """Apply MixUp or CutMix augmentation."""
    r = np.random.rand()
    if r < prob * 0.5 and alpha_mixup > 0:
        # MixUp
        lam = np.random.beta(alpha_mixup, alpha_mixup)
        index = torch.randperm(x.size(0), device=x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam, 'mixup'
    elif r < prob and alpha_cutmix > 0:
        # CutMix
        lam = np.random.beta(alpha_cutmix, alpha_cutmix)
        index = torch.randperm(x.size(0), device=x.device)
        y_a, y_b = y, y[index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(3), x.size(2), lam)
        x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
        return x, y_a, y_b, lam, 'cutmix'
    else:
        # No augmentation
        return x, y, y, 1.0, 'none'


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, 
                    scaler, ema, cfg, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct1 = 0.0
    correct5 = 0.0
    total = 0
    start = time()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Apply MixUp/CutMix
        images_mixed, y_a, y_b, lam, aug_type = mixup_cutmix_data(
            images, targets, cfg.mixup_alpha, cfg.cutmix_alpha, cfg.mixup_prob
        )

        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with mixed precision
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            outputs = model(images_mixed)
            if aug_type == 'none':
                loss = criterion(outputs, targets)
            else:
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)

        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping
        if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        
        scaler.step(optimizer)
        scaler.update()

        # Update EMA
        ema.update(model)

        # Metrics
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

        with torch.no_grad():
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            correct1 += acc1.item() * batch_size / 100.0
            correct5 += acc5.item() * batch_size / 100.0

        # Update progress bar
        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'loss': f"{running_loss / total:.4f}",
                'acc@1': f"{100.0 * correct1 / total:.2f}%",
                'acc@5': f"{100.0 * correct5 / total:.2f}%",
                'lr': f"{scheduler.get_last_lr()[0]:.6f}"
            })

    # Step scheduler
    current_lr = scheduler.step()

    epoch_loss = running_loss / total
    epoch_acc1 = 100.0 * correct1 / total
    epoch_acc5 = 100.0 * correct5 / total
    dt = time() - start

    return epoch_loss, epoch_acc1, epoch_acc5, current_lr, dt


def evaluate(model, val_loader, criterion, device):
    """Evaluate on validation set."""
    model.eval()
    total = 0
    correct1 = 0.0
    correct5 = 0.0
    running_loss = 0.0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size
            correct1 += acc1.item() * batch_size / 100.0
            correct5 += acc5.item() * batch_size / 100.0
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{running_loss / total:.4f}",
                'acc@1': f"{100.0 * correct1 / total:.2f}%",
                'acc@5': f"{100.0 * correct5 / total:.2f}%"
            })

    val_loss = running_loss / total
    val_acc1 = 100.0 * correct1 / total
    val_acc5 = 100.0 * correct5 / total
    
    return val_loss, val_acc1, val_acc5


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    # Load configuration
    cfg = Config()
    
    # Override data_dir from environment variable if set
    cfg.data_dir = os.environ.get("IMAGENET_DIR", cfg.data_dir)
    
    # Set random seeds for reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Validate dataset paths
    train_dir = os.path.join(cfg.data_dir, "train")
    val_dir = os.path.join(cfg.data_dir, "val")
    
    if not (os.path.isdir(train_dir) and os.path.isdir(val_dir)):
        raise FileNotFoundError(
            f"ImageNet folders not found!\n"
            f"Expected structure:\n"
            f"  {cfg.data_dir}/train/\n"
            f"  {cfg.data_dir}/val/\n"
            f"Please set IMAGENET_DIR environment variable or update cfg.data_dir in code."
        )
    
    # Data transforms
    print("\nSetting up data transforms...")
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(cfg.img_size, scale=(0.08, 1.0), 
                                     interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load datasets
    print("Loading ImageNet datasets...")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Build model
    print("\nBuilding ResNet-50 model...")
    model = resnet50(weights=None, num_classes=cfg.num_classes)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (FP32)")
    
    # Setup training components
    print("\nSetting up training components...")
    
    # Learning rate scaling
    scaled_lr = cfg.base_lr * (cfg.batch_size / 256)
    print(f"Base LR: {cfg.base_lr}, Scaled LR: {scaled_lr:.5f}")
    
    criterion = SmoothCrossEntropyLoss(smoothing=cfg.label_smoothing)
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=scaled_lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=cfg.nesterov
    )
    
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=cfg.warmup_epochs,
        total_epochs=cfg.epochs,
        base_lr=scaled_lr,
        min_lr=cfg.min_lr
    )
    
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    ema = ModelEMA(model, decay=cfg.ema_decay)
    
    # Create checkpoint directory
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(ckpt_dir / 'config.json', 'w') as f:
        json.dump(cfg.__dict__, f, indent=2)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc1': [],
        'train_acc5': [],
        'val_loss': [],
        'val_acc1': [],
        'val_acc5': [],
        'learning_rates': []
    }
    
    best_acc1 = 0.0
    best_acc5 = 0.0
    
    # Training loop
    print("\n" + "="*80)
    print(f"Starting training for {cfg.epochs} epochs")
    print(f"Checkpoints will be saved to: {ckpt_dir}")
    print("="*80 + "\n")
    
    for epoch in range(1, cfg.epochs + 1):
        # Training
        train_loss, train_acc1, train_acc5, current_lr, train_time = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, 
            scaler, ema, cfg, device, epoch
        )
        
        print(f"\nEpoch {epoch} Training Summary:")
        print(f"  Loss: {train_loss:.4f}")
        print(f"  Acc@1: {train_acc1:.2f}%, Acc@5: {train_acc5:.2f}%")
        print(f"  Time: {train_time:.1f}s, LR: {current_lr:.6f}")
        
        # Validation
        val_loss, val_acc1, val_acc5 = evaluate(ema.ema, val_loader, criterion, device)
        
        print(f"\nEpoch {epoch} Validation Summary:")
        print(f"  Loss: {val_loss:.4f}")
        print(f"  Acc@1: {val_acc1:.2f}%")
        print(f"  Acc@5: {val_acc5:.2f}%")
        print("="*80 + "\n")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc1'].append(train_acc1)
        history['train_acc5'].append(train_acc5)
        history['val_loss'].append(val_loss)
        history['val_acc1'].append(val_acc1)
        history['val_acc5'].append(val_acc5)
        history['learning_rates'].append(current_lr)
        
        # Save best model
        if val_acc1 > best_acc1:
            best_acc1 = val_acc1
            best_acc5 = val_acc5
            
            checkpoint = {
                'epoch': epoch,
                'model': ema.ema.state_dict(),
                'model_raw': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'best_acc1': best_acc1,
                'best_acc5': best_acc5,
                'cfg': cfg.__dict__,
            }
            
            torch.save(checkpoint, ckpt_dir / 'best_model.pth')
            print(f"✓ Saved new best checkpoint: Acc@1={best_acc1:.2f}%, Acc@5={best_acc5:.2f}%\n")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model': ema.ema.state_dict(),
                'model_raw': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'best_acc1': best_acc1,
                'best_acc5': best_acc5,
                'cfg': cfg.__dict__,
            }
            torch.save(checkpoint, ckpt_dir / f'checkpoint_epoch_{epoch}.pth')
            print(f"✓ Saved checkpoint at epoch {epoch}\n")
        
        # Save training history
        with open(ckpt_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    # Training complete
    print("\n" + "="*80)
    print("Training Complete!")
    print(f"Best Validation Acc@1: {best_acc1:.2f}%")
    print(f"Best Validation Acc@5: {best_acc5:.2f}%")
    print(f"Checkpoints saved to: {ckpt_dir}")
    print("="*80)
    
    return history, best_acc1, best_acc5


if __name__ == "__main__":
    main()

