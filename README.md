# Training ResNet-50 from Scratch on Full ImageNet

This guide explains how to train ResNet-50 from scratch on the full ImageNet (ILSVRC-2012) dataset.

## ImageNet Training Configuration

### 1. **Dataset**
- ImageNet ILSVRC-2012 (1000 classes)
- ~1.28M training images, 50K validation images

### 2. **Image Size**
- Train with 224×224 pixels
- Validation: resize shorter side to 256, then center crop to 224

### 3. **Model Architecture**
- Standard ResNet-50:
  - 7×7 convolution with stride=2
  - 3×3 max pooling with stride=2
  - 4 residual stages (conv2_x through conv5_x)
  - Global average pooling
  - Fully connected layer for 1000 classes
- Standard stem is preferred for 224×224 images

### 4. **Number of Classes**
- 1000

### 5. **Training Hyperparameters**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Epochs | 100 | Standard ImageNet schedule |
| Base LR | 0.1 | Scales with batch size |
| Weight Decay | 1e-4 | Standard ImageNet value |
| MixUp Prob | 0.5 | Less aggressive augmentation |
| EMA Decay | 0.9998 | Suited for longer training |
| Batch Size | 256 | Adjust based on GPU |

### 6. **Data Augmentation**
Augmentation pipeline:
- RandomResizedCrop(224, scale=(0.08, 1.0))
- RandomHorizontalFlip
- TrivialAugmentWide
- RandomErasing(p=0.1)
- Standard ImageNet normalization

### 7. **Training Features**
✓ Mixed precision training (AMP)  
✓ Label smoothing (0.1)  
✓ MixUp & CutMix augmentation  
✓ Exponential Moving Average (EMA)  
✓ Warmup (5 epochs) + Cosine LR scheduling  
✓ Gradient clipping (5.0)  
✓ Nesterov momentum  

## Setup Instructions

### 1. Dataset Preparation

Download and prepare ImageNet ILSVRC-2012:

```bash
# Expected directory structure:
IMAGENET_DIR/
  train/
    n01440764/  (tench)
      n01440764_18.JPEG
      n01440764_36.JPEG
      ...
    n01443537/  (goldfish)
      ...
    ... (1000 class folders)
  val/
    n01440764/
      ILSVRC2012_val_00000293.JPEG
      ...
    n01443537/
      ...
    ... (1000 class folders)
```

**Note**: The validation set should be organized into class folders. If you have a flat `val/` directory, you'll need to reorganize it using the validation labels.

### 2. Set Environment Variable

```bash
# Linux/Mac
export IMAGENET_DIR=/path/to/your/imagenet

# Windows PowerShell
$env:IMAGENET_DIR="C:\path\to\your\imagenet"

# Windows CMD
set IMAGENET_DIR=C:\path\to\your\imagenet
```

Alternatively, edit line 68 in `train_resnet50_imagenet.py`:
```python
data_dir: str = "/path/to/your/imagenet"  # UPDATE THIS!
```

### 3. Install Dependencies

```bash
pip install torch torchvision tqdm numpy matplotlib
```

### 4. Run Training

```bash
python train_resnet50_imagenet.py
```

### Adjusting Batch Size

If you get out-of-memory errors:

```python
# In Config class, reduce batch_size:
batch_size: int = 128  # or 64, 32 depending on GPU
```

The learning rate will be automatically scaled accordingly.

## Training Checkpoints

The script saves:
- `best_model.pth` - Best model based on validation accuracy
- `checkpoint_epoch_X.pth` - Checkpoints every 10 epochs
- `training_history.json` - Training metrics history
- `config.json` - Configuration used

Location: `./checkpoints_imagenet/`

## Loading Trained Model

```python
import torch
from torchvision.models import resnet50

# Load checkpoint
checkpoint = torch.load('checkpoints_imagenet/best_model.pth')

# Create model
model = resnet50(weights=None, num_classes=1000)
model.load_state_dict(checkpoint['model'])  # EMA weights
model.eval()

print(f"Best Acc@1: {checkpoint['best_acc1']:.2f}%")
print(f"Best Acc@5: {checkpoint['best_acc5']:.2f}%")
```

## Monitoring Training

The script provides:
- **Progress bars**: Real-time batch progress with tqdm
- **Epoch summaries**: Loss and accuracy after each epoch
- **Checkpointing**: Automatic saving of best models
- **Training history**: JSON file with all metrics

