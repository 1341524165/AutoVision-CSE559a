"""
Quick Baseline Script — Adapted from GuillaumeErhard/autoresearch-cifar10
https://github.com/GuillaumeErhard/autoresearch-cifar10

Run this in Google Colab to:
  1. Generate sample data images for CIFAR-10 and Oxford-IIIT Pet
  2. Train the ResNet-20 baseline (faithful to He et al. 2016) under a 5-min budget
  3. Print the accuracy for your progress report
"""

import gc
import time
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# ==================== PART 1: Generate Data Sample Images ====================
print("=" * 60)
print("PART 1: Generating data sample images...")
print("=" * 60)

# CIFAR-10 samples
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
classes = dataset.classes
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('CIFAR-10 Sample Images (32x32)', fontsize=14, fontweight='bold')
for i in range(10):
    ax = axes[i // 5, i % 5]
    for idx in range(len(dataset)):
        img, label = dataset[idx]
        if label == i:
            ax.imshow(img)
            ax.set_title(classes[i], fontsize=10)
            ax.axis('off')
            break
plt.tight_layout()
plt.savefig('cifar10_samples.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved cifar10_samples.png")

# Oxford-IIIT Pet samples
from torchvision.datasets import OxfordIIITPet
pet_data = OxfordIIITPet(root='./data', split='trainval', target_types='segmentation', download=True)
fig, axes = plt.subplots(2, 3, figsize=(10, 7))
fig.suptitle('Oxford-IIIT Pet Dataset Samples', fontsize=14, fontweight='bold')
for row in range(2):
    img, mask = pet_data[row * 100]
    img_np = np.array(img)
    mask_np = np.array(mask)
    axes[row, 0].imshow(img_np)
    axes[row, 0].set_title('Input Image')
    axes[row, 0].axis('off')
    axes[row, 1].imshow(mask_np, cmap='gray')
    axes[row, 1].set_title('Trimap Mask')
    axes[row, 1].axis('off')
    axes[row, 2].imshow(img_np)
    axes[row, 2].imshow(mask_np, alpha=0.4, cmap='jet')
    axes[row, 2].set_title('Overlay')
    axes[row, 2].axis('off')
plt.tight_layout()
plt.savefig('pet_samples.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved pet_samples.png")

# ==================== PART 2: ResNet-20 Baseline (from Erhard repo) ====================
# Source: https://github.com/GuillaumeErhard/autoresearch-cifar10/blob/main/train.py
# Faithful ResNet-20 per He et al. (2016) Section 4.2:
#   - 6n+2 = 20 layers (n=3), filter widths {16, 32, 64}
#   - Identity shortcuts with zero-padding (Option A)
#   - SGD: LR 0.1, momentum 0.9, weight decay 1e-4
#   - LR divided by 10 at step 32k and 48k, total 64k steps
#   - Batch size 128, Kaiming init, no dropout
#   - Augmentation: 4px padding + random 32x32 crop + horizontal flip
#   - Normalization: channel-wise mean subtraction, std=1
# ====================================================================================

print("\n" + "=" * 60)
print("PART 2: Training ResNet-20 on CIFAR-10 (5-min limit)...")
print("=" * 60)

# ---------------------------------------------------------------------------
# Config (matching Erhard repo)
# ---------------------------------------------------------------------------
NUM_BLOCKS = 3       # ResNet-20 = 6*3+2
NUM_CLASSES = 10
BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
MAX_STEPS = 64000
TIME_BUDGET_S = 300  # 5 minutes

# ---------------------------------------------------------------------------
# Model: ResNet-20 with zero-padding shortcut (Option A from He et al.)
# ---------------------------------------------------------------------------
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride
        self.need_pad = stride != 1 or in_channels != out_channels
        self.pad_channels = out_channels - in_channels if self.need_pad else 0

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        shortcut = x
        if self.need_pad:
            shortcut = shortcut[:, :, ::self.stride, ::self.stride]
            shortcut = F.pad(shortcut, (0, 0, 0, 0, 0, self.pad_channels))
        out += shortcut
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 16, num_blocks, stride=1)
        self.layer2 = self._make_layer(16, 32, num_blocks, stride=2)
        self.layer3 = self._make_layer(32, 64, num_blocks, stride=2)
        self.fc = nn.Linear(64, num_classes)
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            init.kaiming_normal_(m.weight)

    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        ch = in_ch
        for s in strides:
            layers.append(BasicBlock(ch, out_ch, s))
            ch = out_ch
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)

# ---------------------------------------------------------------------------
# Data (per original paper: channel-wise mean, std=1)
# ---------------------------------------------------------------------------
mean, std = (0.4914, 0.4822, 0.4465), (1, 1, 1)

train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_tf)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_tf)
testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = ResNet(NUM_BLOCKS, NUM_CLASSES).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"ResNet-{6 * NUM_BLOCKS + 2} | params: {num_params:,}")

optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[32000, 48000], gamma=0.1)

print(f"Time budget: {TIME_BUDGET_S}s")

t_start = time.time()
total_training_time = 0.0
best_acc = 0.0
epoch = 0
step = 0

while total_training_time < TIME_BUDGET_S and step < MAX_STEPS:
    epoch += 1
    model.train()

    for inputs, targets in trainloader:
        t0 = time.time()
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if device.type == 'cuda':
            torch.cuda.synchronize()
        dt = time.time() - t0
        total_training_time += dt
        step += 1

        if step % 100 == 0:
            pct_done = 100 * total_training_time / TIME_BUDGET_S
            remaining = max(0, TIME_BUDGET_S - total_training_time)
            print(f"\r  step {step:05d} ep {epoch} ({pct_done:.1f}%) | loss: {loss.item():.4f} | rem: {remaining:.0f}s", end="", flush=True)

        if total_training_time >= TIME_BUDGET_S or step >= MAX_STEPS:
            break

    # Evaluate at end of each epoch
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            correct += outputs.argmax(1).eq(targets).sum().item()
            total += targets.size(0)
    test_acc = 100.0 * correct / total
    if test_acc > best_acc:
        best_acc = test_acc
    print(f"\n  eval ep {epoch:3d} | test_acc: {test_acc:.2f}% | best: {best_acc:.2f}%")

    if epoch == 1:
        gc.collect()

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
t_end = time.time()
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"best_test_acc:    {best_acc:.2f}%")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"num_epochs:       {epoch}")
print(f"num_steps:        {step}")
print(f"num_params:       {num_params:,}")
print(f"\n>>> Use best_test_acc = {best_acc:.2f}% in your progress report! <<<")
