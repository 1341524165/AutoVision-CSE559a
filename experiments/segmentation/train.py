"""
train.py — U-Net Segmentation Baseline for Oxford-IIIT Pet
This is the ONLY file the agent may modify.

Architecture:
  - Encoder: pretrained ResNet-34 (torchvision)
  - Decoder: symmetric expanding path with transposed convolutions + skip connections
  - 3-class output: foreground (pet), background, boundary

Training:
  - Optimizer: Adam, lr=1e-3
  - Loss: Cross-entropy
  - Metric: mIoU (mean Intersection over Union)
"""

import gc
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

from prepare import DATASET_DIR, NUM_WORKERS, TIME_BUDGET_S, NUM_CLASSES, PetSegDataset, Eval

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 200
evaluator = Eval()


# ---------------------------------------------------------------------------
# Model: U-Net with ResNet-34 Encoder
# ---------------------------------------------------------------------------
class DecoderBlock(nn.Module):
    """Upsampling block: upsample + concat skip + double conv."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch from odd dimensions
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetResNet34(nn.Module):
    """U-Net with a pretrained ResNet-34 encoder."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        # Encoder (pretrained ResNet-34)
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64, /2
        self.pool0 = resnet.maxpool  # /4
        self.encoder1 = resnet.layer1  # 64, /4
        self.encoder2 = resnet.layer2  # 128, /8
        self.encoder3 = resnet.layer3  # 256, /16
        self.encoder4 = resnet.layer4  # 512, /32

        # Decoder
        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 64, 64)

        # Final upsample to original resolution + classification
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1),
        )

    def forward(self, x):
        # Encoder
        e0 = self.encoder0(x)   # (B, 64, H/2, W/2)
        p0 = self.pool0(e0)     # (B, 64, H/4, W/4)
        e1 = self.encoder1(p0)  # (B, 64, H/4, W/4)
        e2 = self.encoder2(e1)  # (B, 128, H/8, W/8)
        e3 = self.encoder3(e2)  # (B, 256, H/16, W/16)
        e4 = self.encoder4(e3)  # (B, 512, H/32, W/32)

        # Decoder
        d4 = self.decoder4(e4, e3)  # (B, 256, H/16, W/16)
        d3 = self.decoder3(d4, e2)  # (B, 128, H/8, W/8)
        d2 = self.decoder2(d3, e1)  # (B, 64, H/4, W/4)
        d1 = self.decoder1(d2, e0)  # (B, 64, H/2, W/2)

        # Final
        out = self.final_up(d1)  # (B, 32, H, W)
        out = self.final_conv(out)

        # Ensure output matches input size
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)

        return out


def main():
    # ---------------------------------------------------------------------------
    # Setup
    # ---------------------------------------------------------------------------
    t_start = time.time()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_set = PetSegDataset(split="trainval", is_train=True)
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    model = UNetResNet34(num_classes=NUM_CLASSES).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"UNet-ResNet34 | params: {num_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    print(f"Time budget: {TIME_BUDGET_S}s")
    print(f"Batches per epoch: {len(train_loader)}")

    # ---------------------------------------------------------------------------
    # Training loop (time-budgeted)
    # ---------------------------------------------------------------------------
    t_start_training = time.time()
    smooth_train_loss = 0.0
    total_training_time = 0.0
    epoch = 0
    step = 0
    best_miou = 0.0

    while total_training_time < TIME_BUDGET_S and epoch < MAX_EPOCHS:
        epoch += 1
        model.train()

        for inputs, targets in train_loader:
            t0 = time.time()
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            total_training_time += dt
            step += 1

            train_loss_f = loss.item()
            ema_beta = 0.95
            smooth_train_loss = (
                ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
            )
            debiased = smooth_train_loss / (1 - ema_beta**step)

            lr_current = optimizer.param_groups[0]["lr"]
            pct_done = 100 * total_training_time / TIME_BUDGET_S
            remaining = max(0, TIME_BUDGET_S - total_training_time)

            if step % 20 == 0:
                print(
                    f"\rstep {step:05d} ep {epoch} ({pct_done:.1f}%) | loss: {debiased:.4f} | lr: {lr_current:.6f} | rem: {remaining:.0f}s    ",
                    end="",
                    flush=True,
                )

            if total_training_time >= TIME_BUDGET_S:
                break

        test_loss, test_miou = evaluator.evaluate(model, device)

        if test_miou > best_miou:
            best_miou = test_miou

        print(
            f"\n  eval ep {epoch:3d} | test_loss: {test_loss:.4f} | test_mIoU: {test_miou:.2f}% | best: {best_miou:.2f}%"
        )

        if epoch == 1:
            gc.collect()

    # ---------------------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------------------
    t_end = time.time()
    startup_time = t_start_training - t_start
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0
    )

    print("---")
    print(f"best_test_miou:   {best_miou:.2f}%")
    print(f"final_test_miou:  {test_miou:.2f}%")
    print(f"final_test_loss:  {test_loss:.4f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"startup_seconds:  {startup_time:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_epochs:       {epoch}")
    print(f"num_steps:        {step}")
    print(f"num_params:       {num_params:,}")


if __name__ == "__main__":
    main()
