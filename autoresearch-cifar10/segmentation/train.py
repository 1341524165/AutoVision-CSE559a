from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import ResNet34_Weights, resnet34

from prepare import NUM_CLASSES, TIME_BUDGET_S, Eval, make_train_loader

BATCH_SIZE = 12
LR = 2.5e-4
WEIGHT_DECAY = 1e-4
DECODER_CHANNELS = 64
EVAL_BATCH_SIZE = 16
USE_PRETRAINED = True
FREEZE_ENCODER_EPOCHS = 0


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ResNetUNet(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet34_Weights.DEFAULT if USE_PRETRAINED else None
        encoder = resnet34(weights=weights)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.enc1 = encoder.layer1
        self.enc2 = encoder.layer2
        self.enc3 = encoder.layer3
        self.enc4 = encoder.layer4
        ch = DECODER_CHANNELS
        self.dec4 = DecoderBlock(512, 256, ch * 2)
        self.dec3 = DecoderBlock(ch * 2, 128, ch)
        self.dec2 = DecoderBlock(ch, 64, ch // 2)
        self.dec1 = DecoderBlock(ch // 2, 64, ch // 2)
        self.head = nn.Conv2d(ch // 2, NUM_CLASSES, kernel_size=1)

    def set_encoder_trainable(self, trainable: bool):
        for module in [self.stem, self.enc1, self.enc2, self.enc3, self.enc4]:
            for param in module.parameters():
                param.requires_grad = trainable

    def forward(self, x):
        input_size = x.shape[-2:]
        s0 = self.stem(x)
        x = self.pool(s0)
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        x = self.enc4(s3)
        x = self.dec4(x, s3)
        x = self.dec3(x, s2)
        x = self.dec2(x, s1)
        x = self.dec1(x, s0)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return self.head(x)


def main():
    t_start = time.time()
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_loader = make_train_loader(BATCH_SIZE)
    evaluator = Eval(EVAL_BATCH_SIZE)
    model = ResNetUNet().to(device)
    model.set_encoder_trainable(FREEZE_ENCODER_EPOCHS == 0)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"ResNet34-UNet | params: {num_params:,}")
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=LR / 20)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    t_start_training = time.time()
    total_training_time = 0.0
    epoch = 0
    step = 0
    best_miou = 0.0
    best_dice = 0.0
    final_miou = 0.0
    final_dice = 0.0

    while total_training_time < TIME_BUDGET_S:
        epoch += 1
        if epoch == FREEZE_ENCODER_EPOCHS + 1:
            model.set_encoder_trainable(True)
            optimizer = optim.AdamW(model.parameters(), lr=LR / 2, weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=LR / 30)
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        for images, masks in train_loader:
            t0 = time.time()
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = F.cross_entropy(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_training_time += time.time() - t0
            step += 1
            epoch_steps += 1
            epoch_loss += loss.item()
            if step % 20 == 0:
                print(f"step {step:05d} ep {epoch} loss {epoch_loss / epoch_steps:.4f} rem {max(0, TIME_BUDGET_S - total_training_time):.0f}s", flush=True)
            if total_training_time >= TIME_BUDGET_S:
                break
        scheduler.step()
        final_miou, final_dice = evaluator.evaluate(model, device)
        if final_miou > best_miou:
            best_miou = final_miou
            best_dice = final_dice
        print(f"eval ep {epoch:3d} | miou: {final_miou:.4f} | dice: {final_dice:.4f} | best_miou: {best_miou:.4f}")

    t_end = time.time()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0
    print("---")
    print(f"best_miou:        {best_miou:.4f}")
    print(f"best_dice:        {best_dice:.4f}")
    print(f"final_miou:       {final_miou:.4f}")
    print(f"final_dice:       {final_dice:.4f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"startup_seconds:  {t_start_training - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_epochs:       {epoch}")
    print(f"num_steps:        {step}")
    print(f"num_params:       {num_params:,}")


if __name__ == "__main__":
    main()
