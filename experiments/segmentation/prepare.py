"""
prepare.py — Fixed evaluation harness for Oxford-IIIT Pet Segmentation.
This file is READ-ONLY. The agent must NOT modify it.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

NUM_WORKERS = 4
TIME_BUDGET_S = 480  # 8 minutes (segmentation needs more time)
DATASET_DIR = "../../data"
NUM_CLASSES = 3  # foreground, background, boundary
IMG_SIZE = 128  # smaller resolution for faster iteration


class SegTransform:
    """Joint transform for image + mask pairs."""

    def __init__(self, img_size=IMG_SIZE, is_train=False):
        self.img_size = img_size
        self.is_train = is_train
        self.img_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __call__(self, img, mask):
        # Resize
        img = transforms.functional.resize(img, (self.img_size, self.img_size))
        mask = transforms.functional.resize(
            mask, (self.img_size, self.img_size),
            interpolation=transforms.InterpolationMode.NEAREST,
        )

        if self.is_train:
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                img = transforms.functional.hflip(img)
                mask = transforms.functional.hflip(mask)

        # To tensor
        img = transforms.functional.to_tensor(img)
        img = self.img_normalize(img)

        # Mask: Oxford-IIIT Pet trimaps have values {1, 2, 3}
        # Convert to {0, 1, 2} for cross-entropy
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        mask = mask - 1  # {1,2,3} -> {0,1,2}
        mask = mask.clamp(0, NUM_CLASSES - 1)

        return img, mask


class PetSegDataset(torch.utils.data.Dataset):
    """Wrapper to apply joint transforms to Oxford-IIIT Pet."""

    def __init__(self, split="trainval", is_train=True):
        self.dataset = OxfordIIITPet(
            root=DATASET_DIR,
            split=split,
            target_types="segmentation",
            download=True,
        )
        self.transform = SegTransform(img_size=IMG_SIZE, is_train=is_train)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        return self.transform(img, mask)


class Eval:
    def __init__(self):
        test_set = PetSegDataset(split="test", is_train=False)
        self.loader = DataLoader(
            test_set,
            batch_size=16,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    @torch.inference_mode()
    def evaluate(self, model, device):
        """Returns (mean_loss, mIoU_percent)."""
        model.eval()
        total_loss = 0.0
        total_pixels = 0
        intersection = torch.zeros(NUM_CLASSES, device=device)
        union = torch.zeros(NUM_CLASSES, device=device)

        for inputs, targets in self.loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            # outputs shape: (B, NUM_CLASSES, H, W)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            total_loss += loss.item() * targets.size(0)
            total_pixels += targets.numel()

            preds = outputs.argmax(dim=1)
            for c in range(NUM_CLASSES):
                pred_c = (preds == c)
                target_c = (targets == c)
                intersection[c] += (pred_c & target_c).sum()
                union[c] += (pred_c | target_c).sum()

        # mIoU
        iou_per_class = intersection / (union + 1e-6)
        miou = iou_per_class.mean().item() * 100.0

        mean_loss = total_loss / len(self.loader.dataset)
        return mean_loss, miou
