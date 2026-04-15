from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

NUM_WORKERS = 4
TIME_BUDGET_S = 60
DATASET_DIR = os.environ.get(
    "AUTOVISION_DATA_DIR", "/ocean/projects/cis250278p/ytan8/datasets/autovision"
)
IMAGE_SIZE = 128
NUM_CLASSES = 3


class PetSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, image_size: int = IMAGE_SIZE):
        self.base = datasets.OxfordIIITPet(
            DATASET_DIR,
            split=split,
            target_types="segmentation",
            download=True,
        )
        self.image_tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.mask_tf = transforms.Resize(
            (image_size, image_size), interpolation=InterpolationMode.NEAREST
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, mask = self.base[idx]
        image = self.image_tf(image)
        mask = self.mask_tf(mask)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long) - 1
        mask = mask.clamp_(0, NUM_CLASSES - 1)
        return image, mask


def make_train_loader(batch_size: int):
    full = PetSegmentationDataset("trainval")
    train_len = int(0.85 * len(full))
    val_len = len(full) - train_len
    generator = torch.Generator().manual_seed(42)
    train_set, _ = random_split(full, [train_len, val_len], generator=generator)
    return DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )


def make_eval_loader(batch_size: int):
    full = PetSegmentationDataset("test")
    return DataLoader(
        full,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )


class Eval:
    def __init__(self, batch_size: int = 16):
        self.loader = make_eval_loader(batch_size)

    @torch.inference_mode()
    def evaluate(self, model, device):
        model.eval()
        intersections = torch.zeros(NUM_CLASSES, device=device)
        unions = torch.zeros(NUM_CLASSES, device=device)
        dice_nums = torch.zeros(NUM_CLASSES, device=device)
        dice_dens = torch.zeros(NUM_CLASSES, device=device)

        for images, masks in self.loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            preds = model(images).argmax(1)
            for cls in range(NUM_CLASSES):
                pred_c = preds == cls
                mask_c = masks == cls
                inter = (pred_c & mask_c).sum()
                intersections[cls] += inter
                unions[cls] += (pred_c | mask_c).sum()
                dice_nums[cls] += 2 * inter
                dice_dens[cls] += pred_c.sum() + mask_c.sum()

        miou = (intersections / unions.clamp_min(1)).mean().item()
        dice = (dice_nums / dice_dens.clamp_min(1)).mean().item()
        return miou, dice
