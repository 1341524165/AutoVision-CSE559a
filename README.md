# AutoVision — CSE 559a

An AI coding agent that autonomously improves computer vision pipelines through iterative code modification, applied to two tasks:

- **Classification**: CIFAR-10 with a ResNet-20 baseline (He et al. 2016)
- **Segmentation**: Oxford-IIIT Pet with a U-Net baseline (pretrained ResNet-34 encoder)

The agent reads `program.md` and `train.py`, proposes one change, commits it, trains under a fixed time budget, and keeps the commit only if the metric improves — otherwise reverts via `git reset`.

## Quick Start

```bash
# Run the self-contained baseline (generates dataset figures + trains ResNet-20 for 5 min)
python experiments/quick_baseline.py

# Compile the progress report
cd "Progress Report"
latexmk -pdf 2_progress_template.tex
latexmk -C   # clean build artifacts
```

## ResNet-20 Baseline Spec

| Setting | Value |
|---------|-------|
| Architecture | 3 groups × n=3 blocks = 20 layers; filters 16→32→64 |
| Shortcuts | Identity with zero-padding (Option A, no projection) |
| Init | Kaiming normal |
| Optimizer | SGD, lr=0.1, momentum=0.9, weight\_decay=1e-4 |
| LR schedule | ×0.1 at steps 32k and 48k (total 64k steps) |
| Batch size | 128 |
| Augmentation | 4px padding + random 32×32 crop + horizontal flip |
| Normalization | Channel-wise mean subtraction, std=1 |

## Repository Layout

```
Project Plan/                    project proposal (markdown)
Progress Report/                 LaTeX source + figures for the paper
data/                            CIFAR-10 + Oxford-IIIT Pet datasets
experiments/
├── classification/
│   ├── prepare.py               fixed evaluation harness (READ-ONLY)
│   ├── train.py                 agent-modifiable training script
│   └── program.md               agent instructions
├── segmentation/
│   ├── prepare.py               fixed evaluation harness (READ-ONLY)
│   ├── train.py                 agent-modifiable training script
│   └── program.md               agent instructions
└── quick_baseline.py            quick validation script
```
