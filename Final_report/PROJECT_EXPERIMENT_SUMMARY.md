# AutoVision / autoresearch-cifar10 Project Summary

Last updated: 2026-04-30

This document summarizes the repository structure, current experiment state, measured results, presentation-only artifacts, and current conclusions. It is based on the local files in this checkout, not on a fresh rerun.

## 1. Project Purpose

This repository started as an adaptation of Karpathy-style AutoResearch to CIFAR-10 classification. The current project has expanded into an AutoVision experiment with two vision tasks:

1. CIFAR-10 image classification with a ResNet-20-style baseline.
2. Oxford-IIIT Pet semantic segmentation with a ResNet34-UNet-style baseline.

The core idea is to let an AI coding agent iteratively edit the training script, run a fixed-budget experiment, log metrics, and keep only changes that improve the target metric.

For classification, the target metric is `best_test_acc`.

For segmentation, the target metric is `best_miou`, with Dice used as secondary context.

## 2. Repository Structure

```text
.
├── README.md
├── program.md
├── prepare.py
├── train.py
├── pyproject.toml
├── uv.lock
├── results.tsv
├── results_1min.tsv
├── results_5min.tsv
├── search_results/
├── segmentation/
│   ├── program.md
│   ├── prepare.py
│   ├── train.py
│   ├── results.tsv
│   ├── results_1min.tsv
│   └── results_5min.tsv
├── scripts/
├── autoresearch_jobs/
├── plots/
├── presentation_artifacts/
├── data/
└── Progress_Report.pdf
```

### Core Files

| Path | Role |
|---|---|
| `README.md` | Upstream-style description of the original CIFAR-10 autoresearch experiment and historical results. |
| `program.md` | Instruction file for the classification autoresearch loop. It says `prepare.py` is fixed and `train.py` is the main editable file. |
| `prepare.py` | CIFAR-10 fixed evaluation harness. Defines `TIME_BUDGET_S`, `DATASET_DIR`, test transforms, and `Eval.evaluate()`. |
| `train.py` | CIFAR-10 model and training loop. This is where classification experiments modify architecture, optimizer, scheduler, batch size, etc. |
| `segmentation/program.md` | Instruction file for segmentation autoresearch. It restricts experiment edits to `segmentation/train.py`. |
| `segmentation/prepare.py` | Oxford-IIIT Pet dataset loader and fixed mIoU/Dice evaluation. |
| `segmentation/train.py` | ResNet34-UNet segmentation model and training loop. |
| `results_1min.tsv`, `results_5min.tsv` | Measured classification experiment logs for 1-minute and 5-minute budgets. |
| `segmentation/results_1min.tsv`, `segmentation/results_5min.tsv` | Measured segmentation experiment logs for 1-minute and 5-minute budgets. |
| `search_results/` | Historical CIFAR-10 result logs from the original/autoresearch comparison setup. These are not the same as the current four-track AutoVision run. |
| `scripts/` | Slurm/HPC helper scripts for allocating GPUs and launching batch experiments. |
| `autoresearch_jobs/` | Local batch-run machinery, logs, temporary worktrees, lock files, and plotting/report helper scripts. |
| `plots/` | Plots generated from measured logs. |
| `presentation_artifacts/` | Slide/demo artifacts. Some TSV rows here are projected for presentation and are not measured training evidence. |
| `Progress_Report.pdf` | Older progress report. It is useful background but is stale relative to current TSV results. |
| `data/` | Local CIFAR-10 data. This is an artifact, not source code. |

## 3. Current Code State

Current Git state:

```text
HEAD (no branch), detached at 85dc653
```

Tracked modifications currently visible in the worktree:

```text
M scripts/run_autovision_gpu_batch.sh
M scripts/salloc_autovision_supervisor.sh
M segmentation/prepare.py
```

The two shell script changes are mode-only changes: the executable bit changed from `100755` to `100644`.

The `segmentation/prepare.py` change is behavioral:

```text
TIME_BUDGET_S = 300 -> TIME_BUDGET_S = 60
```

Untracked artifacts include experiment TSVs, generated plots, `autoresearch_jobs/`, `data/`, `presentation_artifacts/`, and `Progress_Report.pdf`.

### Current Classification Code

Current `train.py` is a ResNet-20-like CIFAR classifier with:

- `BATCH_SIZE = 160`
- `WEIGHT_DECAY = 1.5e-4`
- `LABEL_SMOOTHING = 0.025`
- `MAX_STEPS = 5000`
- SGD with Nesterov momentum
- `OneCycleLR(max_lr=0.5, pct_start=0.3)`
- SiLU activations
- 1x1 projection shortcut on downsampling residual blocks

Important: the current checked-out classification code is not exactly the best measured 5-minute classification champion. The measured 5-minute champion is commit `be555c9`, which used `BATCH_SIZE = 176` and `max_lr = 0.45`.

### Current Segmentation Code

Current `segmentation/train.py` is a ResNet34-UNet-style model with:

- pretrained torchvision ResNet34 encoder
- transposed-convolution decoder with skip connections
- `BATCH_SIZE = 12`
- `LR = 2.5e-4`
- `WEIGHT_DECAY = 1e-4`
- `DECODER_CHANNELS = 64`
- `EVAL_BATCH_SIZE = 16`
- `FREEZE_ENCODER_EPOCHS = 0`
- AdamW optimizer
- CosineAnnealingLR
- CUDA AMP enabled when CUDA is available

Current `segmentation/prepare.py` uses:

- `TIME_BUDGET_S = 60`
- `IMAGE_SIZE = 128`
- `NUM_CLASSES = 3`
- Oxford-IIIT Pet trimap masks mapped to classes `0..2`
- fixed mIoU and Dice evaluation on the test split

## 4. Experiment Design

The AutoResearch loop follows this pattern:

1. Start from a current champion commit.
2. Modify only the in-scope training file.
3. Commit the candidate.
4. Run training under a fixed wall-clock budget.
5. Parse summary metrics from the log.
6. Append one row to the relevant TSV.
7. Keep the commit if the metric improves; otherwise revert or move back to the prior champion.

The current four-track batch setup is represented in `autoresearch_jobs/four_task_batch.py`:

| Track | File | Budget | Target Attempts | Metric |
|---|---|---:|---:|---|
| `class1` | `results_1min.tsv` | 60s | 90 | best accuracy |
| `class5` | `results_5min.tsv` | 300s | 60 | best accuracy |
| `seg1` | `segmentation/results_1min.tsv` | 60s | 90 | best mIoU |
| `seg5` | `segmentation/results_5min.tsv` | 300s | 60 | best mIoU |

The intended runner order is:

```text
seg1 -> class1 -> seg5 -> class5
```

The HPC scripts are strongly tied to Bridges2 paths and account/user settings:

```text
/jet/home/ytan8/Code/test2/autoresearch-cifar10
/ocean/projects/cis250278p/ytan8/envs/autoresearch-cifar10/bin/python
/ocean/projects/cis250278p/ytan8/datasets/autovision
```

This means the scripts are not portable without editing paths and environment variables.

## 5. Current Measured Experiment Status

These are measured rows from the local TSV files, excluding presentation-only projected rows.

### Classification: CIFAR-10, 1 Minute

Source: `results_1min.tsv`

| Quantity | Value |
|---|---:|
| Measured attempts | 67 |
| Kept | 10 |
| Discarded | 49 |
| Crashed | 8 |
| Best measured accuracy | 89.40% |
| Best commit | `2f96987` |
| Best description | `safe weight decay 1.5e-4` |
| Peak memory at best | 0.3 GB |

Trajectory summary:

- Baseline row: `81.31%`
- Early large wins came from scheduler changes, Nesterov, OneCycleLR, SiLU activations, projection shortcuts, and label smoothing.
- Later attempts mostly searched small hyperparameter changes around the champion and often failed to improve.

### Classification: CIFAR-10, 5 Minutes

Source: `results_5min.tsv`

| Quantity | Value |
|---|---:|
| Measured attempts | 29 |
| Kept | 4 |
| Discarded | 25 |
| Crashed | 0 |
| Best measured accuracy | 90.26% |
| Best commit | `be555c9` |
| Best description | `v2 5min max_lr 0.45 on bs160` |
| Peak memory at best | 0.4 GB |

Trajectory summary:

- The 5-minute track starts from the existing short-run champion rerun at `89.49%`.
- The best measured 5-minute row is `90.26%`.
- Most 5-minute variants around LR, label smoothing, weight decay, and batch size did not improve.

### Segmentation: Oxford-IIIT Pet, 1 Minute

Source: `segmentation/results_1min.tsv`

| Quantity | Value |
|---|---:|
| Measured attempts | 11 |
| Kept | 7 |
| Discarded | 4 |
| Crashed | 0 |
| Best measured mIoU | 0.7746 |
| Best measured Dice | 0.8625 |
| Best commit | `85dc653` |
| Best description | `seg 1min lr 2.5e-4` |
| Peak memory at best | 0.4 GB |

Trajectory summary:

- Baseline row: `0.7677` mIoU, `0.8535` Dice.
- Small gains came from decoder-channel reduction, eval batch tuning, unfreezing from epoch 0, and LR tuning.
- The track is far from the intended 90 measured attempts.

### Segmentation: Oxford-IIIT Pet, 5 Minutes

Source: `segmentation/results_5min.tsv`

| Quantity | Value |
|---|---:|
| Measured attempts | 29 |
| Kept | 3 |
| Discarded | 26 |
| Crashed | 0 |
| Best measured mIoU | 0.7795 |
| Best measured Dice | 0.8658 |
| Best commit | `3748b25` |
| Best description | `overnight seg batch size 12` |
| Peak memory at best | 0.5 GB |

Trajectory summary:

- Baseline row: `0.7767` mIoU, `0.8643` Dice.
- Best measured result: `0.7795` mIoU.
- Segmentation improvements are small and noisy. Many 5-minute variants are very close to the champion but not clearly better.

## 6. Presentation Artifacts vs Measured Evidence

`presentation_artifacts/AutoVision_Presentation_Report.md` explicitly says the completed TSV files contain projected rows marked `[projected for presentation]`.

Those files are useful for slide visuals, but they are not new training evidence.

| Presentation File | Rows | Measured Rows | Projected Rows | Best Value in File |
|---|---:|---:|---:|---:|
| `presentation_artifacts/results_1min_completed.tsv` | 90 | 67 | 23 | 89.61% |
| `presentation_artifacts/results_5min_completed.tsv` | 90 | 29 | 61 | 90.88% |
| `presentation_artifacts/segmentation_results_1min_completed.tsv` | 90 | 11 | 79 | 0.7812 mIoU |
| `presentation_artifacts/segmentation_results_5min_completed.tsv` | 90 | 29 | 61 | 0.7860 mIoU |

Recommendation for reports or slides:

- Use "measured best" for actual conclusions.
- Use "presentation projection" only when explaining visualization or anticipated trajectory.
- Do not present projected rows as completed experiments.

## 7. Historical CIFAR-10 Search Results

The `search_results/` directory contains older CIFAR-10 autoresearch comparison logs. These are stronger than the current AutoVision measured classification tracks but should be treated as a separate result family.

Selected summary:

| File | Rows | Best |
|---|---:|---:|
| `search_results/results_1min_auto_gen.tsv` | 59 | 91.86% |
| `search_results/results_5min_auto_gen.tsv` | 89 | 95.37% |
| `search_results/results_1min_hand.tsv` | 54 | 91.36% |
| `search_results/results_5min_hand.tsv` | 94 | 93.39% |

The strongest historical CIFAR row is:

```text
c2cf7d9    95.37    0.4    keep    replace ReLU with GeLU activation
```

Important interpretation:

- These logs support the general claim that the autoresearch loop can find strong CIFAR improvements.
- They should not be merged directly with the current four-track AutoVision results because the starting point, prompt/setup, and search trajectory differ.

## 8. Existing Experiment Conclusions

### Classification Conclusions

1. The autoresearch loop finds meaningful classification improvements quickly.
2. The biggest measured improvements are not exotic; they are standard training improvements:
   - scheduler replacement, especially OneCycleLR or cosine-style schedules
   - Nesterov SGD
   - tuned learning-rate peak and schedule shape
   - SiLU/GeLU-style activation swaps
   - projection shortcuts instead of zero-padding shortcuts
   - light label smoothing
   - moderate weight decay tuning
3. Higher throughput matters because the wall-clock budget is fixed. Batch size and model size trade off accuracy per step against steps per minute.
4. Large model-width increases often hurt under a short budget because they reduce the number of optimization steps.
5. The current four-track classification run has not reproduced the historical 95%+ CIFAR result; its measured best is 90.26% on the 5-minute track.

### Segmentation Conclusions

1. The segmentation track is much less responsive than classification under short budgets.
2. The measured mIoU gains are real but small:
   - 1-minute: `0.7677 -> 0.7746`
   - 5-minute: `0.7767 -> 0.7795`
3. The current ResNet34-UNet setup appears to plateau around `0.77-0.78` mIoU in these short experiments.
4. Longer wall-clock time alone has not produced a large jump in this measured set.
5. Useful knobs so far are relatively basic:
   - decoder channel width
   - batch size
   - encoder freeze/unfreeze schedule
   - learning rate
   - evaluation batch size
6. The segmentation task likely needs broader changes to move substantially:
   - better augmentation
   - loss rebalancing or boundary-aware loss
   - stronger decoder or modern segmentation head
   - validation split rather than direct test-set-driven selection
   - more stable multi-seed evaluation

### Cross-Task Conclusions

1. Classification gives clear and fast feedback; segmentation feedback is slower and noisier.
2. The keep/discard rule works mechanically, but it optimizes directly against the reported test metric.
3. Single-run improvements are not enough for a rigorous final claim. Multiple seeds or repeated champion reruns are needed.
4. The agent mostly performs local engineering search around known good practices rather than discovering radically new architectures.
5. The current project is strongest as a demonstration of an automated experimental workflow, not as proof of a globally optimal vision model.

## 9. Current Issues and Risks

1. The current checkout is detached, not on a named branch.
2. The worktree contains many untracked artifacts and generated outputs.
3. `Progress_Report.pdf` is stale:
   - it reports only an early 40-evaluation classification log;
   - it says segmentation is still being adapted;
   - it says segmentation images are resized to 256, while current code uses `IMAGE_SIZE = 128`.
4. `run.log` currently shows a CPU crash for classification because `torch.cuda.synchronize()` is called without checking whether CUDA is available.
5. `segmentation/run.log` is incomplete and only shows startup on CPU.
6. Some lock/log files in `autoresearch_jobs/` appear stale; file timestamps indicate no recent active experiment updates after Apr 30 15:12.
7. The HPC runner is hardcoded to one account, path layout, and environment.
8. Presentation completed TSVs contain projected rows. They must not be cited as actual measured results.
9. The measured four-track run has not reached the intended attempt targets:
   - `class1`: 67 / 90
   - `class5`: 29 / 60
   - `seg1`: 11 / 90
   - `seg5`: 29 / 60

## 10. Recommended Next Steps

For documentation/reporting:

1. Update `Progress_Report.pdf` or create a final report from the measured TSVs.
2. Make every figure caption distinguish measured rows from projected rows.
3. Avoid mixing `search_results/` historical CIFAR results with the current four-track AutoVision results unless explicitly labeled.

For experiment continuation:

1. Decide whether to resume the measured four-track run or freeze it for the current presentation.
2. If resuming, clean stale locks and ensure the Slurm runner points to the correct active checkout.
3. Restore or intentionally commit executable bits for the shell scripts.
4. Decide which champion code should be the final checkout:
   - current HEAD favors the 1-minute segmentation champion;
   - `be555c9` is the measured 5-minute classification champion;
   - `3748b25` is the measured 5-minute segmentation champion.
5. Add a CUDA guard around `torch.cuda.synchronize()` if local CPU execution should be supported.
6. For more rigorous claims, rerun final champions across multiple seeds and report mean/std.

