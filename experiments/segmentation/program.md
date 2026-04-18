# AutoVision — Oxford-IIIT Pet Segmentation Agent Instructions

This is an experiment to have an AI coding agent autonomously improve a U-Net segmentation baseline on the Oxford-IIIT Pet dataset. The agent iteratively modifies the training script, evaluates under a fixed time budget, and keeps only improvements.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr27-seg`) and the time budget in `prepare.py`. The branch `autoresearch/<tag>` must not already exist — check for that.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read the files for full context:
   - `prepare.py`: fixed constants, dataset loading, and evaluation function. **Do not modify.**
   - `train.py`: the file you modify. Model architecture, optimizer, lr, augmentation, regularization, training loop, etc.
4. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
5. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget defined in `prepare.py`** (wall clock training time, excluding startup and validation). You launch it simply with:

```bash
python train.py > run.log 2>&1
```

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, data augmentation, hyperparameters, training loop, batch size, model size, loss function, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, dataset loading, and time budget.
- Install new packages or add dependencies. You can only use what is already available (PyTorch, torchvision, numpy, standard library).
- Modify the evaluation harness. The `Eval.evaluate()` method in `prepare.py` is the ground truth metric.

**The goal is simple: get the highest test mIoU (`best_test_miou`).** Since the training time budget is fixed, focus only on getting the best hyperparameters and training code setup. The first constraint is that the code runs without crashing and finishes within the time budget. The second is not to run the validation more than once per epoch.

**Key differences from the classification task:**
- The metric is **mIoU** (mean Intersection over Union), not accuracy.
- Images are 128×128 (much larger than CIFAR-10's 32×32), so training is slower.
- The model uses a **pretrained encoder** (ResNet-34 from ImageNet).
- There are only **3 classes** (foreground, background, boundary).

**VRAM** is a soft constraint. Some increase is acceptable for meaningful improvement.

**Simplicity criterion**: All else being equal, simpler is better.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as-is.

## Output Format

Once the script finishes it prints a summary like this:

```
---
best_test_miou:   65.43%
final_test_miou:  65.43%
final_test_loss:  0.4321
training_seconds: 480.1
total_seconds:    510.3
startup_seconds:  5.2
peak_vram_mb:     2345.6
num_epochs:       30
num_steps:        6900
num_params:       24,456,259
```

You can extract the key metrics from the log file:

```bash
grep "^best_test_miou:\|^peak_vram_mb:" run.log
```

## Logging Results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
commit	best_miou	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. `best_test_miou` achieved (e.g. 65.43) — use 0.00 for crashes
3. peak memory in GB, round to .1f — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

## Agent Loop

Same as the classification task — LOOP FOREVER:

1. Look at the git state.
2. Tune `train.py` with an experimental idea.
3. `git commit`
4. Run: `python train.py > run.log 2>&1`
5. Read results: `grep "^best_test_miou:\|^peak_vram_mb:" run.log`
6. If crashed, check `tail -n 50 run.log`.
7. Append result to `results.tsv`.
8. If `best_test_miou` improved → keep commit. Otherwise → `git reset`.

**NEVER STOP**: Continue indefinitely until manually interrupted.
