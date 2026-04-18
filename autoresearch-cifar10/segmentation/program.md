# autoresearch Oxford-IIIT Pet segmentation

This is the segmentation companion to the CIFAR-10 autoresearch run.

## Setup

The goal is to improve semantic segmentation on Oxford-IIIT Pet trimaps under a fixed wall-clock training budget.

Read these files before changing anything:
- `segmentation/prepare.py`: fixed constants, dataset loading, and evaluation. Do not modify during experiments.
- `segmentation/train.py`: the only file to edit during segmentation autoresearch.

Initialize `segmentation/results.tsv` with this header if it does not exist:

```text
commit	miou	dice	memory_gb	status	description
```

## Experimentation

Launch one experiment with:

```bash
cd segmentation
uv run python train.py > run.log 2>&1
```

The script prints `best_miou`, `best_dice`, and `peak_vram_mb`. Keep changes that improve `best_miou`; use Dice as a secondary tie-breaker.

## Rules

- Modify only `segmentation/train.py` during the segmentation loop.
- Do not modify `segmentation/prepare.py` during experiments.
- Do not add dependencies; use only the repository environment.
- Validation may run at most once per epoch.
- If a run crashes or exceeds 15 minutes, log it as `crash` or `discard` and revert.

Promising directions: input resolution, batch size, decoder width, pretrained encoder freezing schedule, optimizer/LR schedule, loss combinations, class weighting, boundary handling, augmentations, AMP, and faster data loading.
