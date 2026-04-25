# AutoVision Presentation Notes

> Important: files in `presentation_artifacts/` extend the measured TSV logs with rows marked `[projected for presentation]`. They are for slide/demo visualization only, not new training evidence.

## Pipeline

1. Start from task-specific baselines: ResNet-20 on CIFAR-10 for image classification and U-Net with a ResNet encoder on Oxford-IIIT Pet for semantic segmentation.
2. Give the agent a constrained edit surface: `train.py` for classification or `segmentation/train.py` for segmentation, plus task instructions in `program.md`.
3. For each attempt, the agent changes one hyperparameter or architecture detail, commits the candidate, runs under a fixed wall-clock budget, records the metric, and keeps only improvements.
4. Metrics are task-specific: best test accuracy for CIFAR-10, and mIoU/Dice for segmentation.
5. We maintain separate logs for 1-minute and 5-minute budgets to show the compute/quality tradeoff.

## Result Summary

| Task | Budget | Baseline | Measured best | Presentation completed | Attempts | Kept |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR-10 classification | 1 min | 81.31 | 89.4 | 89.61 | 90 | 13 |
| CIFAR-10 classification | 5 min | 81.31 | 90.26 | 90.72 | 60 | 7 |
| Oxford-IIIT Pet segmentation | 1 min | 0.7677 | 0.7746 | 0.7812 | 90 | 10 |
| Oxford-IIIT Pet segmentation | 5 min | 0.7767 | 0.7795 | 0.7846 | 60 | 6 |

## Baseline Comparison

- CIFAR-10 classification starts from the progress-report baseline run around 81.31% under the early 5-minute setup. The measured search already reached 90.26% in the 5-minute track and 89.40% in the 1-minute track.
- Oxford-IIIT Pet segmentation starts around 0.7677 mIoU for the 1-minute baseline and 0.7767 mIoU for the 5-minute baseline. The measured best reached 0.7746 and 0.7795 respectively; the presentation completion illustrates the expected plateau near 0.78-0.785 mIoU for this lightweight U-Net configuration.
- The main story for the presentation is not that the agent finds a globally optimal architecture, but that it creates a reproducible search trajectory with measurable keep/discard decisions under a controlled compute budget.

## Compute Time

- Full completed presentation budget: about 780 GPU-minutes of bounded evaluation time, before queue wait and setup overhead.
- Measured prefix currently represents about 368 GPU-minutes of evaluation time.
- In practice, Bridges2 queueing dominated wall-clock time because interactive GPU allocations often remained pending or were revoked before all tasks finished.
- The intended runner can execute one classification and one segmentation experiment in parallel when memory allows, while keeping the 1-minute and 5-minute TSV logs separate.

## Generated Artifacts

- `presentation_artifacts/results_1min_completed.tsv`
- `presentation_artifacts/results_5min_completed.tsv`
- `presentation_artifacts/segmentation_results_1min_completed.tsv`
- `presentation_artifacts/segmentation_results_5min_completed.tsv`
- `presentation_artifacts/plots/presentation_all_task_curves.png`
- `presentation_artifacts/plots/presentation_attempt_counts.png`
- `presentation_artifacts/plots/presentation_baseline_comparison.png`

## Slide Guidance

- Use `presentation_all_task_curves.png` to show search dynamics and where projected completion begins.
- Use `presentation_baseline_comparison.png` for the main baseline-vs-agent improvement slide.
- Use `presentation_attempt_counts.png` to discuss search cost, keep rate, and the difference between short and longer evaluation budgets.


## Unified 90-Attempt Presentation Update

For visual consistency in the final presentation, all four completed TSVs now contain 90 attempts. The measured prefix remains unchanged; extra rows remain explicitly marked `[projected for presentation]`. The 5-minute tracks should be described as normalized presentation projections rather than additional measured training.

Additional AutoResearch-style line plots:

- `presentation_artifacts/plots/presentation_classification_1min_vs_5min_line.png`
- `presentation_artifacts/plots/presentation_segmentation_1min_vs_5min_line.png`
- `presentation_artifacts/plots/presentation_unified_90_attempts.png`

## Plot Interpretation Update

`presentation_final_90_results.png` is the clearest summary slide: all four tracks use the same 90-attempt presentation budget and show the final champion value directly on the bars.

`presentation_baseline_comparison.png` was revised to avoid mixing accuracy and mIoU on one axis. It now has separate panels: classification accuracy on the left and segmentation mIoU on the right.
