# Progress Report: AutoVision — Learning to Automatically Design Vision Pipelines for Classification and Segmentation

**Authors:** Yusheng Tan &emsp; Yuanjun Feng &emsp; Yimu Liu

*CSE 559a*

---

## Abstract

We present a progress report for AutoVision, a project that explores whether an AI coding agent can autonomously improve computer vision pipelines through iterative code modification and evaluation. Building on the AutoResearch framework [1, 2], our system uses a large language model (Claude) to iteratively modify training scripts, evaluating each modification under a fixed five-minute compute budget and retaining only those changes that yield measurable improvement. We apply this methodology to two complementary vision tasks: image classification on CIFAR-10 [3] using a ResNet-20 [4] baseline, and semantic segmentation on the Oxford-IIIT Pet dataset [5] using a U-Net [6] baseline. In this report, we describe our current progress: we have collected and validated both datasets, adopted and tested the ResNet-20 classification baseline from the Erhard repository, and studied the agent loop infrastructure. We outline our detailed approach and the remaining steps toward our final evaluation.

---

## 1. Project Overview

Our project investigates whether an AI coding agent can serve as a principled replacement for the manual hyperparameter tuning and architecture search process that is central to building competitive vision models. The key insight, drawn from the AutoResearch framework [1], is to frame the search process as an iterative agent loop: an LLM-based coding agent reads a training script and a natural-language instruction file, proposes a single modification, and the modification is evaluated under a strict five-minute wall-clock budget. Modifications that improve the validation metric are retained; all others are reverted.

We apply this approach to two tasks:

1. **Image Classification**: CIFAR-10 with a ResNet-20 baseline [4]. The ResNet-20 architecture was specifically designed for CIFAR-scale images and provides a well-understood starting point, typically achieving ~91--92% test accuracy with standard training.
2. **Semantic Segmentation**: Oxford-IIIT Pet dataset [5] with a U-Net baseline [6]. This task extends our investigation to a dense prediction setting, where the agent must optimize for pixel-level accuracy (mIoU, Dice coefficient) rather than global classification accuracy.

**Changes from Proposal.** Our project goals and methodology remain unchanged from the original proposal. We continue to target both the minimum goal (successful agent loop for classification) and the maximum goal (both tasks with cross-task analysis of agent behavior).

---

## 2. Team Member Roles/Tasks

### 2.1 Yusheng Tan

1. Implement and maintain the AutoResearch outer loop, including the agent harness, timing mechanism, and keep/revert logic.
2. Author and refine the `program.md` instruction files for both the classification and segmentation experiments.
3. Coordinate the overall experimental workflow, maintain logs of agent decisions across runs, and manage the shared codebase.

### 2.2 Yuanjun Feng

1. Implement and validate the ResNet-20 classification baseline on CIFAR-10, following the original He et al. [4] specification.
2. Run the agent loop for the classification task and record all experimental outcomes.
3. Summarize and analyze the sequence of modifications the agent made throughout the classification search.

### 2.3 Yimu Liu

1. Implement the U-Net segmentation baseline on the Oxford-IIIT Pet dataset with a pretrained ResNet encoder.
2. Run the agent loop for the segmentation task and record results.
3. Implement evaluation metrics (mIoU, Dice coefficient) and produce comparative plots for the final report.

---

## 3. Collaboration Strategy

Our team uses a shared GitHub repository for version control, with each member working on a dedicated branch corresponding to their component (classification baseline, segmentation baseline, and AutoResearch loop). We merge changes to the main branch through pull requests after code review. Data and model checkpoints are stored on a shared Google Drive accessible to all members. We hold brief synchronous meetings twice per week via Zoom to discuss progress and coordinate next steps, and maintain an active group chat for asynchronous communication.

---

## 4. Proposed Approach

Our approach consists of three interconnected components, described below.

### 4.1 Baseline Models

**Classification.** For the classification baseline, we adopt the ResNet-20 implementation from the `autoresearch-cifar10` repository [2], which provides a faithful reproduction of the architecture described in He et al. [4] (Section 4.2). The model consists of three groups of residual blocks (*n*=3, totaling 6*n*+2=20 layers) with filter widths 16→32→64, identity shortcuts with zero-padding (Option A from the original paper), and Kaiming initialization. Training uses SGD with momentum 0.9, weight decay 10⁻⁴, initial learning rate 0.1 with step-based decay (divided by 10 at steps 32k and 48k out of 64k total), batch size 128, and standard CIFAR augmentations (4-pixel padding with random 32×32 crop, random horizontal flip). This baseline achieves 91.89% test accuracy in a full run [2], consistent with the published result of 91.25%.

**Segmentation.** For the segmentation baseline, we implement U-Net [6] using a pretrained ResNet-34 encoder (from `torchvision.models`) as the contracting path, with a symmetric expanding path using transposed convolutions and skip connections. The segmentation target is a 3-class trimap (foreground, background, boundary) from the Oxford-IIIT Pet dataset. We train with cross-entropy loss and evaluate using mean Intersection over Union (mIoU) and Dice coefficient. We are also aware of DeepLabV3+ [7] as a potentially stronger baseline and may explore it depending on the agent's modifications.

### 4.2 AutoResearch Agent Loop

Following the AutoResearch paradigm [1, 2], we do not implement a separate outer-loop script. Instead, the agent (Claude, via Claude Code) is given a `program.md` file containing detailed instructions and operates autonomously. The loop proceeds as follows:

1. The agent reads `program.md` and the current `train.py`, which is the *only* file it may modify.
2. The agent proposes a single modification, edits `train.py`, and commits the change via `git`.
3. The agent runs the training script under a fixed wall-clock time budget (5 minutes for classification). Output is redirected to a log file to avoid flooding the agent's context.
4. If `best_test_acc` improves, the commit is kept and the git branch advances. If it does not improve, the agent performs a `git reset` to revert to the previous best state.
5. The agent appends each result (commit hash, accuracy, memory usage, keep/discard status, description) to a `results.tsv` log file.
6. The process repeats indefinitely until manually stopped, typically running 50--100+ iterations overnight.

The `program.md` file also encodes constraints (e.g., `prepare.py` is read-only, no new dependencies) and a simplicity criterion: changes that add complexity must justify it with significant accuracy gains.

### 4.3 Analysis

After completing the agent loops for both tasks, we will analyze: (1) the trajectory of accuracy/mIoU improvement over iterations; (2) the categories of modifications the agent favors (e.g., data augmentation, architecture changes, optimizer tuning, regularization); and (3) whether the agent's strategies differ meaningfully between classification and segmentation.

---

## 5. Data

### 5.1 CIFAR-10

CIFAR-10 [3] consists of 60,000 color images of size 32×32 across 10 mutually exclusive categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The dataset is split into 50,000 training images and 10,000 test images. It is available through `torchvision.datasets.CIFAR10` and requires no manual preprocessing. We have downloaded and verified the dataset.

![Sample images from the CIFAR-10 dataset, showing the characteristic 32×32 resolution across representative categories.](figures/cifar10_samples.png)

### 5.2 Oxford-IIIT Pet

The Oxford-IIIT Pet dataset [5] contains 7,349 images of cats and dogs spanning 37 breed categories. Each image is accompanied by a pixel-level trimap annotation with three classes: foreground (pet), background, and boundary. The training/validation split contains 3,680 images and the test split contains 3,669 images. Images vary in resolution and are resized to 256×256 for training. The dataset is available through `torchvision.datasets.OxfordIIITPet`. We have downloaded the dataset and confirmed that the segmentation masks load correctly.

![Sample images from the Oxford-IIIT Pet dataset with corresponding trimap segmentation masks. The trimaps distinguish foreground (pet), background, and boundary regions.](figures/pet_samples.png)

---

## 6. Initial Results

We have completed the following steps toward our project goals:

1. **Data collection and verification.** Both CIFAR-10 and the Oxford-IIIT Pet dataset have been downloaded via `torchvision`, loaded into PyTorch dataloaders, and visually inspected to confirm correct labeling and mask alignment.

2. **Reference codebase study.** We have cloned and studied both the original AutoResearch repository by Karpathy [1] and the CIFAR-10 adaptation by Erhard [2]. The Erhard repository provides a complete, working setup for CIFAR-10 classification research under the AutoResearch paradigm, including a faithful ResNet-20 baseline (`train.py`), a fixed evaluation harness (`prepare.py`), and a detailed agent instruction file (`program.md`). We have verified that the code compiles and runs correctly on our hardware.

3. **ResNet-20 classification baseline.** We have adopted and tested the ResNet-20 baseline from the Erhard repository [2], which faithfully reproduces the He et al. [4] specification. Under the 5-minute time budget, the model trains for approximately 40 epochs and reaches ~86--88% test accuracy. This is consistent with published behavior: the full 64k-step schedule achieves 91.89% test accuracy [2].

4. **U-Net segmentation baseline.** The encoder (pretrained ResNet-34) and decoder architecture have been implemented. Data loading with segmentation masks is functional. We are currently adapting the AutoResearch paradigm for this task, which requires defining a new `program.md` and replacing the accuracy metric with mIoU.

5. **AutoResearch loop readiness.** For the classification task, the Erhard repository provides a fully operational agent loop. We have studied the `program.md` structure, which instructs the agent to autonomously iterate: modifying `train.py`, committing changes via `git`, running training, and keeping or reverting based on `best_test_acc`. Our next step is to launch the first full overnight agent run.

Our remaining work is focused on: (a) running the full agent loop for classification (50--100+ iterations); (b) adapting the AutoResearch setup for the segmentation task with a new `program.md`; (c) running the segmentation agent loop; and (d) analyzing and comparing the agent's modification strategies across both tasks.

---

## 7. Current Concerns and Questions

1. **Segmentation convergence under time constraints.** Segmentation training is inherently slower than classification due to larger input sizes (256×256 vs. 32×32) and pixel-level loss computation. We are concerned that 5 minutes may not be sufficient for the agent to observe meaningful improvement signals on the segmentation task. We plan to mitigate this by testing extended time limits (8--10 minutes) or reducing the input resolution for the segmentation loop.

2. **Agent stalling behavior.** Prior work [2] reports that the agent tends to stop producing substantive modifications after 90--100 iterations, effectively plateauing. We plan to address this by periodically updating the `program.md` prompt with refreshed instructions that explicitly encourage novel exploration directions.

3. **Reproducibility.** Since the agent's responses are stochastic (temperature > 0), running the same loop twice may yield different modification trajectories. We plan to run at least two complete loops per task and report the variability in both final metrics and modification strategies.

---

## References

[1] A. Karpathy, "AutoResearch," 2025. <https://github.com/karpathy/autoresearch>

[2] G. Erhard, "autoresearch-cifar10," 2025. <https://github.com/GuillaumeErhard/autoresearch-cifar10>

[3] A. Krizhevsky, "Learning Multiple Layers of Features from Tiny Images," Technical Report, 2009.

[4] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," *CVPR*, 2016.

[5] O. M. Parkhi, A. Vedaldi, A. Zisserman, and C. V. Jawahar, "Cats and Dogs," *CVPR*, 2012.

[6] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," *MICCAI*, 2015.

[7] L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, and H. Adam, "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation," *ECCV*, 2018.
