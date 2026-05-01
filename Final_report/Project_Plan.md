# AutoVision: Learning to Automatically Design Vision Pipelines for Classification and Segmentation

**Authors:** Yusheng Tan &emsp; Yuanjun Feng &emsp; Yimu Liu

*CSE 559a*

---

## Abstract

Designing effective computer vision pipelines typically requires extensive manual experimentation with model architectures, hyperparameters, and preprocessing strategies---a process that is both time-consuming and heavily dependent on prior experience. In this project, we explore whether an AI coding agent can take over this search process in a principled and largely unsupervised manner. Drawing on the AutoResearch framework [1], we propose a system in which an agent iteratively modifies a training script, evaluates each change under a fixed compute budget, and retains only those modifications that yield measurable improvement. We apply this approach to two tasks: image classification on CIFAR-10 [2] and semantic segmentation on the Oxford-IIIT Pet dataset [3], using ResNet-20 [4] and U-Net [5] as respective baselines. Our evaluation will compare agent-discovered configurations against hand-tuned baselines on standard metrics, and we aim to characterize the kinds of decisions the agent makes across the two tasks.

---

## 1. Project Overview

### 1.1 Motivation

Training a competitive vision model involves a large number of interdependent design decisions: architecture choice, optimizer selection, learning rate scheduling, data augmentation, and more. Arriving at a good configuration typically demands significant manual effort and domain knowledge, motivating the broader research interest in automated machine learning and neural architecture search.

A recent and particularly lightweight approach to this problem is the AutoResearch framework, which frames the search process as an iterative agent loop. Rather than predefining a search space, an AI coding agent is given a training script and a natural-language instruction file (`program.md`), and is allowed to freely modify the code. Each modification is evaluated under a fixed wall-clock budget (five minutes per experiment); changes are retained only if they improve the validation metric, and discarded otherwise. This paradigm has already been applied to CIFAR-10 classification [2], yielding configurations that substantially outperform the ResNet-20 baseline [4] without any manual intervention.

Motivated by these results, we aim to reproduce this setup and extend it to a second vision task---semantic segmentation---to examine whether the same agent-driven search strategy transfers across task types.

### 1.2 Project Description and Goals

Our project proceeds in two stages. In the first stage, we implement a faithful ResNet-20 baseline on CIFAR-10, integrate it with the AutoResearch loop, and run the agent for an extended period to observe what modifications it discovers. In the second stage, we apply the same methodology to semantic segmentation, using a U-Net [5] baseline trained on the Oxford-IIIT Pet dataset.

A secondary question we find interesting is whether the strategies the agent adopts differ meaningfully between the two tasks---for instance, whether augmentation techniques that help classification are also pursued in the segmentation setting, or whether the agent converges on distinct approaches for each.

**Minimum goal:** Successfully run the agent loop for image classification and demonstrate consistent improvement over the ResNet-20 baseline within a reasonable compute budget.

**Maximum goal:** Complete the agent loop for both tasks, compare the agent's modification history across tasks, and produce a brief analysis of what the agent found effective in each setting.

---

## 2. Team Member Roles/Tasks

### 2.1 Yusheng Tan

1. Implement and maintain the AutoResearch outer loop, including the agent harness, timing mechanism, and keep/revert logic.
2. Author and refine the `program.md` instruction files for both the classification and segmentation experiments.
3. Coordinate the overall experimental workflow and maintain logs of agent decisions across runs.

### 2.2 Yuanjun Feng

1. Implement the ResNet-20 classification baseline on CIFAR-10, following the original specification in [4].
2. Run the agent loop for the classification task and record all experimental outcomes.
3. Summarize and analyze the sequence of modifications the agent made throughout the search.

### 2.3 Yimu Liu

1. Implement the U-Net segmentation baseline on the Oxford-IIIT Pet dataset, with a pretrained ResNet encoder.
2. Run the agent loop for the segmentation task and record results.
3. Implement evaluation metrics (mIoU, Dice coefficient) and produce comparative plots for the final report.

---

## 3. Resources

**Datasets.** Image classification experiments will use CIFAR-10 (50,000 training and 10,000 test images across 10 categories). Semantic segmentation experiments will use the Oxford-IIIT Pet dataset, which provides pixel-level annotations for 37 pet categories. Both datasets are publicly available through `torchvision` and require no manual collection or annotation.

**Implementation.** All models will be implemented in PyTorch. Our system is inspired by open-source AutoResearch frameworks, including implementations by Karpathy (<https://github.com/karpathy/autoresearch>) and Erhard (<https://github.com/GuillaumeErhard/autoresearch-cifar10>), which automate the experimental loop of model design, training, and evaluation. We adapt these ideas to construct an automated pipeline for our classification and segmentation tasks. The segmentation model will be initialized with ImageNet-pretrained encoder weights from `torchvision.models`.

**Compute.** Experiments will be run on Google Colab (A100 or L4 GPU). Given the five-minute-per-run design of the AutoResearch loop, running 50--100 experiments overnight is computationally and financially feasible. Team members with access to personal NVIDIA GPUs will also contribute compute where needed.

**AI Agent.** We will use Claude (via the Anthropic API or Claude Code) as the coding agent, consistent with the setup in the original AutoResearch repository.

---

## 4. Reservations

Our primary concern is the feasibility of the segmentation loop. Segmentation training is inherently slower and noisier than classification, which may make it difficult for the agent to obtain a reliable improvement signal within a five-minute budget. We may need to increase the per-run time limit or reduce model complexity to make the loop effective.

A secondary concern is agent behavior over long runs. Prior work has observed that the agent tends to stop iterating autonomously after 70--90 attempts [6], requiring a manual prompt to resume. We will need to account for this in our experimental setup.

Finally, as this is our first time working with segmentation architectures such as DeepLabV3+ [7] in depth, there is likely to be a learning curve on that side of the project. We consider this a reasonable challenge for the scope of the course.

---

## 5. Relationship to Background

This project is directly grounded in the AutoResearch framework [1] and its CIFAR-10 adaptation by Erhard [6]. The classification component builds on the ResNet-20 architecture introduced by He et al. [4], which serves as a well-established CIFAR-10 benchmark. For segmentation, we adopt U-Net [5] as our starting point, given its straightforward encoder-decoder structure and its suitability for iterative agent-driven modification. We are also aware of DeepLabV3+ [7] as a more powerful segmentation approach, and may draw on it depending on the direction of the agent's experiments.

---

## References

[1] A. Karpathy, "AutoResearch," 2026. <https://github.com/karpathy/autoresearch>

[2] A. Krizhevsky, "Learning Multiple Layers of Features from Tiny Images," Technical Report, 2009.

[3] O. M. Parkhi, A. Vedaldi, A. Zisserman, and C. V. Jawahar, "Cats and Dogs," *CVPR*, 2012.

[4] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," *CVPR*, 2016.

[5] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," *MICCAI*, 2015.

[6] G. Erhard, "autoresearch-cifar10," 2025. <https://github.com/GuillaumeErhard/autoresearch-cifar10>

[7] L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, and H. Adam, "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation," *ECCV*, 2018.
