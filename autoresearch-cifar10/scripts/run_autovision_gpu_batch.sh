#!/usr/bin/env bash
set -euo pipefail
cd /jet/home/ytan8/Code/test2/autoresearch-cifar10
export AUTOVISION_PYTHON=/ocean/projects/cis250278p/ytan8/envs/autoresearch-cifar10/bin/python
export AUTOVISION_DATA_DIR=/ocean/projects/cis250278p/ytan8/datasets/autovision
export TORCH_HOME=/ocean/projects/cis250278p/ytan8/torch_cache
export XDG_CACHE_HOME=/ocean/projects/cis250278p/ytan8/cache
mkdir -p /ocean/projects/cis250278p/ytan8/datasets/autovision /ocean/projects/cis250278p/ytan8/torch_cache /ocean/projects/cis250278p/ytan8/cache autoresearch_jobs
$AUTOVISION_PYTHON scripts/run_autovision_gpu_batch.py
