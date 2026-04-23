#!/usr/bin/env bash
set -u
cd /jet/home/ytan8/Code/test2/autoresearch-cifar10
mkdir -p autoresearch_jobs
while true; do
  if squeue -h -u ytan8 -p GPU-shared,GPU-share 2>/dev/null | grep -q .; then
    echo "[$(date +%Y%m%d-%H%M%S)] GPU job already active; waiting" >> autoresearch_jobs/salloc-supervisor.log
    sleep 300
    continue
  fi
  ts=$(date +%Y%m%d-%H%M%S)
  echo "[$ts] requesting GPU allocation" >> autoresearch_jobs/salloc-supervisor.log
  salloc -A cis250278p -p GPU-shared --gres=gpu:1 -t 02:00:00 srun --ntasks=1 bash scripts/run_autovision_gpu_batch.sh >> "autoresearch_jobs/salloc-${ts}.log" 2>&1
  rc=$?
  echo "[$(date +%Y%m%d-%H%M%S)] allocation ended rc=$rc" >> autoresearch_jobs/salloc-supervisor.log
  sleep 60
done
