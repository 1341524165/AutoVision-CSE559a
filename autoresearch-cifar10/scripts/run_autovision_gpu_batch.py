from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("AUTOVISION_PYTHON", "/ocean/projects/cis250278p/ytan8/envs/autoresearch-cifar10/bin/python")
ENV = os.environ.copy()
ENV.setdefault("AUTOVISION_DATA_DIR", "/ocean/projects/cis250278p/ytan8/datasets/autovision")
ENV.setdefault("TORCH_HOME", "/ocean/projects/cis250278p/ytan8/torch_cache")
ENV.setdefault("XDG_CACHE_HOME", "/ocean/projects/cis250278p/ytan8/cache")


def sh(cmd: list[str], cwd: Path = ROOT, timeout: int | None = None, check: bool = True):
    return subprocess.run(cmd, cwd=cwd, env=ENV, text=True, capture_output=True, timeout=timeout, check=check)


def git(*args: str, check: bool = True):
    return sh(["git", *args], check=check)


def read_best_acc() -> float:
    path = ROOT / "results.tsv"
    if not path.exists():
        return 0.0
    best = 0.0
    for line in path.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 4 and parts[3] == "keep":
            best = max(best, float(parts[1]))
    return best


def read_best_miou() -> float:
    path = ROOT / "segmentation" / "results.tsv"
    if not path.exists():
        return 0.0
    best = 0.0
    for line in path.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 5 and parts[4] == "keep":
            best = max(best, float(parts[1]))
    return best


def already_logged(path: Path, description: str) -> bool:
    return path.exists() and description in path.read_text()


def parse_classification_log(path: Path):
    text = path.read_text(errors="replace") if path.exists() else ""
    acc = re.search(r"^best_test_acc:\s*([0-9.]+)%", text, re.M)
    mem = re.search(r"^peak_vram_mb:\s*([0-9.]+)", text, re.M)
    return (float(acc.group(1)) if acc else None, float(mem.group(1)) / 1024 if mem else 0.0)


def parse_segmentation_log(path: Path):
    text = path.read_text(errors="replace") if path.exists() else ""
    miou = re.search(r"^best_miou:\s*([0-9.]+)", text, re.M)
    dice = re.search(r"^best_dice:\s*([0-9.]+)", text, re.M)
    mem = re.search(r"^peak_vram_mb:\s*([0-9.]+)", text, re.M)
    return (
        float(miou.group(1)) if miou else None,
        float(dice.group(1)) if dice else 0.0,
        float(mem.group(1)) / 1024 if mem else 0.0,
    )


def replace_many(path: Path, replacements: list[tuple[str, str]]):
    text = path.read_text()
    for old, new in replacements:
        if old not in text:
            raise RuntimeError(f"pattern not found: {old}")
        text = text.replace(old, new, 1)
    path.write_text(text)


def append(path: Path, row: str):
    with path.open("a") as f:
        f.write(row + "\n")


def download_datasets():
    code = """
from torchvision import datasets
from pathlib import Path
import os
Path('./data').mkdir(exist_ok=True)
datasets.CIFAR10('./data', train=True, download=True)
datasets.CIFAR10('./data', train=False, download=True)
root=os.environ.get('AUTOVISION_DATA_DIR')
Path(root).mkdir(parents=True, exist_ok=True)
datasets.OxfordIIITPet(root, split='trainval', target_types='segmentation', download=True)
datasets.OxfordIIITPet(root, split='test', target_types='segmentation', download=True)
print('datasets ready')
"""
    print(sh([PYTHON, "-c", code], timeout=1800).stdout)


def run_classification_experiments(max_new: int = 12):
    experiments = [
        ("OneCycle total_steps=4500", [("total_steps=MAX_STEPS", "total_steps=4500")]),
        ("OneCycle max_lr=0.45", [("max_lr=0.5", "max_lr=0.45")]),
        ("OneCycle max_lr=0.52", [("max_lr=0.5", "max_lr=0.52")]),
        ("OneCycle pct_start=0.35", [("pct_start=0.3", "pct_start=0.35")]),
        ("label smoothing 0.015", [("LABEL_SMOOTHING = 0.02", "LABEL_SMOOTHING = 0.015")]),
        ("label smoothing 0.025 total_steps=4500", [("LABEL_SMOOTHING = 0.02", "LABEL_SMOOTHING = 0.025"), ("total_steps=MAX_STEPS", "total_steps=4500")]),
        ("weight decay 7e-5", [("WEIGHT_DECAY = 1e-4", "WEIGHT_DECAY = 7e-5")]),
        ("weight decay 1.5e-4", [("WEIGHT_DECAY = 1e-4", "WEIGHT_DECAY = 1.5e-4")]),
        ("batch size 160", [("BATCH_SIZE = 128", "BATCH_SIZE = 160")]),
        ("OneCycle max_lr=0.48 pct_start=0.35", [("max_lr=0.5", "max_lr=0.48"), ("pct_start=0.3", "pct_start=0.35")]),
        ("label smoothing 0.01 max_lr=0.45", [("LABEL_SMOOTHING = 0.02", "LABEL_SMOOTHING = 0.01"), ("max_lr=0.5", "max_lr=0.45")]),
        ("OneCycle total_steps=4600", [("total_steps=MAX_STEPS", "total_steps=4600")]),
    ]
    results = ROOT / "results.tsv"
    ran = 0
    for desc, replacements in experiments:
        if ran >= max_new or already_logged(results, desc):
            continue
        base = git("rev-parse", "--short", "HEAD").stdout.strip()
        replace_many(ROOT / "train.py", replacements)
        git("add", "train.py")
        git("commit", "-m", desc)
        commit = git("rev-parse", "--short", "HEAD").stdout.strip()
        log = ROOT / "run.log"
        with log.open("w") as f:
            proc = subprocess.run([PYTHON, "train.py"], cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT, timeout=700)
        acc, mem = parse_classification_log(log)
        if proc.returncode != 0 or acc is None:
            append(results, f"{commit}\t0.00\t0.0\tcrash\t{desc}")
            git("reset", "--hard", base)
        else:
            best_before = read_best_acc()
            status = "keep" if acc > best_before else "discard"
            append(results, f"{commit}\t{acc:.2f}\t{mem:.1f}\t{status}\t{desc}")
            if status != "keep":
                git("reset", "--hard", base)
        ran += 1


def run_segmentation_experiments(max_new: int = 4):
    results = ROOT / "segmentation" / "results.tsv"
    results.parent.mkdir(exist_ok=True)
    if not results.exists():
        results.write_text("commit\tmiou\tdice\tmemory_gb\tstatus\tdescription\n")
    experiments = [
        ("baseline ResNet34 UNet 128px", []),
        ("batch size 12", [("BATCH_SIZE = 16", "BATCH_SIZE = 12")]),
        ("decoder channels 96", [("DECODER_CHANNELS = 128", "DECODER_CHANNELS = 96")]),
        ("lr 5e-4", [("LR = 3e-4", "LR = 5e-4")]),
        ("unfreeze encoder from start", [("FREEZE_ENCODER_EPOCHS = 1", "FREEZE_ENCODER_EPOCHS = 0")]),
    ]
    ran = 0
    for desc, replacements in experiments:
        if ran >= max_new or already_logged(results, desc):
            continue
        base = git("rev-parse", "--short", "HEAD").stdout.strip()
        if replacements:
            replace_many(ROOT / "segmentation" / "train.py", replacements)
            git("add", "segmentation/train.py")
            git("commit", "-m", f"Segmentation: {desc}")
        commit = git("rev-parse", "--short", "HEAD").stdout.strip()
        log = ROOT / "segmentation" / "run.log"
        with log.open("w") as f:
            proc = subprocess.run([PYTHON, "train.py"], cwd=ROOT / "segmentation", env=ENV, stdout=f, stderr=subprocess.STDOUT, timeout=1200)
        miou, dice, mem = parse_segmentation_log(log)
        if proc.returncode != 0 or miou is None:
            append(results, f"{commit}\t0.0000\t0.0000\t0.0\tcrash\t{desc}")
            if replacements:
                git("reset", "--hard", base)
        else:
            best_before = read_best_miou()
            status = "keep" if miou > best_before else "discard"
            append(results, f"{commit}\t{miou:.4f}\t{dice:.4f}\t{mem:.1f}\t{status}\t{desc}")
            if replacements and status != "keep":
                git("reset", "--hard", base)
        ran += 1


def main():
    os.chdir(ROOT)
    download_datasets()
    run_classification_experiments()
    run_segmentation_experiments()
    sh([PYTHON, "scripts/plot_autovision_results.py"], timeout=300)


if __name__ == "__main__":
    main()
