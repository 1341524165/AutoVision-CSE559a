from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)


def plot_classification():
    path = ROOT / "results.tsv"
    if not path.exists():
        return
    df = pd.read_csv(path, sep="\t")
    if df.empty or "best_acc" not in df:
        return
    df["attempt"] = range(1, len(df) + 1)
    df["champion"] = df["best_acc"].cummax()
    colors = df["status"].map({"keep": "#1f77b4", "discard": "#9ca3af", "crash": "#d62728"}).fillna("#6b7280")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.scatter(df["attempt"], df["best_acc"], c=colors, s=36)
    ax.plot(df["attempt"], df["champion"], color="#111827", linewidth=2, label="champion")
    ax.set_title("CIFAR-10 AutoResearch Accuracy")
    ax.set_xlabel("Attempt")
    ax.set_ylabel("Best test accuracy (%)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "classification_autoresearch_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.scatter(df["memory_gb"], df["best_acc"], c=colors, s=42)
    ax.set_title("CIFAR-10 Accuracy vs Peak VRAM")
    ax.set_xlabel("Peak VRAM (GB)")
    ax.set_ylabel("Best test accuracy (%)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS / "classification_memory_vs_accuracy.png", dpi=180)
    plt.close(fig)


def plot_segmentation():
    path = ROOT / "segmentation" / "results.tsv"
    if not path.exists():
        return
    df = pd.read_csv(path, sep="\t")
    if df.empty or "miou" not in df:
        return
    df["attempt"] = range(1, len(df) + 1)
    df["champion_miou"] = df["miou"].cummax()
    colors = df["status"].map({"keep": "#16a34a", "discard": "#9ca3af", "crash": "#d62728"}).fillna("#6b7280")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.scatter(df["attempt"], df["miou"], c=colors, s=36)
    ax.plot(df["attempt"], df["champion_miou"], color="#111827", linewidth=2, label="champion mIoU")
    if "dice" in df:
        ax.plot(df["attempt"], df["dice"], color="#f59e0b", linewidth=1.8, label="Dice")
    ax.set_title("Oxford-IIIT Pet Segmentation AutoResearch")
    ax.set_xlabel("Attempt")
    ax.set_ylabel("Metric")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "segmentation_autoresearch_curve.png", dpi=180)
    plt.close(fig)


def plot_cross_task():
    cls_path = ROOT / "results.tsv"
    seg_path = ROOT / "segmentation" / "results.tsv"
    if not cls_path.exists() or not seg_path.exists():
        return
    cls = pd.read_csv(cls_path, sep="\t")
    seg = pd.read_csv(seg_path, sep="\t")
    if cls.empty or seg.empty:
        return
    labels = ["Classification kept", "Classification tried", "Segmentation kept", "Segmentation tried"]
    values = [int((cls["status"] == "keep").sum()), len(cls), int((seg["status"] == "keep").sum()), len(seg)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color=["#2563eb", "#93c5fd", "#16a34a", "#86efac"])
    ax.set_title("AutoResearch Attempt Counts by Task")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(PLOTS / "cross_task_attempt_counts.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    plot_classification()
    plot_segmentation()
    plot_cross_task()
