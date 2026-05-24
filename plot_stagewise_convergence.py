"""
Plot stage-wise convergence for the proposed two-stage training pipeline.

Usage example:
  python3 plot_stagewise_convergence.py \
    --stage1 results/convergence_training/proposed_full_stage1_damaged_w50_seed72 \
    --stage2 results/convergence_training/proposed_full_stage2_damaged_w50_seed72_warm20_margin \
    --out results/convergence_outputs/stagewise_seed72

This script expects each stage directory to contain eval_log.csv. It also uses
train_log.csv when present for loss/diagnostic plots.
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def _read_eval(path: Path, stage: str) -> pd.DataFrame:
    csv_path = path / "eval_log.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing eval_log.csv: {csv_path}")
    df = pd.read_csv(csv_path)
    if "eval_episode" not in df.columns:
        raise ValueError(f"{csv_path} has no eval_episode column")
    if "system_cost" not in df.columns and "episode_reward" in df.columns:
        df["system_cost"] = -df["episode_reward"]
    df["stage"] = stage
    df["stage_episode"] = df["eval_episode"].astype(float)
    return df

def _read_train(path: Path, stage: str) -> Optional[pd.DataFrame]:
# def _read_train(path: Path, stage: str) -> pd.DataFrame | None:
    csv_path = path / "train_log.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if "episode" not in df.columns:
        return None
    if "system_cost" not in df.columns and "episode_reward" in df.columns:
        df["system_cost"] = -df["episode_reward"]
    df["stage"] = stage
    df["stage_episode"] = df["episode"].astype(float)
    return df


def _smooth(y: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return y
    return y.rolling(window=window, min_periods=1, center=True).mean()

def _concat_stagewise(stage1: pd.DataFrame, stage2: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
# def _concat_stagewise(stage1: pd.DataFrame, stage2: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    transition = float(stage1["stage_episode"].max()) + 1.0
    s1 = stage1.copy()
    s2 = stage2.copy()
    s1["global_episode"] = s1["stage_episode"]
    s2["global_episode"] = transition + s2["stage_episode"]
    out = pd.concat([s1, s2], ignore_index=True)
    return out, transition


def _plot_metric(df: pd.DataFrame, out_dir: Path, metric: str, ylabel: str, transition: float, smooth_win: int):
    if metric not in df.columns:
        print(f"[SKIP] metric not found: {metric}")
        return
    work = df[["global_episode", metric, "stage"]].dropna().copy()
    if work.empty:
        print(f"[SKIP] metric has no numeric data: {metric}")
        return
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna()
    work["smooth"] = _smooth(work[metric], smooth_win)

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(work["global_episode"], work[metric], alpha=0.35, linewidth=1.0, label="Raw")
    ax.plot(work["global_episode"], work["smooth"], linewidth=2.0, label=f"MA{smooth_win}")
    ax.axvline(transition, linestyle="--", linewidth=1.2, label="Stage transition")
    ax.set_xlabel("Training episode")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / f"stagewise_{metric}.png", dpi=300)
    plt.close(fig)


def _plot_overview(df: pd.DataFrame, out_dir: Path, transition: float, smooth_win: int):
    metrics = [
        ("system_cost", "System cost"),
        ("avg_delay", "Average delay"),
        ("avg_deadline_violation", "Deadline violation"),
        ("feasible_ratio", "Feasible ratio"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.6))
    axes = axes.reshape(-1)
    for ax, (metric, ylabel) in zip(axes, metrics):
        if metric not in df.columns:
            ax.axis("off")
            continue
        work = df[["global_episode", metric]].dropna().copy()
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
        work = work.dropna()
        work["smooth"] = _smooth(work[metric], smooth_win)
        ax.plot(work["global_episode"], work[metric], alpha=0.35, linewidth=1.0)
        ax.plot(work["global_episode"], work["smooth"], linewidth=2.0)
        ax.axvline(transition, linestyle="--", linewidth=1.1)
        ax.set_xlabel("Training episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out_dir / "stagewise_overview_2x2.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", required=True, help="Stage-1 result directory containing eval_log.csv")
    parser.add_argument("--stage2", required=True, help="Stage-2 result directory containing eval_log.csv")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--smooth", type=int, default=5, help="Moving average window")
    args = parser.parse_args()

    stage1_dir = Path(args.stage1)
    stage2_dir = Path(args.stage2)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval1 = _read_eval(stage1_dir, "Stage-1 warm-start")
    eval2 = _read_eval(stage2_dir, "Stage-2 CTDE fine-tuning")
    eval_all, transition = _concat_stagewise(eval1, eval2)
    eval_all.to_csv(out_dir / "stagewise_eval_merged.csv", index=False)

    metric_specs = [
        ("system_cost", "System cost (-evaluation reward)"),
        ("episode_reward", "Evaluation reward"),
        ("avg_delay", "Average delay"),
        ("avg_energy", "Average UAV-side energy"),
        ("avg_deadline_violation", "Deadline violation"),
        ("feasible_ratio", "Feasible ratio"),
        ("ratio_mean", "Mean offloading ratio"),
        ("neighbor_exec_ratio", "Neighbor execution ratio"),
    ]
    for metric, ylabel in metric_specs:
        _plot_metric(eval_all, out_dir, metric, ylabel, transition, args.smooth)
    _plot_overview(eval_all, out_dir, transition, args.smooth)

    train1 = _read_train(stage1_dir, "Stage-1 warm-start")
    train2 = _read_train(stage2_dir, "Stage-2 CTDE fine-tuning")
    if train1 is not None and train2 is not None:
        train_all, train_transition = _concat_stagewise(train1, train2)
        train_all.to_csv(out_dir / "stagewise_train_merged.csv", index=False)
        for metric, ylabel in [
            ("avg_ratio_loss", "Stage-1 ratio BC loss"),
            ("avg_ratio_reg_loss", "Stage-2 ratio regularization loss"),
            ("avg_critic_loss", "Critic loss"),
            ("avg_td_abs_error", "TD absolute error"),
        ]:
            _plot_metric(train_all, out_dir, metric, ylabel, train_transition, args.smooth)

    print(f"Saved stage-wise convergence outputs to: {out_dir}")


if __name__ == "__main__":
    main()
