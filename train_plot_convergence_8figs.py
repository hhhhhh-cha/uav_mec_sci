from __future__ import annotations

import argparse
import glob
import os
import re
import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METRICS = {
    "episode_reward": {
        "ylabel": "Evaluation Reward",
        "title_single": "Convergence of evaluation reward under seed = {seed}",
        "title_multi": "Evaluation reward convergence over five random seeds",
        "filename_single": "fig1_seed{seed}_reward.png",
        "filename_multi": "fig5_multi_reward_mean_std.png",
    },
    "avg_delay": {
        "ylabel": "Average Delay",
        "title_single": "Convergence of average delay under seed = {seed}",
        "title_multi": "Average delay convergence over five random seeds",
        "filename_single": "fig2_seed{seed}_delay.png",
        "filename_multi": "fig6_multi_delay_mean_std.png",
    },
    "feasible_ratio": {
        "ylabel": "Feasible Ratio",
        "title_single": "Convergence of feasible ratio under seed = {seed}",
        "title_multi": "Feasible ratio convergence over five random seeds",
        "filename_single": "fig3_seed{seed}_feasible_ratio.png",
        "filename_multi": "fig7_multi_feasible_ratio_mean_std.png",
    },
    "avg_deadline_violation": {
        "ylabel": "Average Deadline Violation",
        "title_single": "Convergence of average deadline violation under seed = {seed}",
        "title_multi": "Average deadline violation convergence over five random seeds",
        "filename_single": "fig4_seed{seed}_deadline_violation.png",
        "filename_multi": "fig8_multi_deadline_violation_mean_std.png",
    },
}


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y.copy()
    return pd.Series(y).rolling(window=window, min_periods=1, center=False).mean().to_numpy()


def extract_zip_if_needed(input_path: Path, work_root: Path) -> Path:
    if input_path.is_dir():
        return input_path
    if input_path.suffix.lower() != ".zip":
        raise ValueError(f"Unsupported input: {input_path}. Please provide a folder or a .zip file.")
    extract_dir = work_root / (input_path.stem + "_extracted")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(input_path, "r") as zf:
        zf.extractall(extract_dir)
    return extract_dir


def find_eval_logs(root_dir: Path) -> List[Path]:
    logs = sorted(Path(p) for p in glob.glob(str(root_dir / "**" / "eval_log.csv"), recursive=True))
    if not logs:
        raise FileNotFoundError(f"No eval_log.csv found under {root_dir}")
    return logs


def parse_seed(path: Path) -> int:
    m = re.search(r"seed(\d+)", str(path.parent))
    if not m:
        raise ValueError(f"Cannot parse seed from path: {path}")
    return int(m.group(1))


def load_eval_data(root_dir: Path) -> Dict[int, pd.DataFrame]:
    data: Dict[int, pd.DataFrame] = {}
    for log_path in find_eval_logs(root_dir):
        seed = parse_seed(log_path)
        df = pd.read_csv(log_path)
        required = ["eval_episode", *METRICS.keys()]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"{log_path} missing columns: {missing}")
        data[seed] = df.sort_values("eval_episode").reset_index(drop=True)
    return dict(sorted(data.items(), key=lambda x: x[0]))


def save_single_seed_plot(df: pd.DataFrame, metric: str, seed: int, out_dir: Path, smooth_window: int) -> None:
    cfg = METRICS[metric]
    x = df["eval_episode"].to_numpy()
    y = df[metric].to_numpy(dtype=float)
    y_ma = moving_average(y, smooth_window)

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(x, y, linewidth=1.0, alpha=0.35, label="Raw")
    plt.plot(x, y_ma, linewidth=2.2, label=f"Moving average (w={smooth_window})")
    plt.xlabel("Evaluation Episode")
    plt.ylabel(cfg["ylabel"])
    plt.title(cfg["title_single"].format(seed=seed))
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / cfg["filename_single"].format(seed=seed), dpi=300, bbox_inches="tight")
    plt.close()


def align_metric_frames(data: Dict[int, pd.DataFrame], metric: str) -> pd.DataFrame:
    merged = None
    for seed, df in data.items():
        tmp = df[["eval_episode", metric]].copy().rename(columns={metric: f"seed_{seed}"})
        merged = tmp if merged is None else pd.merge(merged, tmp, on="eval_episode", how="inner")
    if merged is None:
        raise ValueError("No data available.")
    return merged.sort_values("eval_episode").reset_index(drop=True)


def save_multi_seed_plot(data: Dict[int, pd.DataFrame], metric: str, out_dir: Path, smooth_window: int) -> None:
    cfg = METRICS[metric]
    merged = align_metric_frames(data, metric)
    x = merged["eval_episode"].to_numpy()
    values = merged.drop(columns=["eval_episode"]).to_numpy(dtype=float)
    mean = values.mean(axis=1)
    std = values.std(axis=1)
    mean_ma = moving_average(mean, smooth_window)
    upper_ma = moving_average(mean + std, smooth_window)
    lower_ma = moving_average(mean - std, smooth_window)

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(x, mean, linewidth=1.0, alpha=0.25, label="Mean (raw)")
    plt.plot(x, mean_ma, linewidth=2.2, label=f"Mean (moving average, w={smooth_window})")
    plt.fill_between(x, lower_ma, upper_ma, alpha=0.20, label="±1 std")
    plt.xlabel("Evaluation Episode")
    plt.ylabel(cfg["ylabel"])
    plt.title(cfg["title_multi"])
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / cfg["filename_multi"], dpi=300, bbox_inches="tight")
    plt.close()


def save_summary_csv(data: Dict[int, pd.DataFrame], out_dir: Path, tail_ratio: float = 0.2) -> None:
    rows = []
    for seed, df in data.items():
        n_tail = max(1, int(len(df) * tail_ratio))
        tail = df.tail(n_tail)
        row = {"seed": seed, "tail_points_used": n_tail}
        for metric in METRICS.keys():
            row[f"{metric}_tail_mean"] = tail[metric].mean()
            row[f"{metric}_tail_std"] = tail[metric].std(ddof=0)
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("seed")
    summary.to_csv(out_dir / "convergence_tail_summary.csv", index=False)


def save_combined_overview(data: Dict[int, pd.DataFrame], rep_seed: int, out_dir: Path, smooth_window: int) -> None:
    rep_df = data[rep_seed]
    metrics_order = ["episode_reward", "avg_delay", "feasible_ratio", "avg_deadline_violation"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    for ax, metric in zip(axes, metrics_order):
        cfg = METRICS[metric]
        x = rep_df["eval_episode"].to_numpy()
        y = rep_df[metric].to_numpy(dtype=float)
        y_ma = moving_average(y, smooth_window)
        ax.plot(x, y, linewidth=1.0, alpha=0.35)
        ax.plot(x, y_ma, linewidth=2.0)
        ax.set_title(cfg["title_single"].format(seed=rep_seed), fontsize=10)
        ax.set_xlabel("Evaluation Episode")
        ax.set_ylabel(cfg["ylabel"])
        ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / f"combined_single_seed{rep_seed}_4panel.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 8 convergence figures from eval_log.csv files.")
    parser.add_argument("--input", required=True, help="Path to the zip file or extracted folder.")
    parser.add_argument("--output", required=True, help="Directory to save figures.")
    parser.add_argument("--rep-seed", type=int, default=72, help="Representative seed for the single-seed plots.")
    parser.add_argument("--smooth-window", type=int, default=5, help="Moving-average window size.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    work_root = out_dir / "_work"
    work_root.mkdir(parents=True, exist_ok=True)
    root_dir = extract_zip_if_needed(input_path, work_root)
    data = load_eval_data(root_dir)

    if args.rep_seed not in data:
        available = sorted(data.keys())
        raise ValueError(f"Representative seed {args.rep_seed} not found. Available seeds: {available}")

    for metric in ["episode_reward", "avg_delay", "feasible_ratio", "avg_deadline_violation"]:
        save_single_seed_plot(data[args.rep_seed], metric, args.rep_seed, out_dir, args.smooth_window)
        save_multi_seed_plot(data, metric, out_dir, args.smooth_window)

    save_summary_csv(data, out_dir)
    save_combined_overview(data, args.rep_seed, out_dir, args.smooth_window)

    print(f"Done. Figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
