# -*- coding: utf-8 -*-
"""
Plot SCI-style convergence curves for multi-seed UAV MEC experiments.

Expected directory structure:
results/convergence_training/
    500_proposed_full_stage2_seed42_run1/
        eval_log.csv
        train_log.csv
    500_proposed_full_stage2_seed52_run1/
        eval_log.csv
        train_log.csv
    ...

Main outputs:
    1) single-seed evaluation convergence curves
    2) multi-seed mean ± 95% CI evaluation convergence curves

Author: customized for UAV MEC SCI paper figures
"""

from __future__ import annotations

import os
import re
import glob
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator, FormatStrFormatter


# ============================================================
# 1. User configuration
# ============================================================

@dataclass
class PlotConfig:
    # Your experiment root directory
    root_dir: str = "results/convergence_training"

    # Figure output directory
    out_dir: str = "results/convergence_outputs_sci"

    # Main single seed to plot separately
    single_seed: int = 72

    # If some seed is incomplete, put it here, e.g. {92}
    # If all ten seeds are complete, keep it empty.
    exclude_seeds: Tuple[int, ...] = ()

    # Smoothing window for evaluation curves.
    # eval_log has 101 points for 500 episodes, so 5 or 7 is usually reasonable.
    smooth_window_single: int = 5
    smooth_window_multi: int = 5

    # Multi-seed uncertainty band:
    # "ci95" = mean ± 95% confidence interval, recommended for paper
    # "std"  = mean ± standard deviation
    band_mode: str = "ci95"

    # Save formats
    save_png: bool = True
    save_pdf: bool = True
    save_svg: bool = True

    # Figure size
    fig_width: float = 4.8
    fig_height: float = 3.4

    # DPI for png
    dpi: int = 600

    # Academic font style
    # font_family: str = "Times New Roman"

    # Whether to print detailed loading info
    verbose: bool = True


CFG = PlotConfig()


# ============================================================
# 2. Metric settings
# ============================================================

# metric column, y-axis label, filename key, tick interval
EVAL_METRICS = [
    ("episode_reward", "Evaluation Reward", "reward", 5000),
    ("avg_delay", "Average Delay", "delay", 5),
    ("avg_energy", "Average Energy", "energy", None),
    ("feasible_ratio", "Feasibility Ratio", "feasible_ratio", 0.05),
]

# If you also want training diagnostic curves later, you can use these.
TRAIN_METRICS = [
    ("episode_reward", "Training Reward", "train_reward", None),
    ("avg_actor_policy_loss", "Actor Policy Loss", "actor_policy_loss", None),
    ("avg_critic_loss", "Critic Loss", "critic_loss", None),
    ("avg_move_sched_bc_loss", "Move/Scheduling BC Loss", "move_sched_bc_loss", None),
    ("avg_ratio_bc_loss", "Ratio BC Loss", "ratio_bc_loss", None),
]


# ============================================================
# 3. Matplotlib academic style
# ============================================================

def set_academic_style(cfg: PlotConfig) -> None:
    import logging

    # Suppress repeated matplotlib font warnings
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    plt.rcParams.update({
        "figure.dpi": cfg.dpi,
        "savefig.dpi": cfg.dpi,

        # Use Times New Roman if available; otherwise use close alternatives
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Liberation Serif",
            "Nimbus Roman",
            "DejaVu Serif",
        ],

        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10,

        "axes.linewidth": 1.0,
        "lines.linewidth": 2.0,

        # Important for LaTeX / PDF paper submission
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",

        "axes.unicode_minus": False,
    })

# ============================================================
# 4. Loading utilities
# ============================================================

def extract_seed_from_path(path: str) -> Optional[int]:
    """
    Extract seed from folder name such as:
    500_proposed_full_stage2_seed72_run1
    """
    m = re.search(r"seed(\d+)", path)
    if m is None:
        return None
    return int(m.group(1))


def find_seed_dirs(root_dir: str) -> Dict[int, str]:
    """
    Find all seed folders under root_dir.
    """
    pattern = os.path.join(root_dir, "*seed*_run*")
    dirs = sorted(glob.glob(pattern))

    seed_to_dir: Dict[int, str] = {}
    for d in dirs:
        if not os.path.isdir(d):
            continue
        seed = extract_seed_from_path(os.path.basename(d))
        if seed is not None:
            seed_to_dir[seed] = d

    return seed_to_dir


def load_eval_logs(cfg: PlotConfig) -> Dict[int, pd.DataFrame]:
    """
    Load eval_log.csv from each seed folder.
    """
    seed_to_dir = find_seed_dirs(cfg.root_dir)
    exclude = set(cfg.exclude_seeds)

    logs: Dict[int, pd.DataFrame] = {}

    for seed, d in sorted(seed_to_dir.items()):
        if seed in exclude:
            if cfg.verbose:
                print(f"[SKIP] Seed {seed} is excluded.")
            continue

        csv_path = os.path.join(d, "eval_log.csv")
        if not os.path.exists(csv_path):
            if cfg.verbose:
                print(f"[WARN] Seed {seed}: eval_log.csv not found.")
            continue

        df = pd.read_csv(csv_path)

        if "eval_episode" not in df.columns:
            raise ValueError(f"Seed {seed}: eval_log.csv must contain column 'eval_episode'.")

        missing = [m[0] for m in EVAL_METRICS if m[0] not in df.columns]
        if missing:
            raise ValueError(f"Seed {seed}: eval_log.csv missing columns: {missing}")

        df = df.sort_values("eval_episode").reset_index(drop=True)
        logs[seed] = df

        if cfg.verbose:
            ep_min = df["eval_episode"].min()
            ep_max = df["eval_episode"].max()
            print(f"[LOAD] Seed {seed}: {len(df)} eval rows, episode {ep_min} -> {ep_max}")

    if not logs:
        raise RuntimeError(f"No valid eval_log.csv found under: {cfg.root_dir}")

    return logs


def load_train_logs(cfg: PlotConfig) -> Dict[int, pd.DataFrame]:
    """
    Optional: load train_log.csv from each seed folder.
    """
    seed_to_dir = find_seed_dirs(cfg.root_dir)
    exclude = set(cfg.exclude_seeds)

    logs: Dict[int, pd.DataFrame] = {}

    for seed, d in sorted(seed_to_dir.items()):
        if seed in exclude:
            continue

        csv_path = os.path.join(d, "train_log.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)

        if "episode" not in df.columns:
            raise ValueError(f"Seed {seed}: train_log.csv must contain column 'episode'.")

        df = df.sort_values("episode").reset_index(drop=True)
        logs[seed] = df

    return logs


# ============================================================
# 5. Data processing utilities
# ============================================================

def smooth_series(y: np.ndarray, window: int) -> np.ndarray:
    """
    Centered moving average smoothing.
    """
    if window is None or window <= 1:
        return y.astype(float)

    s = pd.Series(y.astype(float))
    return s.rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def build_common_eval_table(
    logs: Dict[int, pd.DataFrame],
    metric: str,
    smooth_window: int,
) -> pd.DataFrame:
    """
    Build a table indexed by eval_episode and columns are seeds.
    If different seeds have slightly different eval_episode points,
    this function aligns them and interpolates missing values.
    """
    series_list = []

    for seed, df in sorted(logs.items()):
        x = df["eval_episode"].to_numpy()
        y = df[metric].to_numpy(dtype=float)
        y = smooth_series(y, smooth_window)

        s = pd.Series(data=y, index=x, name=seed)
        series_list.append(s)

    table = pd.concat(series_list, axis=1).sort_index()

    # Interpolate if some seeds are missing values at some eval_episode
    table = table.interpolate(method="index", limit_direction="both")

    return table


def calc_band(values: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    values shape: [num_points, num_seeds]
    Return mean, lower, upper.
    """
    mean = np.nanmean(values, axis=1)
    std = np.nanstd(values, axis=1, ddof=1)
    n = np.sum(~np.isnan(values), axis=1)

    if mode.lower() == "std":
        half_width = std
    elif mode.lower() == "ci95":
        # For n around 10, 95% CI is approximately t * std / sqrt(n).
        # t≈2.262 for df=9. Use 1.96 when n is large.
        t_critical = np.where(n <= 10, 2.262, 1.96)
        half_width = t_critical * std / np.sqrt(np.maximum(n, 1))
    else:
        raise ValueError("band_mode must be 'ci95' or 'std'.")

    lower = mean - half_width
    upper = mean + half_width

    return mean, lower, upper


def apply_axis_format(
    ax: plt.Axes,
    metric_key: str,
    y_tick_step: Optional[float],
) -> None:
    """
    Axis formatting for publication-quality figures.
    """
    ax.set_xlabel("Episode")

    if y_tick_step is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_tick_step))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    if metric_key == "feasible_ratio":
        ax.set_ylim(-0.02, 1.02)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))

    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.20)
    ax.minorticks_on()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="both", direction="in", length=4, width=0.9)
    ax.tick_params(axis="both", which="minor", direction="in", length=2, width=0.7)


def save_figure(fig: plt.Figure, out_dir: str, filename: str, cfg: PlotConfig) -> None:
    os.makedirs(out_dir, exist_ok=True)

    if cfg.save_png:
        path = os.path.join(out_dir, filename + ".png")
        fig.savefig(path, bbox_inches="tight", dpi=cfg.dpi)
        print(f"[SAVE] {path}")

    if cfg.save_pdf:
        path = os.path.join(out_dir, filename + ".pdf")
        fig.savefig(path, bbox_inches="tight")
        print(f"[SAVE] {path}")

    if cfg.save_svg:
        path = os.path.join(out_dir, filename + ".svg")
        fig.savefig(path, bbox_inches="tight")
        print(f"[SAVE] {path}")


# ============================================================
# 6. Plot single-seed curves
# ============================================================

def plot_single_seed_eval(
    logs: Dict[int, pd.DataFrame],
    cfg: PlotConfig,
) -> None:
    if cfg.single_seed not in logs:
        available = sorted(logs.keys())
        raise ValueError(
            f"single_seed={cfg.single_seed} not found. Available seeds: {available}"
        )

    df = logs[cfg.single_seed]

    for metric, ylabel, key, y_tick_step in EVAL_METRICS:
        x = df["eval_episode"].to_numpy()
        y_raw = df[metric].to_numpy(dtype=float)
        y_smooth = smooth_series(y_raw, cfg.smooth_window_single)

        fig, ax = plt.subplots(figsize=(cfg.fig_width, cfg.fig_height))

        ax.plot(
            x,
            y_smooth,
            color="black",
            linewidth=2.2,
            label=f"Seed {cfg.single_seed}",
        )

        # raw evaluation points, light markers
        ax.scatter(
            x,
            y_raw,
            s=12,
            color="black",
            alpha=0.28,
            linewidths=0,
            label="Raw eval.",
        )

        ax.set_ylabel(ylabel)
        apply_axis_format(ax, key, y_tick_step)

        ax.legend(frameon=False, loc="best")

        filename = f"single_seed{cfg.single_seed}_{key}"
        save_figure(fig, cfg.out_dir, filename, cfg)
        plt.close(fig)


# ============================================================
# 7. Plot multi-seed mean ± band curves
# ============================================================

def plot_multi_seed_eval(
    logs: Dict[int, pd.DataFrame],
    cfg: PlotConfig,
) -> None:
    seeds = sorted(logs.keys())
    n_seed = len(seeds)

    for metric, ylabel, key, y_tick_step in EVAL_METRICS:
        table = build_common_eval_table(
            logs=logs,
            metric=metric,
            smooth_window=cfg.smooth_window_multi,
        )

        x = table.index.to_numpy(dtype=float)
        values = table.to_numpy(dtype=float)

        mean, lower, upper = calc_band(values, cfg.band_mode)

        fig, ax = plt.subplots(figsize=(cfg.fig_width, cfg.fig_height))

        # Individual seeds, light gray lines
        for seed in seeds:
            ax.plot(
                x,
                table[seed].to_numpy(dtype=float),
                color="0.72",
                linewidth=0.9,
                alpha=0.55,
            )

        # Mean curve
        ax.plot(
            x,
            mean,
            color="black",
            linewidth=2.4,
            label=f"Mean over {n_seed} seeds",
        )

        # Uncertainty band
        band_label = "95% CI" if cfg.band_mode.lower() == "ci95" else "Std."
        ax.fill_between(
            x,
            lower,
            upper,
            color="0.45",
            alpha=0.22,
            linewidth=0,
            label=band_label,
        )

        ax.set_ylabel(ylabel)
        apply_axis_format(ax, key, y_tick_step)

        ax.legend(frameon=False, loc="best")

        filename = f"multi_seed_{key}_{cfg.band_mode}"
        save_figure(fig, cfg.out_dir, filename, cfg)
        plt.close(fig)


# ============================================================
# 8. Optional training diagnostic curves
# ============================================================

def plot_multi_seed_train_diagnostics(
    train_logs: Dict[int, pd.DataFrame],
    cfg: PlotConfig,
    smooth_window: int = 15,
) -> None:
    """
    Optional diagnostic plots for training reward/loss.
    These are not necessarily required in the main paper, but useful for appendix.
    """
    if not train_logs:
        print("[WARN] No train logs found. Skip training diagnostics.")
        return

    for metric, ylabel, key, y_tick_step in TRAIN_METRICS:
        valid_logs = {
            seed: df for seed, df in train_logs.items()
            if metric in df.columns
        }
        if not valid_logs:
            continue

        series_list = []
        for seed, df in sorted(valid_logs.items()):
            x = df["episode"].to_numpy()
            y = df[metric].to_numpy(dtype=float)
            y = smooth_series(y, smooth_window)
            series_list.append(pd.Series(y, index=x, name=seed))

        table = pd.concat(series_list, axis=1).sort_index()
        table = table.interpolate(method="index", limit_direction="both")

        x = table.index.to_numpy(dtype=float)
        values = table.to_numpy(dtype=float)
        mean, lower, upper = calc_band(values, cfg.band_mode)

        fig, ax = plt.subplots(figsize=(cfg.fig_width, cfg.fig_height))

        for seed in sorted(valid_logs.keys()):
            ax.plot(
                x,
                table[seed].to_numpy(dtype=float),
                color="0.72",
                linewidth=0.8,
                alpha=0.45,
            )

        ax.plot(
            x,
            mean,
            color="black",
            linewidth=2.4,
            label=f"Mean over {len(valid_logs)} seeds",
        )

        band_label = "95% CI" if cfg.band_mode.lower() == "ci95" else "Std."
        ax.fill_between(
            x,
            lower,
            upper,
            color="0.45",
            alpha=0.22,
            linewidth=0,
            label=band_label,
        )

        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        apply_axis_format(ax, key, y_tick_step)

        ax.legend(frameon=False, loc="best")

        filename = f"train_multi_seed_{key}_{cfg.band_mode}"
        save_figure(fig, cfg.out_dir, filename, cfg)
        plt.close(fig)


# ============================================================
# 9. Main
# ============================================================

def main() -> None:
    set_academic_style(CFG)

    # Create timestamped output folder
    base_out_dir = CFG.out_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    CFG.out_dir = os.path.join(base_out_dir, timestamp)
    os.makedirs(CFG.out_dir, exist_ok=True)

    print("=" * 80)
    print("SCI-style convergence plotting")
    print("=" * 80)
    print(f"Root dir      : {CFG.root_dir}")
    print(f"Output dir    : {CFG.out_dir}")
    print(f"Single seed   : {CFG.single_seed}")
    print(f"Exclude seeds : {CFG.exclude_seeds}")
    print(f"Band mode     : {CFG.band_mode}")
    print("=" * 80)

    eval_logs = load_eval_logs(CFG)

    print("\n[INFO] Valid seeds:", sorted(eval_logs.keys()))
    print(f"[INFO] Number of seeds: {len(eval_logs)}")

    # Main paper figures: 8 evaluation convergence curves
    plot_single_seed_eval(eval_logs, CFG)
    plot_multi_seed_eval(eval_logs, CFG)

    # Optional appendix/diagnostic figures
    train_logs = load_train_logs(CFG)
    plot_multi_seed_train_diagnostics(train_logs, CFG, smooth_window=15)

    print("\n[DONE] All figures have been generated.")
    print(f"[DONE] Saved to: {CFG.out_dir}")


if __name__ == "__main__":
    main()