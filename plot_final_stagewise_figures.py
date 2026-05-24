#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot final stage-wise convergence figures for the proposed two-stage UAV-MEC training.

Recommended usage:
    python3 plot_final_stagewise_figures.py \
      --stage1 results/convergence_training/proposed_full_stage1_final_env_seed72_ep120 \
      --stage2 results/convergence_training/proposed_full_stage2_final_env_seed72_ep700 \
      --out results/convergence_outputs/final_env_seed72_figures \
      --seed-label 72

The script also accepts .zip inputs:
    python3 plot_final_stagewise_figures.py \
      --stage1 proposed_full_stage1_final_env_seed72_ep120.zip \
      --stage2 proposed_full_stage2_final_env_seed72_ep700.zip \
      --out final_env_seed72_figures \
      --seed-label 72
"""

import argparse
import json
import math
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter


# ============================================================
# Global plotting style
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 180,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# Utilities
# ============================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_zip_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".zip"


def extract_if_zip(input_path: Path, work_root: Path, tag: str) -> Path:
    """
    Return a directory containing train_log.csv / eval_log.csv.
    If input_path is a zip, extract it into work_root/tag.
    If input_path is already a directory, return it unchanged.
    """
    input_path = input_path.expanduser().resolve()

    if input_path.is_dir():
        return input_path

    if not is_zip_path(input_path):
        raise FileNotFoundError(f"Input path is neither a directory nor a .zip file: {input_path}")

    extract_dir = work_root / tag
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    ensure_dir(extract_dir)

    with zipfile.ZipFile(input_path, "r") as zf:
        zf.extractall(extract_dir)

    # Common case: zip contains one top-level directory.
    candidates = []
    for p in extract_dir.rglob("train_log.csv"):
        candidates.append(p.parent)

    if not candidates:
        raise FileNotFoundError(f"Cannot find train_log.csv after extracting {input_path}")

    # Prefer the deepest / most specific folder that also has eval_log.csv.
    candidates = sorted(candidates, key=lambda p: len(str(p)), reverse=True)
    for c in candidates:
        if (c / "eval_log.csv").exists():
            return c

    return candidates[0]


def read_csv_required(folder: Path, name: str) -> pd.DataFrame:
    path = folder / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    df = pd.read_csv(path)
    return df


def read_config_optional(folder: Path) -> Dict:
    path = folder / "config.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def rolling_mean(y: pd.Series, window: int) -> pd.Series:
    y = pd.to_numeric(y, errors="coerce")
    if window <= 1:
        return y
    return y.rolling(window=window, min_periods=max(1, window // 3)).mean()


def plot_raw_ma(
    ax,
    x,
    y,
    *,
    smooth: int,
    label_ma: str,
    color: Optional[str] = None,
    raw_alpha: float = 0.18,
    ma_lw: float = 1.6,
):
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    x = np.asarray(x, dtype=float)

    # Raw line: always light gray to indicate noisy observation.
    ax.plot(x, y.values, color="0.70", lw=0.8, alpha=raw_alpha, linestyle="-", label="Raw")

    y_ma = rolling_mean(y, smooth)
    ax.plot(x, y_ma.values, lw=ma_lw, color=color, label=label_ma)


def set_common_axis(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.45)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))


def finite_min_max(values) -> Tuple[Optional[float], Optional[float]]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None
    return float(np.min(arr)), float(np.max(arr))


def pad_ylim(ax, y, pad_ratio: float = 0.08, min_span: Optional[float] = None) -> None:
    lo, hi = finite_min_max(y)
    if lo is None:
        return
    span = hi - lo
    if min_span is not None:
        span = max(span, min_span)
    if span <= 1e-12:
        center = 0.5 * (lo + hi)
        span = max(abs(center) * 0.1, 1e-3)
        lo, hi = center - 0.5 * span, center + 0.5 * span
    pad = pad_ratio * span
    ax.set_ylim(lo - pad, hi + pad)


def draw_stage_boundary(ax, stage1_end: int) -> None:
    ax.axvline(stage1_end, color="0.25", linestyle="--", linewidth=0.9, alpha=0.8)


def save_figure(fig, out_dir: Path, basename: str) -> None:
    fig.savefig(out_dir / f"{basename}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{basename}.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Data assembly
# ============================================================
def load_stage_data(stage1_path: Path, stage2_path: Path, out_dir: Path):
    work_root = out_dir / "_extracted_inputs"
    ensure_dir(work_root)

    stage1_dir = extract_if_zip(stage1_path, work_root, "stage1")
    stage2_dir = extract_if_zip(stage2_path, work_root, "stage2")

    s1_train = read_csv_required(stage1_dir, "train_log.csv")
    s1_eval = read_csv_required(stage1_dir, "eval_log.csv")
    s2_train = read_csv_required(stage2_dir, "train_log.csv")
    s2_eval = read_csv_required(stage2_dir, "eval_log.csv")
    s1_config = read_config_optional(stage1_dir)
    s2_config = read_config_optional(stage2_dir)

    return stage1_dir, stage2_dir, s1_train, s1_eval, s2_train, s2_eval, s1_config, s2_config


def add_global_eval_episode(
    s1_eval: pd.DataFrame,
    s2_eval: pd.DataFrame,
    stage1_train_len: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    s1_eval = s1_eval.copy()
    s2_eval = s2_eval.copy()

    if "eval_episode" not in s1_eval.columns:
        s1_eval["eval_episode"] = np.arange(len(s1_eval))
    if "eval_episode" not in s2_eval.columns:
        s2_eval["eval_episode"] = np.arange(len(s2_eval))

    s1_eval["global_episode"] = pd.to_numeric(s1_eval["eval_episode"], errors="coerce")
    s2_eval["global_episode"] = stage1_train_len + pd.to_numeric(s2_eval["eval_episode"], errors="coerce")
    return s1_eval, s2_eval


def concat_eval(s1_eval: pd.DataFrame, s2_eval: pd.DataFrame) -> pd.DataFrame:
    s1 = s1_eval.copy()
    s2 = s2_eval.copy()
    s1["stage"] = "Stage-1"
    s2["stage"] = "Stage-2"
    return pd.concat([s1, s2], axis=0, ignore_index=True)


# ============================================================
# Figure 1: training losses and stability
# ============================================================
def plot_training_loss_stability(
    s1_train: pd.DataFrame,
    s2_train: pd.DataFrame,
    out_dir: Path,
    *,
    seed_label: str,
    stage1_smooth: int,
    stage2_smooth: int,
    log_critic: bool,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.8))
    axes = axes.reshape(-1)

    s1_x = numeric_series(s1_train, "episode")
    s2_x = numeric_series(s2_train, "episode")

    panels = [
        (
            axes[0],
            s1_x,
            numeric_series(s1_train, "avg_total_actor_loss"),
            stage1_smooth,
            f"MA{stage1_smooth}",
            "Episode",
            "Loss",
            "(a) Stage-1 warm-start total loss",
            "C1",
        ),
        (
            axes[1],
            s1_x,
            numeric_series(s1_train, "avg_move_sched_loss"),
            stage1_smooth,
            f"MA{stage1_smooth}",
            "Episode",
            "Loss",
            "(b) Stage-1 mobility/scheduling BC loss",
            "C2",
        ),
        (
            axes[2],
            s1_x,
            numeric_series(s1_train, "avg_ratio_loss"),
            stage1_smooth,
            f"MA{stage1_smooth}",
            "Episode",
            "Loss",
            "(c) Stage-1 offloading-ratio BC loss",
            "C3",
        ),
        (
            axes[3],
            s2_x,
            numeric_series(s2_train, "avg_critic_loss"),
            stage2_smooth,
            f"MA{stage2_smooth}",
            "Episode",
            "Critic loss",
            "(d) Stage-2 critic TD loss",
            "C0",
        ),
        (
            axes[4],
            s2_x,
            numeric_series(s2_train, "avg_td_abs_error"),
            stage2_smooth,
            f"MA{stage2_smooth}",
            "Episode",
            "TD absolute error",
            "(e) Stage-2 TD error",
            "C4",
        ),
        (
            axes[5],
            s2_x,
            numeric_series(s2_train, "avg_actor_policy_loss"),
            stage2_smooth,
            f"MA{stage2_smooth}",
            "Episode",
            "Policy objective loss",
            "(f) Stage-2 actor policy objective",
            "C5",
        ),
    ]

    for ax, x, y, smooth, ma_label, xlabel, ylabel, title, color in panels:
        plot_raw_ma(ax, x, y, smooth=smooth, label_ma=ma_label, color=color)
        set_common_axis(ax, xlabel, ylabel, title)
        if "critic" in title.lower() and log_critic:
            # Critic loss can span several scales in some runs.
            ax.set_yscale("log")
        else:
            pad_ylim(ax, y.dropna().values if isinstance(y, pd.Series) else y, pad_ratio=0.10)
        ax.legend(loc="best", frameon=False)

    fig.suptitle(
        f"Training-loss stability of the proposed two-stage framework (seed={seed_label})",
        y=1.02,
        fontsize=14,
    )
    fig.tight_layout()
    save_figure(fig, out_dir, "fig1_training_loss_stability_seed" + seed_label)


# ============================================================
# Figure 2: evaluation convergence
# ============================================================
def plot_evaluation_convergence(
    eval_df: pd.DataFrame,
    out_dir: Path,
    *,
    seed_label: str,
    eval_smooth: int,
    stage1_end: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.2))
    axes = axes.reshape(-1)

    x = numeric_series(eval_df, "global_episode")

    metric_specs = [
        (
            "system_cost",
            "Weighted system cost (×10$^3$)",
            "(a) Evaluation system cost",
            lambda s: pd.to_numeric(s, errors="coerce") / 1000.0,
            "C0",
        ),
        (
            "avg_delay",
            "Average delay (s)",
            "(b) Average task completion delay",
            lambda s: pd.to_numeric(s, errors="coerce"),
            "C1",
        ),
        (
            "avg_energy",
            "Average UAV energy (J/slot)",
            "(c) UAV-side energy consumption",
            lambda s: pd.to_numeric(s, errors="coerce"),
            "C2",
        ),
        (
            "feasible_ratio",
            "Feasible ratio",
            "(d) Feasible ratio",
            lambda s: pd.to_numeric(s, errors="coerce"),
            "C3",
        ),
    ]

    for ax, (col, ylabel, title, transform, color) in zip(axes, metric_specs):
        y = transform(eval_df[col]) if col in eval_df.columns else pd.Series(dtype=float)
        plot_raw_ma(ax, x, y, smooth=eval_smooth, label_ma=f"MA{eval_smooth}", color=color, raw_alpha=0.25)
        draw_stage_boundary(ax, stage1_end)
        set_common_axis(ax, "Global episode", ylabel, title)

        if col == "feasible_ratio":
            y_min, y_max = finite_min_max(y)
            if y_min is not None:
                low = max(0.0, min(0.94, y_min - 0.02))
                high = min(1.02, max(1.01, y_max + 0.01))
                ax.set_ylim(low, high)
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        else:
            if col == "avg_energy":
                pad_ylim(ax, y.dropna().values, pad_ratio=0.25, min_span=0.08)
            else:
                pad_ylim(ax, y.dropna().values, pad_ratio=0.10)
        ax.legend(loc="best", frameon=False)

    fig.suptitle(
        f"Evaluation convergence of the proposed two-stage framework (seed={seed_label})",
        y=1.02,
        fontsize=14,
    )
    fig.text(
        0.5,
        0.005,
        "The vertical dashed line denotes the transition from Stage-1 warm-start to Stage-2 CTDE fine-tuning.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.025, 1, 1])
    save_figure(fig, out_dir, "fig2_evaluation_convergence_seed" + seed_label)


# ============================================================
# Figure 3: auxiliary diagnostics
# ============================================================
def plot_auxiliary_diagnostics(
    eval_df: pd.DataFrame,
    out_dir: Path,
    *,
    seed_label: str,
    eval_smooth: int,
    stage1_end: int,
) -> None:
    available_cols = set(eval_df.columns)
    specs = [
        ("ratio_mean", "Mean offloading ratio", "(a) Offloading-ratio mean", "C0"),
        ("ratio_std", "Std. of offloading ratio", "(b) Offloading-ratio diversity", "C1"),
        ("local_exec_ratio", "Local execution ratio", "(c) Local execution ratio", "C2"),
        ("avg_battery_violation", "Battery diagnostic", "(d) Battery diagnostic violation", "C3"),
    ]
    specs = [s for s in specs if s[0] in available_cols]
    if not specs:
        return

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.2))
    axes = axes.reshape(-1)
    x = numeric_series(eval_df, "global_episode")

    for i, ax in enumerate(axes):
        if i >= len(specs):
            ax.axis("off")
            continue

        col, ylabel, title, color = specs[i]
        y = numeric_series(eval_df, col)
        plot_raw_ma(ax, x, y, smooth=eval_smooth, label_ma=f"MA{eval_smooth}", color=color, raw_alpha=0.25)
        draw_stage_boundary(ax, stage1_end)
        set_common_axis(ax, "Global episode", ylabel, title)
        if "ratio" in col:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        pad_ylim(ax, y.dropna().values, pad_ratio=0.12)
        ax.legend(loc="best", frameon=False)

    fig.suptitle(
        f"Auxiliary training diagnostics of the proposed framework (seed={seed_label})",
        y=1.02,
        fontsize=14,
    )
    fig.text(
        0.5,
        0.005,
        "These diagnostics are recommended for appendix/internal analysis rather than the main loss-convergence figure.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.025, 1, 1])
    save_figure(fig, out_dir, "fig3_auxiliary_diagnostics_seed" + seed_label)


# ============================================================
# Figure 4: all-in-one overview
# ============================================================
def plot_all_in_one_overview(
    s1_train: pd.DataFrame,
    s2_train: pd.DataFrame,
    eval_df: pd.DataFrame,
    out_dir: Path,
    *,
    seed_label: str,
    stage1_smooth: int,
    stage2_smooth: int,
    eval_smooth: int,
    stage1_end: int,
) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(18.0, 10.2))
    axes = axes.reshape(-1)

    s1_x = numeric_series(s1_train, "episode")
    s2_x = numeric_series(s2_train, "episode")
    ev_x = numeric_series(eval_df, "global_episode")

    # Row 1: loss/stability
    top_specs = [
        (s1_x, numeric_series(s1_train, "avg_total_actor_loss"), stage1_smooth, "(a) Stage-1 warm-start loss", "Loss", "C1", "Episode"),
        (s1_x, numeric_series(s1_train, "avg_ratio_loss"), stage1_smooth, "(b) Stage-1 ratio BC loss", "Loss", "C3", "Episode"),
        (s2_x, numeric_series(s2_train, "avg_critic_loss"), stage2_smooth, "(c) Stage-2 critic loss", "Critic loss", "C0", "Episode"),
        (s2_x, numeric_series(s2_train, "avg_td_abs_error"), stage2_smooth, "(d) Stage-2 TD error", "TD error", "C4", "Episode"),
    ]
    for ax, (x, y, sm, title, ylabel, color, xlabel) in zip(axes[:4], top_specs):
        plot_raw_ma(ax, x, y, smooth=sm, label_ma=f"MA{sm}", color=color, raw_alpha=0.18)
        set_common_axis(ax, xlabel, ylabel, title)
        pad_ylim(ax, y.dropna().values, pad_ratio=0.10)
        ax.legend(loc="best", frameon=False)

    # Row 2: evaluation metrics
    eval_specs = [
        ("system_cost", "(e) Evaluation system cost", "Cost (×10$^3$)", lambda s: pd.to_numeric(s, errors="coerce") / 1000.0, "C0"),
        ("avg_delay", "(f) Evaluation average delay", "Delay (s)", lambda s: pd.to_numeric(s, errors="coerce"), "C1"),
        ("avg_energy", "(g) Evaluation average energy", "Energy (J/slot)", lambda s: pd.to_numeric(s, errors="coerce"), "C2"),
        ("feasible_ratio", "(h) Feasible ratio", "Feasible ratio", lambda s: pd.to_numeric(s, errors="coerce"), "C3"),
    ]
    for ax, (col, title, ylabel, trans, color) in zip(axes[4:8], eval_specs):
        y = trans(eval_df[col]) if col in eval_df.columns else pd.Series(dtype=float)
        plot_raw_ma(ax, ev_x, y, smooth=eval_smooth, label_ma=f"MA{eval_smooth}", color=color, raw_alpha=0.25)
        draw_stage_boundary(ax, stage1_end)
        set_common_axis(ax, "Global episode", ylabel, title)
        # if col == "feasible_ratio":
        #     ax.set_ylim(0.94, 1.01)
        #     ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        if col == "feasible_ratio":
            y_min = float(np.nanmin(y))
            y_max = float(np.nanmax(y))
            low = max(0.0, y_min - 0.05)
            high = min(1.02, y_max + 0.05)
            if high - low < 0.10:
                mid = 0.5 * (low + high)
                low = max(0.0, mid - 0.05)
                high = min(1.02, mid + 0.05)
            ax.set_ylim(low, high)
        else:
            pad_ylim(ax, y.dropna().values, pad_ratio=0.10 if col != "avg_energy" else 0.25, min_span=0.08 if col == "avg_energy" else None)
        ax.legend(loc="best", frameon=False)

    # Row 3: diagnostics
    diag_specs = [
        ("ratio_mean", "(i) Mean offloading ratio", "Ratio", "C0"),
        ("ratio_std", "(j) Offloading-ratio std.", "Std.", "C1"),
        ("local_exec_ratio", "(k) Local execution ratio", "Ratio", "C2"),
        ("avg_battery_violation", "(l) Battery diagnostic", "Diagnostic value", "C3"),
    ]
    for ax, (col, title, ylabel, color) in zip(axes[8:12], diag_specs):
        y = numeric_series(eval_df, col) if col in eval_df.columns else pd.Series(dtype=float)
        plot_raw_ma(ax, ev_x, y, smooth=eval_smooth, label_ma=f"MA{eval_smooth}", color=color, raw_alpha=0.25)
        draw_stage_boundary(ax, stage1_end)
        set_common_axis(ax, "Global episode", ylabel, title)
        pad_ylim(ax, y.dropna().values, pad_ratio=0.12)
        if "ratio" in col:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.legend(loc="best", frameon=False)

    fig.suptitle(
        f"Stage-wise convergence and diagnostics of the proposed framework (seed={seed_label})",
        y=1.01,
        fontsize=15,
    )
    fig.tight_layout()
    save_figure(fig, out_dir, "fig4_all_in_one_overview_seed" + seed_label)


# ============================================================
# Summary report
# ============================================================
def build_summary_report(
    s1_train: pd.DataFrame,
    s1_eval: pd.DataFrame,
    s2_train: pd.DataFrame,
    s2_eval: pd.DataFrame,
    eval_df: pd.DataFrame,
    out_dir: Path,
    *,
    stage1_dir: Path,
    stage2_dir: Path,
    s1_config: Dict,
    s2_config: Dict,
) -> None:
    def tail_mean(df: pd.DataFrame, col: str, n: int = 10):
        if col not in df.columns:
            return np.nan
        return float(pd.to_numeric(df[col], errors="coerce").tail(n).mean())

    def first_valid(df: pd.DataFrame, col: str):
        if col not in df.columns:
            return np.nan
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        return float(s.iloc[0]) if len(s) else np.nan

    def last_valid(df: pd.DataFrame, col: str):
        if col not in df.columns:
            return np.nan
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        return float(s.iloc[-1]) if len(s) else np.nan

    summary = {
        "stage1_dir": str(stage1_dir),
        "stage2_dir": str(stage2_dir),
        "stage1_train_rows": int(len(s1_train)),
        "stage1_eval_rows": int(len(s1_eval)),
        "stage2_train_rows": int(len(s2_train)),
        "stage2_eval_rows": int(len(s2_eval)),
        "stage1_config": s1_config,
        "stage2_config": s2_config,
        "stage1_final": {
            "warm_start_total_loss": last_valid(s1_train, "avg_total_actor_loss"),
            "ratio_bc_loss": last_valid(s1_train, "avg_ratio_loss"),
            "eval_system_cost": last_valid(s1_eval, "system_cost"),
            "eval_avg_delay": last_valid(s1_eval, "avg_delay"),
            "eval_avg_energy": last_valid(s1_eval, "avg_energy"),
            "eval_feasible_ratio": last_valid(s1_eval, "feasible_ratio"),
            "eval_battery_diagnostic": last_valid(s1_eval, "avg_battery_violation"),
        },
        "stage2_final": {
            "critic_loss": last_valid(s2_train, "avg_critic_loss"),
            "td_abs_error": last_valid(s2_train, "avg_td_abs_error"),
            "actor_policy_loss": last_valid(s2_train, "avg_actor_policy_loss"),
            "actor_total_loss": last_valid(s2_train, "avg_total_actor_loss"),
            "eval_system_cost": last_valid(s2_eval, "system_cost"),
            "eval_avg_delay": last_valid(s2_eval, "avg_delay"),
            "eval_avg_energy": last_valid(s2_eval, "avg_energy"),
            "eval_feasible_ratio": last_valid(s2_eval, "feasible_ratio"),
            "eval_battery_diagnostic": last_valid(s2_eval, "avg_battery_violation"),
        },
        "stage2_last10_mean": {
            "critic_loss": tail_mean(s2_train, "avg_critic_loss", 10),
            "td_abs_error": tail_mean(s2_train, "avg_td_abs_error", 10),
            "actor_policy_loss": tail_mean(s2_train, "avg_actor_policy_loss", 10),
            "eval_system_cost": tail_mean(s2_eval, "system_cost", 10),
            "eval_avg_delay": tail_mean(s2_eval, "avg_delay", 10),
            "eval_avg_energy": tail_mean(s2_eval, "avg_energy", 10),
            "eval_feasible_ratio": tail_mean(s2_eval, "feasible_ratio", 10),
            "eval_battery_diagnostic": tail_mean(s2_eval, "avg_battery_violation", 10),
        }
    }

    with open(out_dir / "summary_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Also save a compact CSV for easy copy into tables.
    rows = []
    for stage_name, d in [("Stage-1 final", summary["stage1_final"]), ("Stage-2 final", summary["stage2_final"]), ("Stage-2 last10 mean", summary["stage2_last10_mean"])]:
        row = {"stage": stage_name}
        row.update(d)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "summary_report.csv", index=False)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", required=True, help="Stage-1 log directory or .zip file")
    parser.add_argument("--stage2", required=True, help="Stage-2 log directory or .zip file")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--seed-label", default="72", help="Seed label shown in figure titles")
    parser.add_argument("--stage1-smooth", type=int, default=10, help="Moving average window for Stage-1 training curves")
    parser.add_argument("--stage2-smooth", type=int, default=20, help="Moving average window for Stage-2 training curves")
    parser.add_argument("--eval-smooth", type=int, default=5, help="Moving average window for evaluation curves")
    parser.add_argument("--log-critic", action="store_true", help="Use log scale for Stage-2 critic loss in Figure 1")
    args = parser.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    ensure_dir(out_dir)

    (
        stage1_dir,
        stage2_dir,
        s1_train,
        s1_eval,
        s2_train,
        s2_eval,
        s1_config,
        s2_config,
    ) = load_stage_data(Path(args.stage1), Path(args.stage2), out_dir)

    stage1_end = int(len(s1_train))
    s1_eval_global, s2_eval_global = add_global_eval_episode(s1_eval, s2_eval, stage1_end)
    eval_df = concat_eval(s1_eval_global, s2_eval_global)

    plot_training_loss_stability(
        s1_train,
        s2_train,
        out_dir,
        seed_label=str(args.seed_label),
        stage1_smooth=args.stage1_smooth,
        stage2_smooth=args.stage2_smooth,
        log_critic=args.log_critic,
    )

    plot_evaluation_convergence(
        eval_df,
        out_dir,
        seed_label=str(args.seed_label),
        eval_smooth=args.eval_smooth,
        stage1_end=stage1_end,
    )

    plot_auxiliary_diagnostics(
        eval_df,
        out_dir,
        seed_label=str(args.seed_label),
        eval_smooth=args.eval_smooth,
        stage1_end=stage1_end,
    )

    plot_all_in_one_overview(
        s1_train,
        s2_train,
        eval_df,
        out_dir,
        seed_label=str(args.seed_label),
        stage1_smooth=args.stage1_smooth,
        stage2_smooth=args.stage2_smooth,
        eval_smooth=args.eval_smooth,
        stage1_end=stage1_end,
    )

    build_summary_report(
        s1_train,
        s1_eval_global,
        s2_train,
        s2_eval_global,
        eval_df,
        out_dir,
        stage1_dir=stage1_dir,
        stage2_dir=stage2_dir,
        s1_config=s1_config,
        s2_config=s2_config,
    )

    print("=" * 90)
    print("Stage-wise convergence figures generated successfully.")
    print(f"Stage-1 folder : {stage1_dir}")
    print(f"Stage-2 folder : {stage2_dir}")
    print(f"Output folder  : {out_dir}")
    print("Generated files:")
    for p in sorted(out_dir.glob("*")):
        if p.is_file() and p.suffix.lower() in {".png", ".pdf", ".csv", ".json"}:
            print(f"  - {p.name}")
    print("=" * 90)


if __name__ == "__main__":
    main()
