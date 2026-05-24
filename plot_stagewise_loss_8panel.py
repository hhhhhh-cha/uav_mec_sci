#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot an 8-panel convergence figure for the two-stage UAV-MEC Proposed method.

This version is designed to avoid the main problem in the previous 8-panel figure:
Stage-1 and Stage-2 losses have different mathematical meanings, so they should
NOT be drawn as one continuous loss curve.

Recommended command:
  python3 plot_stagewise_loss_eval_8panel_v3.py \
    --stage1 results/convergence_training/proposed_full_stage1_damaged_w50_seed72_logged_v2_ep120 \
    --stage2 results/convergence_training/proposed_full_stage2_damaged_w50_seed72_logged_v2_warm20_margin \
    --out results/convergence_outputs/stagewise_loss_eval_8panel_v3_seed72 \
    --seed-label 72 \
    --stage1-smooth 10 \
    --stage2-smooth 10 \
    --eval-smooth 5

Outputs:
  stagewise_loss_eval_8panel_v3.png
  stagewise_loss_eval_8panel_v3.pdf
  merged_eval_stagewise.csv
  stage1_train_processed.csv
  stage2_train_processed.csv

Figure layout:
  (a) Stage-1 warm-start loss
  (b) Stage-1 ratio BC loss
  (c) Stage-2 critic loss
  (d) Stage-2 TD absolute error
  (e) Evaluation system cost
  (f) Evaluation average delay
  (g) Evaluation deadline violation
  (h) Evaluation feasible ratio

Author note:
  For reinforcement learning, actor/critic losses are generally not expected to
  decrease monotonically. The correct claim is: losses become bounded/stable,
  while evaluation cost, delay and constraint metrics remain stable.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter


# =========================================================
# Colors: explicitly separated by figure type
# =========================================================
COLOR_STAGE1 = "#D55E00"      # orange / vermillion
COLOR_STAGE2 = "#0072B2"      # blue
COLOR_EVAL_COST = "#009E73"   # green
COLOR_EVAL_DELAY = "#CC79A7"  # purple-pink
COLOR_EVAL_CONSTR = "#E69F00" # amber
COLOR_RAW = "#9E9E9E"         # gray raw traces
COLOR_TRANSITION = "#404040"  # dark gray transition line


# =========================================================
# Data utilities
# =========================================================
def _read_csv(path: Path, required_columns: Iterable[str], name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{name} is empty: {path}")
    for c in required_columns:
        if c not in df.columns:
            raise ValueError(f"{name} must contain column '{c}', but got columns: {list(df.columns)}")
    return df


def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="ignore")
    return out


def _ensure_system_cost(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "system_cost" not in out.columns:
        if "episode_reward" not in out.columns:
            raise ValueError("eval_log.csv must contain either 'system_cost' or 'episode_reward'.")
        out["system_cost"] = -pd.to_numeric(out["episode_reward"], errors="coerce")
    return out


def _smooth(y: pd.Series, window: int) -> pd.Series:
    y = pd.to_numeric(y, errors="coerce")
    if window <= 1:
        return y
    return y.rolling(window=window, min_periods=1, center=True).mean()


def _safe_log10(y: pd.Series) -> pd.Series:
    y = pd.to_numeric(y, errors="coerce")
    return pd.Series(np.where(y > 0, np.log10(y), np.nan), index=y.index)


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _stagewise_eval(stage1_eval: pd.DataFrame, stage2_eval: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    e1 = _ensure_system_cost(_numeric(stage1_eval)).copy()
    e2 = _ensure_system_cost(_numeric(stage2_eval)).copy()

    e1["eval_episode"] = pd.to_numeric(e1["eval_episode"], errors="coerce")
    e2["eval_episode"] = pd.to_numeric(e2["eval_episode"], errors="coerce")
    e1 = e1.dropna(subset=["eval_episode"])
    e2 = e2.dropna(subset=["eval_episode"])

    transition = float(e1["eval_episode"].max()) + 1.0

    e1["stage"] = "Stage-1 warm-start"
    e2["stage"] = "Stage-2 CTDE fine-tuning"
    e1["global_episode"] = e1["eval_episode"].astype(float)
    e2["global_episode"] = transition + e2["eval_episode"].astype(float)

    merged = pd.concat([e1, e2], ignore_index=True)
    merged = merged.sort_values("global_episode").reset_index(drop=True)
    return merged, transition


def _prepare_stage1_train(df: pd.DataFrame) -> pd.DataFrame:
    out = _numeric(df).copy()
    out["episode"] = pd.to_numeric(out["episode"], errors="coerce")
    out = out.dropna(subset=["episode"]).sort_values("episode").reset_index(drop=True)

    # Stage-1 warm-start loss is the supervised actor-side loss.
    # Prefer avg_total_actor_loss if available.
    warm_col = _first_existing_column(out, ["avg_total_actor_loss", "avg_move_sched_loss"])
    if warm_col is not None:
        out["stage1_warm_start_loss"] = pd.to_numeric(out[warm_col], errors="coerce")
    else:
        out["stage1_warm_start_loss"] = np.nan

    ratio_col = _first_existing_column(out, ["avg_ratio_loss", "avg_ratio_bc_loss"])
    if ratio_col is not None:
        out["stage1_ratio_bc_loss"] = pd.to_numeric(out[ratio_col], errors="coerce")
    else:
        out["stage1_ratio_bc_loss"] = np.nan

    return out


def _prepare_stage2_train(df: pd.DataFrame, use_log_critic: bool = False) -> pd.DataFrame:
    out = _numeric(df).copy()
    out["episode"] = pd.to_numeric(out["episode"], errors="coerce")
    out = out.dropna(subset=["episode"]).sort_values("episode").reset_index(drop=True)

    if "avg_critic_loss" in out.columns:
        c = pd.to_numeric(out["avg_critic_loss"], errors="coerce")
        out["stage2_critic_loss_plot"] = _safe_log10(c) if use_log_critic else c
    else:
        out["stage2_critic_loss_plot"] = np.nan

    if "avg_td_abs_error" in out.columns:
        td = pd.to_numeric(out["avg_td_abs_error"], errors="coerce")
        out["stage2_td_error_plot"] = _safe_log10(td) if use_log_critic else td
    else:
        out["stage2_td_error_plot"] = np.nan

    return out


# =========================================================
# Plot helpers
# =========================================================
def _style_axis(ax):
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.tick_params(axis="both", labelsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _plot_train_metric(
    ax,
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    smooth_window: int,
    color: str,
    y_fmt: Optional[str] = None,
    y_lim: Optional[Tuple[float, float]] = None,
    legend_loc: str = "best",
):
    if metric not in df.columns:
        ax.text(0.5, 0.5, f"Missing metric:\n{metric}", ha="center", va="center", fontsize=9)
        ax.set_axis_off()
        return

    work = df[["episode", metric]].copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=["episode", metric])

    if work.empty:
        ax.text(0.5, 0.5, f"No valid data:\n{metric}", ha="center", va="center", fontsize=9)
        ax.set_axis_off()
        return

    y_raw = work[metric]
    y_ma = _smooth(y_raw, smooth_window)

    ax.plot(
        work["episode"],
        y_raw,
        linestyle="--",
        linewidth=0.9,
        color=COLOR_RAW,
        alpha=0.35,
        label="Raw",
    )
    ax.plot(
        work["episode"],
        y_ma,
        linestyle="-",
        linewidth=2.1,
        color=color,
        label=f"MA{smooth_window}",
    )

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Episode", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if y_fmt:
        ax.yaxis.set_major_formatter(FormatStrFormatter(y_fmt))
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    _style_axis(ax)
    ax.legend(frameon=False, fontsize=8, loc=legend_loc)


def _plot_eval_metric(
    ax,
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    transition: float,
    smooth_window: int,
    color: str,
    scale: float = 1.0,
    y_fmt: Optional[str] = None,
    y_lim: Optional[Tuple[float, float]] = None,
    legend_loc: str = "best",
):
    if metric not in df.columns:
        ax.text(0.5, 0.5, f"Missing metric:\n{metric}", ha="center", va="center", fontsize=9)
        ax.set_axis_off()
        return

    work = df[["global_episode", metric]].copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=["global_episode", metric])

    if work.empty:
        ax.text(0.5, 0.5, f"No valid data:\n{metric}", ha="center", va="center", fontsize=9)
        ax.set_axis_off()
        return

    y_raw = work[metric] / scale
    y_ma = _smooth(y_raw, smooth_window)

    ax.plot(
        work["global_episode"],
        y_raw,
        linestyle="--",
        linewidth=0.9,
        color=COLOR_RAW,
        alpha=0.35,
        label="Raw",
    )
    ax.plot(
        work["global_episode"],
        y_ma,
        linestyle="-",
        linewidth=2.1,
        color=color,
        label=f"MA{smooth_window}",
    )
    ax.axvline(
        transition,
        linestyle="--",
        linewidth=1.2,
        color=COLOR_TRANSITION,
        alpha=0.85,
    )

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Global episode", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if y_fmt:
        ax.yaxis.set_major_formatter(FormatStrFormatter(y_fmt))
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    _style_axis(ax)
    ax.legend(frameon=False, fontsize=8, loc=legend_loc)


def _auto_feasible_ylim(eval_df: pd.DataFrame) -> Tuple[float, float]:
    if "feasible_ratio" not in eval_df.columns:
        return 0.0, 1.05
    y = pd.to_numeric(eval_df["feasible_ratio"], errors="coerce").dropna()
    if y.empty:
        return 0.0, 1.05
    ymin, ymax = float(y.min()), float(y.max())
    if ymin >= 0.95 and ymax <= 1.000001:
        return 0.94, 1.01
    return max(0.0, ymin - 0.05), min(1.05, ymax + 0.05)


def draw_figure(
    stage1_train: pd.DataFrame,
    stage2_train: pd.DataFrame,
    eval_df: pd.DataFrame,
    eval_transition: float,
    out_dir: Path,
    seed_label: str,
    stage1_smooth: int,
    stage2_smooth: int,
    eval_smooth: int,
    use_log_critic: bool,
):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "axes.unicode_minus": False,
    })

    fig, axes = plt.subplots(2, 4, figsize=(18.5, 7.2))
    axes = axes.reshape(-1)

    # -------------------------
    # Top row: stage-specific losses
    # -------------------------
    _plot_train_metric(
        axes[0],
        stage1_train,
        "stage1_warm_start_loss",
        "(a) Stage-1 warm-start loss",
        "Warm-start loss",
        stage1_smooth,
        COLOR_STAGE1,
        y_fmt="%.3f",
        legend_loc="best",
    )

    _plot_train_metric(
        axes[1],
        stage1_train,
        "stage1_ratio_bc_loss",
        "(b) Stage-1 ratio BC loss",
        "Ratio BC loss",
        stage1_smooth,
        COLOR_STAGE1,
        y_fmt="%.4f",
        legend_loc="best",
    )

    critic_ylabel = r"log$_{10}$(critic loss)" if use_log_critic else "Critic loss"
    td_ylabel = r"log$_{10}$(TD absolute error)" if use_log_critic else "TD absolute error"

    _plot_train_metric(
        axes[2],
        stage2_train,
        "stage2_critic_loss_plot",
        "(c) Stage-2 critic loss",
        critic_ylabel,
        stage2_smooth,
        COLOR_STAGE2,
        y_fmt="%.3f",
        legend_loc="best",
    )

    _plot_train_metric(
        axes[3],
        stage2_train,
        "stage2_td_error_plot",
        "(d) Stage-2 TD error",
        td_ylabel,
        stage2_smooth,
        COLOR_STAGE2,
        y_fmt="%.3f",
        legend_loc="best",
    )

    # -------------------------
    # Bottom row: evaluation performance / constraints
    # -------------------------
    _plot_eval_metric(
        axes[4],
        eval_df,
        "system_cost",
        "(e) Evaluation system cost",
        "Weighted system cost (×10³)",
        eval_transition,
        eval_smooth,
        COLOR_EVAL_COST,
        scale=1000.0,
        y_fmt="%.1f",
        legend_loc="best",
    )

    _plot_eval_metric(
        axes[5],
        eval_df,
        "avg_delay",
        "(f) Evaluation average delay",
        "Average delay (s)",
        eval_transition,
        eval_smooth,
        COLOR_EVAL_DELAY,
        scale=1.0,
        y_fmt="%.2f",
        legend_loc="best",
    )

    _plot_eval_metric(
        axes[6],
        eval_df,
        "avg_deadline_violation",
        "(g) Deadline violation",
        "Deadline violation",
        eval_transition,
        eval_smooth,
        COLOR_EVAL_CONSTR,
        scale=1.0,
        y_fmt="%.3f",
        legend_loc="best",
    )

    _plot_eval_metric(
        axes[7],
        eval_df,
        "feasible_ratio",
        "(h) Feasible ratio",
        "Feasible ratio",
        eval_transition,
        eval_smooth,
        COLOR_EVAL_CONSTR,
        scale=1.0,
        y_fmt="%.2f",
        y_lim=_auto_feasible_ylim(eval_df),
        legend_loc="best",
    )

    fig.suptitle(
        f"Convergence behavior of the proposed two-stage training framework (seed={seed_label})",
        fontsize=14,
        y=0.995,
    )

    # Stage transition note for bottom-row evaluation panels.
    fig.text(
        0.5,
        0.018,
        "The vertical dashed line in evaluation panels denotes the transition from Stage-1 warm-start to Stage-2 CTDE fine-tuning.",
        ha="center",
        va="center",
        fontsize=9,
    )

    fig.tight_layout(rect=[0.0, 0.045, 1.0, 0.96])

    png_path = out_dir / "stagewise_loss_eval_8panel_v3.png"
    pdf_path = out_dir / "stagewise_loss_eval_8panel_v3.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", required=True, help="Stage-1 result directory containing train_log.csv and eval_log.csv")
    parser.add_argument("--stage2", required=True, help="Stage-2 result directory containing train_log.csv and eval_log.csv")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--seed-label", default="72", help="Seed label in the title")
    parser.add_argument("--stage1-smooth", type=int, default=10, help="MA window for Stage-1 loss curves")
    parser.add_argument("--stage2-smooth", type=int, default=10, help="MA window for Stage-2 loss curves")
    parser.add_argument("--eval-smooth", type=int, default=5, help="MA window for evaluation curves")
    parser.add_argument(
        "--log-critic",
        action="store_true",
        help="Plot log10 critic loss and log10 TD error. Default: plot Stage-2 losses in linear scale.",
    )
    args = parser.parse_args()

    stage1_dir = Path(args.stage1)
    stage2_dir = Path(args.stage2)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage1_train_raw = _read_csv(stage1_dir / "train_log.csv", ["episode"], "Stage-1 train_log.csv")
    stage2_train_raw = _read_csv(stage2_dir / "train_log.csv", ["episode"], "Stage-2 train_log.csv")
    stage1_eval_raw = _read_csv(stage1_dir / "eval_log.csv", ["eval_episode"], "Stage-1 eval_log.csv")
    stage2_eval_raw = _read_csv(stage2_dir / "eval_log.csv", ["eval_episode"], "Stage-2 eval_log.csv")

    stage1_train = _prepare_stage1_train(stage1_train_raw)
    stage2_train = _prepare_stage2_train(stage2_train_raw, use_log_critic=args.log_critic)
    eval_df, eval_transition = _stagewise_eval(stage1_eval_raw, stage2_eval_raw)

    stage1_train.to_csv(out_dir / "stage1_train_processed.csv", index=False)
    stage2_train.to_csv(out_dir / "stage2_train_processed.csv", index=False)
    eval_df.to_csv(out_dir / "merged_eval_stagewise.csv", index=False)

    png_path, pdf_path = draw_figure(
        stage1_train=stage1_train,
        stage2_train=stage2_train,
        eval_df=eval_df,
        eval_transition=eval_transition,
        out_dir=out_dir,
        seed_label=args.seed_label,
        stage1_smooth=args.stage1_smooth,
        stage2_smooth=args.stage2_smooth,
        eval_smooth=args.eval_smooth,
        use_log_critic=args.log_critic,
    )

    print("Saved:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")
    print(f"  {out_dir / 'stage1_train_processed.csv'}")
    print(f"  {out_dir / 'stage2_train_processed.csv'}")
    print(f"  {out_dir / 'merged_eval_stagewise.csv'}")


if __name__ == "__main__":
    main()
