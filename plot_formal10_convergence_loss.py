#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot formal 10-seed convergence curves for Proposed Stage-2.

Recommended paper figures:
  Figure 1: Stage-2 training convergence
    (a) Critic loss
    (b) TD absolute error
    (c) Self-imitation loss
    (d) Evaluation system cost

  Figure 2: Stage-2 performance convergence
    (a) System cost
    (b) Average delay
    (c) Deadline violation
    (d) Feasible ratio

Why not plot avg_total_actor_loss as the main convergence evidence?
  avg_total_actor_loss is a dynamically weighted composite loss. Its coefficients
  change during training, so its absolute value is not directly comparable across
  training episodes. Use critic loss, TD error, self-imitation loss, and evaluation
  performance as the main convergence evidence.
"""

import os
import glob
import argparse
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ============================================================
# User defaults
# ============================================================
DEFAULT_SEEDS = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132]

DEFAULT_STAGE2_TEMPLATE = (
    "proposed_full_stage2_main_d25_seed{seed}_ep700_v6_model_refine_formal10"
)

TRAIN_METRICS_MAIN = [
    ("avg_critic_loss", "Critic loss", "Critic loss"),
    ("avg_td_abs_error", "TD absolute error", "TD error"),
    ("avg_self_imitation_loss", "Self-imitation loss", "Self-imitation loss"),
]

EVAL_METRICS_MAIN = [
    ("system_cost", "Evaluation system cost", "System cost"),
]

EVAL_METRICS_PERF = [
    ("system_cost", "System cost", "System cost"),
    ("avg_delay", "Average delay", "Delay"),
    ("avg_deadline_violation", "Deadline violation", "Deadline violation"),
    ("feasible_ratio", "Feasible ratio", "Feasible ratio"),
]

# Optional diagnostic actor sub-losses.
ACTOR_DIAGNOSTIC_METRICS = [
    ("avg_move_sched_bc_loss", "Move/schedule trust-region loss", "Loss"),
    ("avg_ratio_bc_loss", "Ratio BC loss", "Loss"),
    ("avg_ratio_reg_loss", "Ratio regularization loss", "Loss"),
    ("avg_total_actor_loss", "Total actor loss (dynamic composite)", "Loss"),
]


# ============================================================
# Plot style
# ============================================================
def set_paper_style():
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.9,
        "grid.linewidth": 0.45,
        "lines.linewidth": 1.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ============================================================
# IO helpers
# ============================================================
def parse_seeds(seed_text: str) -> List[int]:
    if seed_text.strip().lower() in {"default", "formal10"}:
        return DEFAULT_SEEDS
    return [int(x.strip()) for x in seed_text.split(",") if x.strip()]


def find_stage2_dir(root: str, template: str, seed: int) -> str:
    expected = os.path.join(root, template.format(seed=seed))
    if os.path.isdir(expected):
        return expected

    # Fallback fuzzy search.
    patterns = [
        os.path.join(root, f"*seed{seed}*stage2*formal10*"),
        os.path.join(root, f"*stage2*seed{seed}*formal10*"),
        os.path.join(root, f"*seed{seed}*v6_model_refine*"),
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))

    candidates = sorted(set([c for c in candidates if os.path.isdir(c)]))
    if not candidates:
        raise FileNotFoundError(
            f"Cannot find Stage-2 result directory for seed={seed}. "
            f"Expected: {expected}"
        )

    # Prefer exact formal10 and ep700.
    def score(path: str) -> int:
        name = os.path.basename(path)
        s = 0
        if "formal10" in name:
            s += 10
        if "ep700" in name:
            s += 5
        if "v6_model_refine" in name:
            s += 3
        return s

    candidates = sorted(candidates, key=score, reverse=True)
    return candidates[0]


def read_seed_logs(
    root: str,
    template: str,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    run_dir = find_stage2_dir(root=root, template=template, seed=seed)
    train_path = os.path.join(run_dir, "train_log.csv")
    eval_path = os.path.join(run_dir, "eval_log.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing train_log.csv: {train_path}")
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"Missing eval_log.csv: {eval_path}")

    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)

    if "episode" not in train_df.columns:
        raise ValueError(f"{train_path} does not contain 'episode' column.")
    if "eval_episode" not in eval_df.columns:
        raise ValueError(f"{eval_path} does not contain 'eval_episode' column.")

    train_df["seed"] = seed
    eval_df["seed"] = seed
    return train_df, eval_df, run_dir


def load_all_logs(
    root: str,
    template: str,
    seeds: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, str]]:
    train_list = []
    eval_list = []
    dirs = {}

    for seed in seeds:
        train_df, eval_df, run_dir = read_seed_logs(root, template, seed)
        train_list.append(train_df)
        eval_list.append(eval_df)
        dirs[seed] = run_dir

    train_all = pd.concat(train_list, ignore_index=True)
    eval_all = pd.concat(eval_list, ignore_index=True)
    return train_all, eval_all, dirs


# ============================================================
# Data processing
# ============================================================
def moving_average_by_seed(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    smooth: int,
) -> pd.DataFrame:
    out = []
    for seed, g in df.groupby("seed"):
        g = g.sort_values(x_col).copy()
        values = pd.to_numeric(g[y_col], errors="coerce")
        if smooth > 1:
            values = values.rolling(window=smooth, min_periods=1, center=True).mean()
        g["_y_smooth"] = values
        out.append(g[[x_col, "seed", "_y_smooth"]])
    return pd.concat(out, ignore_index=True)


def aggregate_curve(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    smooth: int,
    band: str,
) -> pd.DataFrame:
    smoothed = moving_average_by_seed(df, x_col=x_col, y_col=y_col, smooth=smooth)

    rows = []
    for x, g in smoothed.groupby(x_col):
        vals = pd.to_numeric(g["_y_smooth"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue

        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if vals.size >= 2 else 0.0

        if band == "std":
            half = std
        elif band == "ci95":
            half = 1.96 * std / np.sqrt(vals.size) if vals.size >= 2 else 0.0
        elif band == "sem":
            half = std / np.sqrt(vals.size) if vals.size >= 2 else 0.0
        elif band == "none":
            half = 0.0
        else:
            raise ValueError(f"Unknown band mode: {band}")

        rows.append({
            x_col: x,
            "mean": mean,
            "std": std,
            "half": half,
            "n": int(vals.size),
            "lower": mean - half,
            "upper": mean + half,
        })

    return pd.DataFrame(rows).sort_values(x_col)


def get_seed_smoothed_curves(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    smooth: int,
) -> pd.DataFrame:
    return moving_average_by_seed(df, x_col=x_col, y_col=y_col, smooth=smooth)


def audit_logs(train_all: pd.DataFrame, eval_all: pd.DataFrame, seeds: List[int]) -> None:
    print("=" * 100)
    print("Data audit")
    print("=" * 100)

    for seed in seeds:
        tr = train_all[train_all["seed"] == seed]
        ev = eval_all[eval_all["seed"] == seed]

        ep_min = int(tr["episode"].min()) if len(tr) else -1
        ep_max = int(tr["episode"].max()) if len(tr) else -1
        ev_min = int(ev["eval_episode"].min()) if len(ev) else -1
        ev_max = int(ev["eval_episode"].max()) if len(ev) else -1

        msg = (
            f"seed={seed:<3d} | "
            f"train_rows={len(tr):<4d} episode={ep_min}->{ep_max} | "
            f"eval_rows={len(ev):<4d} eval_episode={ev_min}->{ev_max}"
        )

        warnings_list = []
        for col in ["avg_critic_loss", "avg_td_abs_error", "avg_total_actor_loss"]:
            if col in tr.columns:
                x = pd.to_numeric(tr[col], errors="coerce")
                n_nan = int(x.isna().sum())
                n_inf = int(np.isinf(x.dropna()).sum())
                if n_nan > 0:
                    warnings_list.append(f"{col}: NaN={n_nan}")
                if n_inf > 0:
                    warnings_list.append(f"{col}: Inf={n_inf}")

        for col in ["system_cost", "avg_delay", "feasible_ratio"]:
            if col in ev.columns:
                x = pd.to_numeric(ev[col], errors="coerce")
                if x.isna().any():
                    warnings_list.append(f"{col}: NaN in eval")

        print(msg)
        if warnings_list:
            print(" " * 10 + "notes:", "; ".join(warnings_list))

    print("=" * 100)
    print()


# ============================================================
# Plot helpers
# ============================================================
def plot_one_axis(
    ax,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    smooth: int,
    band: str,
    show_individual: bool,
    mean_label: str,
):
    # Individual seed curves as light background.
    if show_individual:
        seed_curves = get_seed_smoothed_curves(df, x_col=x_col, y_col=y_col, smooth=smooth)
        for _, g in seed_curves.groupby("seed"):
            g = g.sort_values(x_col)
            ax.plot(
                g[x_col].to_numpy(),
                g["_y_smooth"].to_numpy(),
                color="0.78",
                linewidth=0.7,
                alpha=0.55,
                zorder=1,
            )

    agg = aggregate_curve(df, x_col=x_col, y_col=y_col, smooth=smooth, band=band)
    x = agg[x_col].to_numpy()
    mean = agg["mean"].to_numpy()
    lower = agg["lower"].to_numpy()
    upper = agg["upper"].to_numpy()

    ax.plot(x, mean, color="#1f77b4", linewidth=2.0, label=mean_label, zorder=3)

    if band != "none":
        ax.fill_between(
            x,
            lower,
            upper,
            color="#1f77b4",
            alpha=0.18,
            linewidth=0.0,
            label=band,
            zorder=2,
        )

    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.30)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))


def make_2x2_figure(
    df_train: pd.DataFrame,
    df_eval: pd.DataFrame,
    panels: List[Tuple[str, str, str, str, str]],
    out_path_base: str,
    train_smooth: int,
    eval_smooth: int,
    band: str,
    show_individual: bool,
    suptitle: str,
):
    """
    panels layout:
      (source, metric, title, ylabel, x_col)
      source = "train" or "eval"
    """
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    axes = axes.reshape(-1)

    for ax, (source, metric, title, ylabel, x_col) in zip(axes, panels):
        if source == "train":
            df = df_train
            smooth = train_smooth
        elif source == "eval":
            df = df_eval
            smooth = eval_smooth
        else:
            raise ValueError(source)

        if metric not in df.columns:
            ax.text(
                0.5,
                0.5,
                f"Missing column:\n{metric}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(title)
            continue

        plot_one_axis(
            ax=ax,
            df=df,
            x_col=x_col,
            y_col=metric,
            title=title,
            ylabel=ylabel,
            smooth=smooth,
            band=band,
            show_individual=show_individual,
            mean_label="Mean over 10 seeds",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(handles), 3),
            frameon=False,
            bbox_to_anchor=(0.5, 0.985),
        )

    fig.suptitle(suptitle, y=1.03, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    png_path = out_path_base + ".png"
    pdf_path = out_path_base + ".pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVE] {png_path}")
    print(f"[SAVE] {pdf_path}")


def save_aggregate_csv(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    smooth: int,
    band: str,
    out_csv: str,
):
    agg = aggregate_curve(df, x_col=x_col, y_col=y_col, smooth=smooth, band=band)
    agg.insert(0, "metric", y_col)
    agg.to_csv(out_csv, index=False)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="results/convergence_training",
        help="Root directory containing convergence_training runs.",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=DEFAULT_STAGE2_TEMPLATE,
        help="Stage-2 run directory template. Must contain {seed}.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="formal10",
        help="Comma-separated seeds or 'formal10'.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/convergence_outputs/formal10_stage2_loss",
        help="Output directory.",
    )
    parser.add_argument(
        "--train-smooth",
        type=int,
        default=15,
        help="Moving-average window for train_log curves.",
    )
    parser.add_argument(
        "--eval-smooth",
        type=int,
        default=3,
        help="Moving-average window for eval_log curves.",
    )
    parser.add_argument(
        "--band",
        type=str,
        default="ci95",
        choices=["ci95", "std", "sem", "none"],
        help="Shaded band type.",
    )
    parser.add_argument(
        "--show-individual",
        action="store_true",
        help="Show individual seed curves as light gray background.",
    )
    parser.add_argument(
        "--no-individual",
        action="store_true",
        help="Do not show individual seed curves.",
    )
    args = parser.parse_args()

    set_paper_style()

    seeds = parse_seeds(args.seeds)
    os.makedirs(args.out, exist_ok=True)

    train_all, eval_all, dirs = load_all_logs(
        root=args.root,
        template=args.template,
        seeds=seeds,
    )

    audit_logs(train_all, eval_all, seeds)

    # By default, show faint individual curves unless explicitly disabled.
    show_individual = True
    if args.no_individual:
        show_individual = False
    if args.show_individual:
        show_individual = True

    # ------------------------------------------------------------
    # Figure 1: loss convergence
    # ------------------------------------------------------------
    panels_loss = [
        ("train", "avg_critic_loss", "Critic loss", "Critic loss", "episode"),
        ("train", "avg_td_abs_error", "TD absolute error", "TD error", "episode"),
        ("train", "avg_self_imitation_loss", "Self-imitation loss", "Self-imitation loss", "episode"),
        ("eval", "system_cost", "Evaluation system cost", "System cost", "eval_episode"),
    ]

    make_2x2_figure(
        df_train=train_all,
        df_eval=eval_all,
        panels=panels_loss,
        out_path_base=os.path.join(args.out, "fig_stage2_loss_convergence"),
        train_smooth=args.train_smooth,
        eval_smooth=args.eval_smooth,
        band=args.band,
        show_individual=show_individual,
        suptitle="Stage-2 training convergence",
    )

    # ------------------------------------------------------------
    # Figure 2: performance convergence
    # ------------------------------------------------------------
    panels_perf = [
        ("eval", "system_cost", "System cost", "System cost", "eval_episode"),
        ("eval", "avg_delay", "Average delay", "Delay", "eval_episode"),
        ("eval", "avg_deadline_violation", "Deadline violation", "Deadline violation", "eval_episode"),
        ("eval", "feasible_ratio", "Feasible ratio", "Feasible ratio", "eval_episode"),
    ]

    make_2x2_figure(
        df_train=train_all,
        df_eval=eval_all,
        panels=panels_perf,
        out_path_base=os.path.join(args.out, "fig_stage2_performance_convergence"),
        train_smooth=args.train_smooth,
        eval_smooth=args.eval_smooth,
        band=args.band,
        show_individual=show_individual,
        suptitle="Stage-2 performance convergence",
    )

    # ------------------------------------------------------------
    # Figure 3: actor diagnostic sub-losses
    # Not recommended as the main convergence evidence, but useful for advisor review.
    # ------------------------------------------------------------
    panels_actor = [
        ("train", "avg_move_sched_bc_loss", "Move/schedule trust-region loss", "Loss", "episode"),
        ("train", "avg_ratio_bc_loss", "Ratio BC loss", "Loss", "episode"),
        ("train", "avg_ratio_reg_loss", "Ratio regularization loss", "Loss", "episode"),
        ("train", "avg_total_actor_loss", "Total actor loss (dynamic composite)", "Loss", "episode"),
    ]

    make_2x2_figure(
        df_train=train_all,
        df_eval=eval_all,
        panels=panels_actor,
        out_path_base=os.path.join(args.out, "fig_stage2_actor_diagnostic_losses"),
        train_smooth=args.train_smooth,
        eval_smooth=args.eval_smooth,
        band=args.band,
        show_individual=show_individual,
        suptitle="Stage-2 actor-side diagnostic losses",
    )

    # ------------------------------------------------------------
    # Save aggregated numerical curves for reproducibility.
    # ------------------------------------------------------------
    curve_dir = os.path.join(args.out, "aggregated_curves")
    os.makedirs(curve_dir, exist_ok=True)

    for metric, _, _ in TRAIN_METRICS_MAIN + ACTOR_DIAGNOSTIC_METRICS:
        if metric in train_all.columns:
            save_aggregate_csv(
                train_all,
                x_col="episode",
                y_col=metric,
                smooth=args.train_smooth,
                band=args.band,
                out_csv=os.path.join(curve_dir, f"{metric}_aggregate.csv"),
            )

    for metric, _, _ in EVAL_METRICS_PERF:
        if metric in eval_all.columns:
            save_aggregate_csv(
                eval_all,
                x_col="eval_episode",
                y_col=metric,
                smooth=args.eval_smooth,
                band=args.band,
                out_csv=os.path.join(curve_dir, f"{metric}_aggregate.csv"),
            )

    print()
    print("=" * 100)
    print("Finished.")
    print(f"Output directory: {args.out}")
    print("=" * 100)
    print("Recommended paper figures:")
    print(f"  {os.path.join(args.out, 'fig_stage2_loss_convergence.pdf')}")
    print(f"  {os.path.join(args.out, 'fig_stage2_performance_convergence.pdf')}")
    print()
    print("Advisor diagnostic figure:")
    print(f"  {os.path.join(args.out, 'fig_stage2_actor_diagnostic_losses.pdf')}")
    print()
    print("Note:")
    print("  Do not use avg_total_actor_loss alone as convergence evidence.")
    print("  It is a dynamically weighted composite objective.")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        main()