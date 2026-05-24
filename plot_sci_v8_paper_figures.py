#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCI-style plotting script for v8 UAV-MEC experiments.

Inputs expected:
  --pgboost-root: directories each containing train_log.csv/eval_log.csv/eval_actoronly_log.csv
  --nopg-root:    same structure for w/o policy-gradient ablation
  --main-root:    main comparison root containing trainseed*_eval10/detailed_results.csv

Outputs:
  PNG/PDF/SVG figures and CSV summary tables under --out.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

SEEDS_DEFAULT = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132]

METHOD_ORDER = [
    "Random",
    "Greedy",
    "Greedy_Refine4",
    "Pure_MADDPG",
    "Pure_MADDPG_Refine4",
    "Proposed_woPG_ActorOnly",
    "Proposed_wPG_ActorOnly",
    "Proposed_wPG_Refine4",
]

METHOD_LABELS = {
    "Random": "Random",
    "Greedy": "Greedy",
    "Greedy_Refine4": "Greedy+Refine4",
    "Pure_MADDPG": "Pure MADDPG",
    "Pure_MADDPG_Refine4": "Pure MADDPG+Refine4",
    "Proposed_woPG_ActorOnly": "Proposed w/o PG",
    "Proposed_wPG_ActorOnly": "Proposed ActorOnly",
    "Proposed_wPG_Refine4": "Proposed+Refine4",
}

LOWER_BETTER = {
    "system_cost": True,
    "avg_delay": True,
    "avg_energy": True,
    "avg_deadline_violation": True,
    "episode_reward": False,
    "feasible_ratio": False,
    "neighbor_exec_ratio": False,
}


def parse_seed_from_name(name: str) -> int | None:
    m = re.search(r"seed(\d+)", name)
    return int(m.group(1)) if m else None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in df.columns:
        if c not in {"method", "pgboost_prefix", "nopg_prefix", "pure_prefix"}:
            try:
                df[c] = pd.to_numeric(df[c])
            except Exception:
                pass
    return df


def find_run_dirs(root: Path, seeds: Iterable[int]) -> Dict[int, Path]:
    seeds = list(seeds)
    out: Dict[int, Path] = {}
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "train_log.csv").exists():
            continue
        s = parse_seed_from_name(d.name)
        if s in seeds:
            out[s] = d
    missing = [s for s in seeds if s not in out]
    if missing:
        raise FileNotFoundError(f"Missing run directories for seeds {missing} under {root}")
    return out


def load_training_logs(root: Path, variant: str, seeds: Iterable[int]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_dirs = find_run_dirs(root, seeds)
    train_list, eval_list, actoronly_list = [], [], []
    for s, d in sorted(run_dirs.items()):
        tr = read_csv(d / "train_log.csv")
        ev = read_csv(d / "eval_log.csv")
        ao = read_csv(d / "eval_actoronly_log.csv")
        tr["seed"] = s
        ev["seed"] = s
        ao["seed"] = s
        tr["variant"] = variant
        ev["variant"] = variant
        ao["variant"] = variant
        train_list.append(tr)
        eval_list.append(ev)
        actoronly_list.append(ao)
    return pd.concat(train_list, ignore_index=True), pd.concat(eval_list, ignore_index=True), pd.concat(actoronly_list, ignore_index=True)


def aggregate_curve(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    tmp = df[[x, y]].copy()
    tmp[y] = pd.to_numeric(tmp[y], errors="coerce")
    g = tmp.groupby(x, as_index=False)[y].agg(["mean", "std", "count"]).reset_index()
    g["std"] = g["std"].fillna(0.0)
    g["ci95"] = 1.96 * g["std"] / np.sqrt(g["count"].clip(lower=1))
    return g


def smooth_series(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    s = pd.Series(y, dtype="float64")
    return s.rolling(window=window, min_periods=1, center=False).mean().to_numpy()


def plot_curve_panel(ax, df: pd.DataFrame, x: str, y: str, title: str, ylabel: str, smooth: int, band: str = "std") -> None:
    agg = aggregate_curve(df, x, y)
    xv = agg[x].to_numpy(dtype=float)
    mean = agg["mean"].to_numpy(dtype=float)
    spread = agg["ci95" if band == "ci95" else "std"].to_numpy(dtype=float)
    mean_s = smooth_series(mean, smooth)
    spread_s = smooth_series(spread, smooth)
    ax.plot(xv, mean_s, linewidth=1.8, label="Mean")
    ax.fill_between(xv, mean_s - spread_s, mean_s + spread_s, alpha=0.18, linewidth=0)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Episode", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.tick_params(axis="both", labelsize=8)


def save_fig(fig, out_dir: Path, name: str, dpi: int = 300) -> None:
    ensure_dir(out_dir)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"{name}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_training_convergence(pg_train: pd.DataFrame, np_train: pd.DataFrame, out_dir: Path, smooth: int, band: str) -> None:
    fig_dir = out_dir / "fig_training_convergence"
    ensure_dir(fig_dir)

    train_metrics = [
        ("episode_reward", "Training reward", "Episode reward"),
        ("avg_critic_loss", "Critic loss", "Loss"),
        ("avg_td_abs_error", "TD absolute error", "TD error"),
        ("avg_total_actor_loss", "Actor total loss", "Loss"),
        ("avg_refined_ratio_bc_loss", "Ratio distillation loss", "Loss"),
        ("avg_ratio_target_delta", "Ratio target delta", "Abs. delta"),
        ("avg_refined_sched_ce_loss", "Scheduling CE loss", "Loss"),
        ("avg_refined_sched_acc", "Scheduling accuracy", "Accuracy"),
    ]

    # Main 8-panel figure for pgboost.
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.2))
    for ax, (metric, title, ylabel) in zip(axes.ravel(), train_metrics):
        plot_curve_panel(ax, pg_train, "episode", metric, title, ylabel, smooth=smooth, band=band)
    fig.suptitle("Training convergence of Proposed wPG", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, fig_dir, "training_convergence_pgboost_8panel")

    # Individual figures for paper flexibility.
    for metric, title, ylabel in train_metrics:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        plot_curve_panel(ax, pg_train, "episode", metric, title, ylabel, smooth=smooth, band=band)
        fig.tight_layout()
        save_fig(fig, fig_dir, f"pgboost_{metric}")

    # Compare wPG vs w/o PG for selected loss/performance metrics.
    compare_metrics = [
        ("episode_reward", "Training reward", "Episode reward"),
        ("avg_critic_loss", "Critic loss", "Loss"),
        ("avg_td_abs_error", "TD absolute error", "TD error"),
        ("avg_total_actor_loss", "Actor total loss", "Loss"),
        ("avg_refined_ratio_bc_loss", "Ratio distillation loss", "Loss"),
        ("avg_refined_sched_ce_loss", "Scheduling CE loss", "Loss"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0))
    for ax, (metric, title, ylabel) in zip(axes.ravel(), compare_metrics):
        for df, label in [(np_train, "w/o PG"), (pg_train, "wPG")]:
            agg = aggregate_curve(df, "episode", metric)
            xv = agg["episode"].to_numpy(dtype=float)
            mean = smooth_series(agg["mean"].to_numpy(dtype=float), smooth)
            ax.plot(xv, mean, linewidth=1.8, label=label)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Episode", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.tick_params(axis="both", labelsize=8)
    axes.ravel()[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Training comparison: Proposed w/o PG vs Proposed wPG", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, fig_dir, "training_compare_wopg_vs_wpg_6panel")


def plot_eval_convergence(pg_eval_actor: pd.DataFrame, pg_eval_refine: pd.DataFrame, out_dir: Path, smooth: int, band: str) -> None:
    fig_dir = out_dir / "fig_eval_convergence"
    ensure_dir(fig_dir)

    eval_metrics = [
        ("system_cost", "System cost", "Cost"),
        ("avg_delay", "Average delay", "Delay"),
        ("feasible_ratio", "Feasible ratio", "Ratio"),
        ("avg_deadline_violation", "Deadline violation", "Violation"),
        ("ratio_mean", "Offloading ratio mean", "Ratio"),
        ("neighbor_exec_ratio", "Neighbor execution ratio", "Ratio"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0))
    for ax, (metric, title, ylabel) in zip(axes.ravel(), eval_metrics):
        plot_curve_panel(ax, pg_eval_actor, "eval_episode", metric, title, ylabel, smooth=smooth, band=band)
    fig.suptitle("Evaluation convergence of Proposed wPG ActorOnly", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, fig_dir, "eval_convergence_pgboost_actoronly_6panel")

    # ActorOnly vs Refine4 overlay for deployment effect.
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0))
    for ax, (metric, title, ylabel) in zip(axes.ravel(), eval_metrics):
        for df, label in [(pg_eval_actor, "ActorOnly"), (pg_eval_refine, "Refine4")]:
            agg = aggregate_curve(df, "eval_episode", metric)
            xv = agg["eval_episode"].to_numpy(dtype=float)
            mean = smooth_series(agg["mean"].to_numpy(dtype=float), smooth)
            ax.plot(xv, mean, linewidth=1.8, label=label)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Episode", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.tick_params(axis="both", labelsize=8)
    axes.ravel()[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Evaluation convergence: ActorOnly vs deployment Refine4", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, fig_dir, "eval_compare_actoronly_vs_refine4_6panel")

def load_main_comparison(main_root: Path) -> pd.DataFrame:
    files = sorted(main_root.glob("trainseed*_eval10/detailed_results.csv"))
    if not files:
        # Support if user passes parent root such as results/main_comparison_sci_v8
        files = sorted(main_root.glob("matched_formal10_trainseed_eval10/trainseed*_eval10/detailed_results.csv"))
    if not files:
        raise FileNotFoundError(f"No detailed_results.csv found under {main_root}")
    rows = []
    for f in files:
        df = read_csv(f)
        m = re.search(r"trainseed(\d+)_eval10", str(f.parent))
        train_seed = int(m.group(1)) if m else -1
        df["train_seed"] = train_seed
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def dedupe_main_results(df: pd.DataFrame) -> pd.DataFrame:
    """Avoid treating heuristic methods repeated across trainseed dirs as 100 independent samples."""
    no_train_methods = {"Random", "Random_Refine4", "Greedy", "Greedy_Refine4"}
    parts = []
    for method, g in df.groupby("method"):
        if method in no_train_methods:
            parts.append(g.drop_duplicates(subset=["method", "seed"]).copy())
        else:
            parts.append(g.copy())
    return pd.concat(parts, ignore_index=True)


def summarize_main(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "system_cost", "avg_delay", "avg_energy", "avg_deadline_violation", "feasible_ratio",
        "ratio_mean", "ratio_std", "local_exec_ratio", "neighbor_exec_ratio",
        "decision_time_per_slot_sec", "avg_battery_violation",
        "avg_move_violation", "avg_boundary_violation", "avg_collision_violation",
        "avg_rate_violation", "avg_nan_count",
    ]
    rows = []
    for method in METHOD_ORDER:
        g = df[df["method"] == method]
        if g.empty:
            continue
        row = {"method": method, "label": METHOD_LABELS.get(method, method), "n": len(g)}
        for metric in metrics:
            arr = pd.to_numeric(g[metric], errors="coerce").to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            row[f"{metric}_mean"] = float(np.mean(arr)) if len(arr) else np.nan
            row[f"{metric}_std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            row[f"{metric}_ci95"] = float(1.96 * row[f"{metric}_std"] / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
        rows.append(row)
    summary = pd.DataFrame(rows)
    return summary


def plot_bar(ax, summary: pd.DataFrame, metric: str, title: str, ylabel: str, ci: bool = True, rotate: int = 35) -> None:
    labels = summary["label"].tolist()
    y = summary[f"{metric}_mean"].to_numpy(dtype=float)
    err_col = f"{metric}_ci95" if ci else f"{metric}_std"
    yerr = summary[err_col].to_numpy(dtype=float)
    x = np.arange(len(labels))
    ax.bar(x, y, yerr=yerr, capsize=3, linewidth=0.6, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.tick_params(axis="y", labelsize=8)


def plot_main_comparison(main_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    fig_dir = out_dir / "fig_main_comparison"
    ensure_dir(fig_dir)

    main_df2 = dedupe_main_results(main_df)
    summary = summarize_main(main_df2)
    summary.to_csv(out_dir / "main_comparison_summary_dedup.csv", index=False)
    main_df2.to_csv(out_dir / "main_comparison_detailed_dedup.csv", index=False)

    metrics4 = [
        ("system_cost", "System cost", "Cost"),
        ("avg_delay", "Average delay", "Delay"),
        ("avg_deadline_violation", "Deadline violation", "Violation"),
        ("feasible_ratio", "Feasible ratio", "Ratio"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    for ax, (metric, title, ylabel) in zip(axes.ravel(), metrics4):
        plot_bar(ax, summary, metric, title, ylabel, ci=True)
    fig.suptitle("Main comparison across baselines", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, fig_dir, "main_comparison_4panel")

    for metric, title, ylabel in metrics4 + [("avg_energy", "Average energy", "Energy"), ("decision_time_per_slot_sec", "Decision time per slot", "Seconds")]:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        plot_bar(ax, summary, metric, title, ylabel, ci=True)
        fig.tight_layout()
        save_fig(fig, fig_dir, f"main_{metric}")

    # Ablation figure.
    ablation_methods = ["Proposed_woPG_ActorOnly", "Proposed_wPG_ActorOnly", "Proposed_wPG_Refine4"]
    abl = summary[summary["method"].isin(ablation_methods)].copy()
    abl["method"] = pd.Categorical(abl["method"], categories=ablation_methods, ordered=True)
    abl = abl.sort_values("method")
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.6))
    for ax, (metric, title, ylabel) in zip(axes.ravel(), metrics4):
        plot_bar(ax, abl, metric, title, ylabel, ci=True, rotate=20)
    fig.suptitle("Ablation study of proposed framework", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, fig_dir, "ablation_proposed_4panel")

    # Local vs neighbor stacked bar.
    labels = summary["label"].tolist()
    x = np.arange(len(labels))
    local = summary["local_exec_ratio_mean"].to_numpy(dtype=float)
    neigh = summary["neighbor_exec_ratio_mean"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.bar(x, local, label="Local execution", edgecolor="black", linewidth=0.6)
    ax.bar(x, neigh, bottom=local, label="Neighbor execution", edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Execution ratio", fontsize=9)
    ax.set_title("Local/neighbor execution behavior", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    save_fig(fig, fig_dir, "execution_ratio_stacked")

    # Constraint diagnostics, excluding battery from hard feasibility interpretation.
    constraint_metrics = [
        ("avg_deadline_violation", "Deadline"),
        ("avg_rate_violation", "Rate"),
        ("avg_move_violation", "Move"),
        ("avg_boundary_violation", "Boundary"),
        ("avg_collision_violation", "Collision"),
        ("avg_nan_count", "NaN"),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    width = 0.12
    x = np.arange(len(summary))
    for i, (metric, label) in enumerate(constraint_metrics):
        ax.bar(x + (i - len(constraint_metrics)/2) * width + width/2, summary[f"{metric}_mean"], width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Violation value", fontsize=9)
    ax.set_title("Hard-constraint violation diagnostics", fontsize=10)
    ax.legend(fontsize=8, frameon=False, ncol=3)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    save_fig(fig, fig_dir, "hard_constraint_diagnostics")

    # Battery diagnostic as soft metric.
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    plot_bar(ax, summary, "avg_battery_violation", "Battery-energy diagnostic", "Soft violation", ci=True)
    fig.tight_layout()
    save_fig(fig, fig_dir, "battery_soft_diagnostic")

    return summary


def write_paper_tables(summary: pd.DataFrame, out_dir: Path) -> None:
    ensure_dir(out_dir)
    table = summary.copy()
    columns = [
        "method", "n", "system_cost_mean", "system_cost_std", "avg_delay_mean", "avg_delay_std",
        "avg_energy_mean", "avg_energy_std", "avg_deadline_violation_mean", "avg_deadline_violation_std",
        "feasible_ratio_mean", "feasible_ratio_std", "neighbor_exec_ratio_mean", "neighbor_exec_ratio_std",
        "decision_time_per_slot_sec_mean", "decision_time_per_slot_sec_std",
    ]
    table[columns].to_csv(out_dir / "paper_main_table_values.csv", index=False)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Main comparison under the post-disaster multi-UAV MEC scenario.}")
    lines.append(r"\label{tab:main_comparison}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\hline")
    lines.append(r"Method & Cost $\downarrow$ & Delay $\downarrow$ & Energy $\downarrow$ & Deadline vio. $\downarrow$ & Feasible $\uparrow$ \\")
    lines.append(r"\hline")
    for _, r in summary.iterrows():
        label = r["label"].replace("_", r"\_")
        cost = f"{r['system_cost_mean']:.2f}$\\pm${r['system_cost_std']:.2f}"
        delay = f"{r['avg_delay_mean']:.2f}$\\pm${r['avg_delay_std']:.2f}"
        energy = f"{r['avg_energy_mean']:.2f}$\\pm${r['avg_energy_std']:.2f}"
        deadline = f"{r['avg_deadline_violation_mean']:.3f}$\\pm${r['avg_deadline_violation_std']:.3f}"
        feasible = f"{r['feasible_ratio_mean']:.3f}$\\pm${r['feasible_ratio_std']:.3f}"
        lines.append(f"{label} & {cost} & {delay} & {energy} & {deadline} & {feasible} \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    (out_dir / "paper_main_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgboost-root", type=str, required=True)
    parser.add_argument("--nopg-root", type=str, required=True)
    parser.add_argument("--main-root", type=str, required=True)
    parser.add_argument("--out", type=str, default="results/paper_figures_v8")
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEEDS_DEFAULT))
    parser.add_argument("--smooth-train", type=int, default=9)
    parser.add_argument("--smooth-eval", type=int, default=3)
    parser.add_argument("--band", type=str, choices=["std", "ci95"], default="std")
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    print("[LOAD] training logs")
    pg_train, pg_eval_refine, pg_eval_actor = load_training_logs(Path(args.pgboost_root), "pgboost", seeds)
    np_train, np_eval_refine, np_eval_actor = load_training_logs(Path(args.nopg_root), "no_pg", seeds)

    pg_train.to_csv(out_dir / "merged_pgboost_train_log.csv", index=False)
    np_train.to_csv(out_dir / "merged_nopg_train_log.csv", index=False)
    pg_eval_actor.to_csv(out_dir / "merged_pgboost_eval_actoronly_log.csv", index=False)
    pg_eval_refine.to_csv(out_dir / "merged_pgboost_eval_refine_log.csv", index=False)

    print("[PLOT] training convergence")
    plot_training_convergence(pg_train, np_train, out_dir, smooth=args.smooth_train, band=args.band)

    print("[PLOT] evaluation convergence")
    plot_eval_convergence(pg_eval_actor, pg_eval_refine, out_dir, smooth=args.smooth_eval, band=args.band)

    print("[LOAD/PLOT] main comparison")
    main_df = load_main_comparison(Path(args.main_root))
    main_df.to_csv(out_dir / "main_comparison_detailed_raw.csv", index=False)
    summary = plot_main_comparison(main_df, out_dir)
    write_paper_tables(summary, out_dir)

    print("[DONE] Figures saved to:", out_dir)
    print("Key outputs:")
    print("  ", out_dir / "fig_training_convergence")
    print("  ", out_dir / "fig_eval_convergence")
    print("  ", out_dir / "fig_main_comparison")
    print("  ", out_dir / "paper_main_table_values.csv")
    print("  ", out_dir / "paper_main_table.tex")


if __name__ == "__main__":
    main()
