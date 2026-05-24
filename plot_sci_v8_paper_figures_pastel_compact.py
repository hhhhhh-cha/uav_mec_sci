#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compact SCI-style plotting script for the final MRD-TMADDPG figures.
It generates only the paper-needed figures to avoid excessive float clutter and slow SVG/PDF generation.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

SEEDS_DEFAULT = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132]
OUTPUT_FORMATS = ["png", "pdf"]

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

# Pastel palette inspired by the user's reference screenshot.
PALETTE = {
    "cream": "#F8F5CB",
    "olive": "#DFDB9B",
    "sky": "#BDEFF5",
    "peach": "#F0B790",
    "pink": "#F5BDBE",
    "orange": "#F1BD73",
    "teal": "#61A49D",
    "purple": "#BC9CCB",
    "red": "#D36B6B",
    "gray": "#B9B9B9",
    "dark": "#3A3A3A",
    "grid": "#E7E7E7",
}

METHOD_COLORS = {
    "Random": PALETTE["gray"],
    "Greedy": PALETTE["olive"],
    "Greedy_Refine4": PALETTE["cream"],
    "Pure_MADDPG": PALETTE["peach"],
    "Pure_MADDPG_Refine4": PALETTE["pink"],
    "Proposed_woPG_ActorOnly": PALETTE["sky"],
    "Proposed_wPG_ActorOnly": PALETTE["teal"],
    "Proposed_wPG_Refine4": PALETTE["red"],
}

LINE_COLORS = {
    "wPG": PALETTE["teal"],
    "woPG": PALETTE["peach"],
    "actor": PALETTE["teal"],
    "refine": PALETTE["red"],
}


def short_label(label: str) -> str:
    return {
        "Greedy+Refine4": "Greedy\n+Refine4",
        "Pure MADDPG": "Pure\nMADDPG",
        "Pure MADDPG+Refine4": "Pure MADDPG\n+Refine4",
        "Proposed w/o PG": "Proposed\nw/o PG",
        "Proposed ActorOnly": "Proposed\nActorOnly",
        "Proposed+Refine4": "Proposed\n+Refine4",
    }.get(label, label)


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.size": 9,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def axes_style(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(True, color=PALETTE["grid"], linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#666666")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(axis="both", colors="#333333", width=0.8, length=3)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_fig(fig, out_dir: Path, name: str) -> None:
    ensure_dir(out_dir)
    for ext in OUTPUT_FORMATS:
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in df.columns:
        if c not in {"method", "pgboost_prefix", "nopg_prefix", "pure_prefix"}:
            try:
                df[c] = pd.to_numeric(df[c])
            except Exception:
                pass
    return df


def parse_seed_from_name(name: str) -> int | None:
    m = re.search(r"seed(\d+)", name)
    return int(m.group(1)) if m else None


def find_run_dirs(root: Path, seeds: Iterable[int]) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "train_log.csv").exists():
            s = parse_seed_from_name(d.name)
            if s in seeds:
                out[s] = d
    missing = [s for s in seeds if s not in out]
    if missing:
        raise FileNotFoundError(f"Missing run dirs for seeds {missing} under {root}")
    return out


def load_training_logs(root: Path, variant: str, seeds: List[int]):
    train_list, eval_list, actor_list = [], [], []
    for s, d in sorted(find_run_dirs(root, seeds).items()):
        tr = read_csv(d / "train_log.csv"); tr["seed"] = s; tr["variant"] = variant
        ev = read_csv(d / "eval_log.csv"); ev["seed"] = s; ev["variant"] = variant
        ao = read_csv(d / "eval_actoronly_log.csv"); ao["seed"] = s; ao["variant"] = variant
        train_list.append(tr); eval_list.append(ev); actor_list.append(ao)
    return pd.concat(train_list, ignore_index=True), pd.concat(eval_list, ignore_index=True), pd.concat(actor_list, ignore_index=True)


def aggregate_curve(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    tmp = df[[x, y]].copy()
    tmp[y] = pd.to_numeric(tmp[y], errors="coerce")
    g = tmp.groupby(x, as_index=False)[y].agg(["mean", "std", "count"]).reset_index()
    g["std"] = g["std"].fillna(0.0)
    g["ci95"] = 1.96 * g["std"] / np.sqrt(g["count"].clip(lower=1))
    return g


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    return pd.Series(y, dtype="float64").rolling(window=window, min_periods=1).mean().to_numpy()


def curve(ax, df, x, y, title, ylabel, smooth_win, band="std", color=None, label=None, fill=True):
    color = color or PALETTE["teal"]
    agg = aggregate_curve(df, x, y)
    xv = agg[x].to_numpy(float)
    mean = smooth(agg["mean"].to_numpy(float), smooth_win)
    spread = smooth(agg["ci95" if band == "ci95" else "std"].to_numpy(float), smooth_win)
    ax.plot(xv, mean, color=color, linewidth=2.15, label=label)
    if fill:
        ax.fill_between(xv, mean - spread, mean + spread, color=color, alpha=0.16, linewidth=0)
    ax.set_title(title, pad=6)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    axes_style(ax)


def load_main_comparison(main_root: Path) -> pd.DataFrame:
    files = sorted(main_root.glob("trainseed*_eval10/detailed_results.csv"))
    if not files:
        files = sorted(main_root.glob("matched_formal10_trainseed_eval10/trainseed*_eval10/detailed_results.csv"))
    if not files:
        raise FileNotFoundError(f"No detailed_results.csv under {main_root}")
    parts = []
    for f in files:
        df = read_csv(f)
        m = re.search(r"trainseed(\d+)_eval10", str(f.parent))
        df["train_seed"] = int(m.group(1)) if m else -1
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def dedupe_main(df: pd.DataFrame) -> pd.DataFrame:
    no_train = {"Random", "Random_Refine4", "Greedy", "Greedy_Refine4"}
    parts = []
    for method, g in df.groupby("method"):
        if method in no_train:
            parts.append(g.drop_duplicates(subset=["method", "seed"]).copy())
        else:
            parts.append(g.copy())
    return pd.concat(parts, ignore_index=True)


def summarize_main(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "system_cost", "avg_delay", "avg_energy", "avg_deadline_violation", "feasible_ratio",
        "ratio_mean", "ratio_std", "local_exec_ratio", "neighbor_exec_ratio", "decision_time_per_slot_sec",
        "avg_battery_violation", "avg_move_violation", "avg_boundary_violation", "avg_collision_violation",
        "avg_rate_violation", "avg_nan_count",
    ]
    rows = []
    for method in METHOD_ORDER:
        g = df[df["method"] == method]
        if g.empty:
            continue
        row = {"method": method, "label": METHOD_LABELS.get(method, method), "n": len(g)}
        for metric in metrics:
            arr = pd.to_numeric(g[metric], errors="coerce").to_numpy(float)
            arr = arr[np.isfinite(arr)]
            row[f"{metric}_mean"] = float(np.mean(arr)) if len(arr) else np.nan
            row[f"{metric}_std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            row[f"{metric}_ci95"] = float(1.96 * row[f"{metric}_std"] / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def bar_panel(ax, summary: pd.DataFrame, metric: str, title: str, ylabel: str, ci=True, rotate=22):
    methods = summary["method"].tolist()
    labels = [short_label(x) for x in summary["label"].tolist()]
    x = np.arange(len(labels))
    y = summary[f"{metric}_mean"].to_numpy(float)
    e = summary[f"{metric}_{'ci95' if ci else 'std'}"].to_numpy(float)
    colors = [METHOD_COLORS.get(m, PALETTE["gray"]) for m in methods]
    ax.bar(x, y, yerr=e, capsize=3, color=colors, edgecolor=PALETTE["dark"], linewidth=0.75,
           error_kw={"elinewidth": 1.0, "ecolor": PALETTE["dark"], "capthick": 1.0})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="right")
    ax.set_title(title, pad=6)
    ax.set_ylabel(ylabel)
    axes_style(ax)
    ax.grid(True, axis="y", color=PALETTE["grid"], linewidth=0.7, alpha=0.85)
    ax.grid(False, axis="x")


def ablation_singlecol_label(method: str) -> str:
    """Very short labels for IEEE single-column ablation figures."""
    return {
        "Proposed_woPG_ActorOnly": "w/o\nPG",
        "Proposed_wPG_ActorOnly": "Actor\nOnly",
        "Proposed_wPG_Refine4": "+Refine4",
    }.get(method, METHOD_LABELS.get(method, method))


def execution_singlecol_label(method: str) -> str:
    """Compact labels for the single-column execution-ratio plot."""
    return {
        "Random": "Random",
        "Greedy": "Greedy",
        "Greedy_Refine4": "G+R4",
        "Pure_MADDPG": "Pure",
        "Pure_MADDPG_Refine4": "Pure+R4",
        "Proposed_woPG_ActorOnly": "w/o PG",
        "Proposed_wPG_ActorOnly": "ActorOnly",
        "Proposed_wPG_Refine4": "+R4",
    }.get(method, METHOD_LABELS.get(method, method))


def bar_panel_singlecol(ax, summary: pd.DataFrame, metric: str, title: str, ylabel: str, ci=True):
    """A small but readable bar panel designed for one IEEE column."""
    methods = summary["method"].astype(str).tolist()
    labels = [ablation_singlecol_label(m) for m in methods]
    x = np.arange(len(labels))
    y = summary[f"{metric}_mean"].to_numpy(float)
    e = summary[f"{metric}_{'ci95' if ci else 'std'}"].to_numpy(float)

    # Use 10^3 scaling only in the single-column ablation plot to avoid crowded y-ticks.
    if metric == "system_cost":
        y = y / 1000.0
        e = e / 1000.0
        ylabel = r"Cost ($10^3$)"

    colors = [METHOD_COLORS.get(m, PALETTE["gray"]) for m in methods]
    ax.bar(
        x, y, yerr=e, width=0.62, capsize=2.2,
        color=colors, edgecolor=PALETTE["dark"], linewidth=0.55,
        error_kw={"elinewidth": 0.75, "ecolor": PALETTE["dark"], "capthick": 0.75},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_title(title, pad=3, fontsize=7.6)
    ax.set_ylabel(ylabel, fontsize=7.0, labelpad=1.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis="x", labelsize=6.2, pad=1.5)
    ax.tick_params(axis="y", labelsize=6.2, pad=1.5)
    axes_style(ax)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.65)
    ax.grid(True, axis="y", color=PALETTE["grid"], linewidth=0.55, alpha=0.85)
    ax.grid(False, axis="x")


def plot_ablation_singlecol(abl: pd.DataFrame, out_dir: Path) -> None:
    """Generate Fig. 5 as a true single-column figure, not a compressed wide figure."""
    metrics4 = [
        ("system_cost", "System cost", "Cost"),
        ("avg_delay", "Average delay", "Delay"),
        ("avg_deadline_violation", "Deadline violation", "Violation"),
        ("feasible_ratio", "Feasible ratio", "Ratio"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(3.55, 3.75))
    for ax, (m, t, y) in zip(axes.ravel(), metrics4):
        bar_panel_singlecol(ax, abl, m, t, y)
    fig.subplots_adjust(left=0.145, right=0.985, bottom=0.115, top=0.93, wspace=0.52, hspace=0.66)
    save_fig(fig, out_dir, "ablation_proposed_singlecol")


def plot_execution_ratio_singlecol(summary: pd.DataFrame, out_dir: Path) -> None:
    """Generate Fig. 6 as a readable one-column horizontal stacked-bar plot."""
    methods = summary["method"].astype(str).tolist()
    labels = [execution_singlecol_label(m) for m in methods]
    local = summary["local_exec_ratio_mean"].to_numpy(float)
    neigh = summary["neighbor_exec_ratio_mean"].to_numpy(float)

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(3.55, 2.55))
    ax.barh(
        y, local, label="Local", color="#DCE3E2",
        edgecolor=PALETTE["dark"], linewidth=0.55,
    )
    ax.barh(
        y, neigh, left=local, label="Neighbor", color=PALETTE["teal"],
        edgecolor=PALETTE["dark"], linewidth=0.55,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.7)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Execution ratio", fontsize=7.2, labelpad=2)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis="x", labelsize=6.6, pad=1.5)
    ax.tick_params(axis="y", labelsize=6.6, pad=1.5)
    axes_style(ax)
    ax.grid(True, axis="x", color=PALETTE["grid"], linewidth=0.55, alpha=0.85)
    ax.grid(False, axis="y")
    # ax.legend(
    #     frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01),
    #     fontsize=6.8, handlelength=1.4, columnspacing=1.1, borderaxespad=0.0,
    # )
    # fig.subplots_adjust(left=0.245, right=0.985, bottom=0.16, top=0.88)
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=2,
        fontsize=8.2,
        handlelength=1.8,
        columnspacing=1.6,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.24, top=0.86)
    save_fig(fig, out_dir, "execution_ratio_singlecol")


def write_table(summary: pd.DataFrame, out_dir: Path) -> None:
    cols = [
        "method", "n", "system_cost_mean", "system_cost_std", "avg_delay_mean", "avg_delay_std",
        "avg_energy_mean", "avg_energy_std", "avg_deadline_violation_mean", "avg_deadline_violation_std",
        "feasible_ratio_mean", "feasible_ratio_std", "neighbor_exec_ratio_mean", "neighbor_exec_ratio_std",
        "decision_time_per_slot_sec_mean", "decision_time_per_slot_sec_std",
    ]
    summary[cols].to_csv(out_dir / "paper_main_table_values.csv", index=False)


def plot_all(pg_train, np_train, pg_eval_refine, pg_eval_actor, summary, out_dir, smooth_train, smooth_eval, band):
    train_dir = out_dir / "fig_training_convergence"
    eval_dir = out_dir / "fig_eval_convergence"
    main_dir = out_dir / "fig_main_comparison"
    ensure_dir(train_dir); ensure_dir(eval_dir); ensure_dir(main_dir)

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
    fig, axes = plt.subplots(2, 4, figsize=(14.2, 6.4))
    for ax, (m, t, y) in zip(axes.ravel(), train_metrics):
        curve(ax, pg_train, "episode", m, t, y, smooth_train, band, color=LINE_COLORS["wPG"])
    fig.tight_layout()
    save_fig(fig, train_dir, "training_convergence_pgboost_8panel")

    eval_metrics = [
        ("system_cost", "System cost", "Cost"),
        ("avg_delay", "Average delay", "Delay"),
        ("feasible_ratio", "Feasible ratio", "Ratio"),
        ("avg_deadline_violation", "Deadline violation", "Violation"),
        ("ratio_mean", "Offloading ratio mean", "Ratio"),
        ("neighbor_exec_ratio", "Neighbor execution ratio", "Ratio"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 6.2))
    for ax, (m, t, y) in zip(axes.ravel(), eval_metrics):
        curve(ax, pg_eval_actor, "eval_episode", m, t, y, smooth_eval, band, color=LINE_COLORS["actor"])
    fig.tight_layout()
    save_fig(fig, eval_dir, "eval_convergence_pgboost_actoronly_6panel")

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.0))
    metrics4 = [
        ("system_cost", "System cost", "Cost"),
        ("avg_delay", "Average delay", "Delay"),
        ("avg_deadline_violation", "Deadline violation", "Violation"),
        ("feasible_ratio", "Feasible ratio", "Ratio"),
    ]
    for ax, (m, t, y) in zip(axes.ravel(), metrics4):
        bar_panel(ax, summary, m, t, y)
    fig.tight_layout()
    save_fig(fig, main_dir, "main_comparison_4panel")

    abl_order = ["Proposed_woPG_ActorOnly", "Proposed_wPG_ActorOnly", "Proposed_wPG_Refine4"]
    abl = summary[summary["method"].isin(abl_order)].copy()
    abl["method"] = pd.Categorical(abl["method"], categories=abl_order, ordered=True)
    abl = abl.sort_values("method")

    # Backup wide version. Use this only if the figure is placed across two columns.
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.8))
    for ax, (m, t, y) in zip(axes.ravel(), metrics4):
        bar_panel(ax, abl, m, t, y, rotate=18)
    fig.tight_layout()
    save_fig(fig, main_dir, "ablation_proposed_4panel")

    # Paper-use single-column version for Fig. 5.
    # This is NOT a compressed copy of the wide figure: labels, font sizes, and axis scaling are redesigned.
    plot_ablation_singlecol(abl, main_dir)

    # Paper-use single-column version for Fig. 6.
    # Horizontal stacking avoids crowded rotated method names in a one-column layout.
    plot_execution_ratio_singlecol(summary, main_dir)

    # Backup wide version. Use this only if Fig. 6 is placed across two columns.
    labels = [short_label(x) for x in summary["label"].tolist()]
    x = np.arange(len(labels))
    local = summary["local_exec_ratio_mean"].to_numpy(float)
    neigh = summary["neighbor_exec_ratio_mean"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.bar(x, local, label="Local execution", color="#DCE3E2", edgecolor=PALETTE["dark"], linewidth=0.75)
    ax.bar(x, neigh, bottom=local, label="Neighbor execution", color=PALETTE["teal"], edgecolor=PALETTE["dark"], linewidth=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Execution ratio")
    ax.legend(frameon=False, loc="upper right")
    axes_style(ax)
    ax.grid(True, axis="y", color=PALETTE["grid"], linewidth=0.7, alpha=0.85)
    ax.grid(False, axis="x")
    fig.tight_layout()
    save_fig(fig, main_dir, "execution_ratio_stacked")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgboost-root", required=True)
    parser.add_argument("--nopg-root", required=True)
    parser.add_argument("--main-root", required=True)
    parser.add_argument("--out", default="results/paper_figures_v8_pastel")
    parser.add_argument("--seeds", default=",".join(str(x) for x in SEEDS_DEFAULT))
    parser.add_argument("--smooth-train", type=int, default=21)
    parser.add_argument("--smooth-eval", type=int, default=5)
    parser.add_argument("--band", choices=["std", "ci95"], default="std")
    parser.add_argument("--formats", default="png,pdf")
    args = parser.parse_args()

    global OUTPUT_FORMATS
    OUTPUT_FORMATS = [x.strip().lower() for x in args.formats.split(",") if x.strip()]
    setup_style()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    print("[LOAD] training logs")
    pg_train, pg_eval_refine, pg_eval_actor = load_training_logs(Path(args.pgboost_root), "pgboost", seeds)
    np_train, _, _ = load_training_logs(Path(args.nopg_root), "no_pg", seeds)

    print("[LOAD] main comparison")
    main_df = load_main_comparison(Path(args.main_root))
    main_df2 = dedupe_main(main_df)
    summary = summarize_main(main_df2)

    pg_train.to_csv(out_dir / "merged_pgboost_train_log.csv", index=False)
    pg_eval_actor.to_csv(out_dir / "merged_pgboost_eval_actoronly_log.csv", index=False)
    main_df2.to_csv(out_dir / "main_comparison_detailed_dedup.csv", index=False)
    summary.to_csv(out_dir / "main_comparison_summary_dedup.csv", index=False)
    write_table(summary, out_dir)

    print("[PLOT] compact SCI figures")
    plot_all(pg_train, np_train, pg_eval_refine, pg_eval_actor, summary, out_dir, args.smooth_train, args.smooth_eval, args.band)

    print("[DONE] saved to", out_dir)

if __name__ == "__main__":
    main()



# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Compact SCI-style plotting script for the final MRD-TMADDPG figures.
# It generates only the paper-needed figures to avoid excessive float clutter and slow SVG/PDF generation.
# """
# from __future__ import annotations

# import argparse
# import math
# import re
# from pathlib import Path
# from typing import Dict, Iterable, List

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator

# SEEDS_DEFAULT = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132]
# OUTPUT_FORMATS = ["png", "pdf"]

# METHOD_ORDER = [
#     "Random",
#     "Greedy",
#     "Greedy_Refine4",
#     "Pure_MADDPG",
#     "Pure_MADDPG_Refine4",
#     "Proposed_woPG_ActorOnly",
#     "Proposed_wPG_ActorOnly",
#     "Proposed_wPG_Refine4",
# ]

# METHOD_LABELS = {
#     "Random": "Random",
#     "Greedy": "Greedy",
#     "Greedy_Refine4": "Greedy+Refine4",
#     "Pure_MADDPG": "Pure MADDPG",
#     "Pure_MADDPG_Refine4": "Pure MADDPG+Refine4",
#     "Proposed_woPG_ActorOnly": "Proposed w/o PG",
#     "Proposed_wPG_ActorOnly": "Proposed ActorOnly",
#     "Proposed_wPG_Refine4": "Proposed+Refine4",
# }

# # Pastel palette inspired by the user's reference screenshot.
# PALETTE = {
#     "cream": "#F8F5CB",
#     "olive": "#DFDB9B",
#     "sky": "#BDEFF5",
#     "peach": "#F0B790",
#     "pink": "#F5BDBE",
#     "orange": "#F1BD73",
#     "teal": "#61A49D",
#     "purple": "#BC9CCB",
#     "red": "#D36B6B",
#     "gray": "#B9B9B9",
#     "dark": "#3A3A3A",
#     "grid": "#E7E7E7",
# }

# METHOD_COLORS = {
#     "Random": PALETTE["gray"],
#     "Greedy": PALETTE["olive"],
#     "Greedy_Refine4": PALETTE["cream"],
#     "Pure_MADDPG": PALETTE["peach"],
#     "Pure_MADDPG_Refine4": PALETTE["pink"],
#     "Proposed_woPG_ActorOnly": PALETTE["sky"],
#     "Proposed_wPG_ActorOnly": PALETTE["teal"],
#     "Proposed_wPG_Refine4": PALETTE["red"],
# }

# LINE_COLORS = {
#     "wPG": PALETTE["teal"],
#     "woPG": PALETTE["peach"],
#     "actor": PALETTE["teal"],
#     "refine": PALETTE["red"],
# }


# def short_label(label: str) -> str:
#     return {
#         "Greedy+Refine4": "Greedy\n+Refine4",
#         "Pure MADDPG": "Pure\nMADDPG",
#         "Pure MADDPG+Refine4": "Pure MADDPG\n+Refine4",
#         "Proposed w/o PG": "Proposed\nw/o PG",
#         "Proposed ActorOnly": "Proposed\nActorOnly",
#         "Proposed+Refine4": "Proposed\n+Refine4",
#     }.get(label, label)


# def setup_style() -> None:
#     plt.rcParams.update({
#         "figure.dpi": 160,
#         "savefig.dpi": 300,
#         "font.size": 9,
#         "font.family": "DejaVu Sans",
#         "axes.titlesize": 10,
#         "axes.labelsize": 9,
#         "legend.fontsize": 8,
#         "xtick.labelsize": 8,
#         "ytick.labelsize": 8,
#         "axes.linewidth": 0.8,
#         "pdf.fonttype": 42,
#         "ps.fonttype": 42,
#         "figure.facecolor": "white",
#         "axes.facecolor": "white",
#     })


# def axes_style(ax) -> None:
#     ax.set_facecolor("white")
#     ax.grid(True, color=PALETTE["grid"], linewidth=0.7, alpha=0.85)
#     ax.set_axisbelow(True)
#     for side in ["top", "right"]:
#         ax.spines[side].set_visible(False)
#     for side in ["left", "bottom"]:
#         ax.spines[side].set_color("#666666")
#         ax.spines[side].set_linewidth(0.8)
#     ax.tick_params(axis="both", colors="#333333", width=0.8, length=3)


# def ensure_dir(path: Path) -> None:
#     path.mkdir(parents=True, exist_ok=True)


# def save_fig(fig, out_dir: Path, name: str) -> None:
#     ensure_dir(out_dir)
#     for ext in OUTPUT_FORMATS:
#         fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight")
#     plt.close(fig)


# def read_csv(path: Path) -> pd.DataFrame:
#     df = pd.read_csv(path)
#     for c in df.columns:
#         if c not in {"method", "pgboost_prefix", "nopg_prefix", "pure_prefix"}:
#             try:
#                 df[c] = pd.to_numeric(df[c])
#             except Exception:
#                 pass
#     return df


# def parse_seed_from_name(name: str) -> int | None:
#     m = re.search(r"seed(\d+)", name)
#     return int(m.group(1)) if m else None


# def find_run_dirs(root: Path, seeds: Iterable[int]) -> Dict[int, Path]:
#     out: Dict[int, Path] = {}
#     for d in sorted(root.iterdir()):
#         if d.is_dir() and (d / "train_log.csv").exists():
#             s = parse_seed_from_name(d.name)
#             if s in seeds:
#                 out[s] = d
#     missing = [s for s in seeds if s not in out]
#     if missing:
#         raise FileNotFoundError(f"Missing run dirs for seeds {missing} under {root}")
#     return out


# def load_training_logs(root: Path, variant: str, seeds: List[int]):
#     train_list, eval_list, actor_list = [], [], []
#     for s, d in sorted(find_run_dirs(root, seeds).items()):
#         tr = read_csv(d / "train_log.csv"); tr["seed"] = s; tr["variant"] = variant
#         ev = read_csv(d / "eval_log.csv"); ev["seed"] = s; ev["variant"] = variant
#         ao = read_csv(d / "eval_actoronly_log.csv"); ao["seed"] = s; ao["variant"] = variant
#         train_list.append(tr); eval_list.append(ev); actor_list.append(ao)
#     return pd.concat(train_list, ignore_index=True), pd.concat(eval_list, ignore_index=True), pd.concat(actor_list, ignore_index=True)


# def aggregate_curve(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
#     tmp = df[[x, y]].copy()
#     tmp[y] = pd.to_numeric(tmp[y], errors="coerce")
#     g = tmp.groupby(x, as_index=False)[y].agg(["mean", "std", "count"]).reset_index()
#     g["std"] = g["std"].fillna(0.0)
#     g["ci95"] = 1.96 * g["std"] / np.sqrt(g["count"].clip(lower=1))
#     return g


# def smooth(y: np.ndarray, window: int) -> np.ndarray:
#     if window <= 1:
#         return y
#     return pd.Series(y, dtype="float64").rolling(window=window, min_periods=1).mean().to_numpy()


# def curve(ax, df, x, y, title, ylabel, smooth_win, band="std", color=None, label=None, fill=True):
#     color = color or PALETTE["teal"]
#     agg = aggregate_curve(df, x, y)
#     xv = agg[x].to_numpy(float)
#     mean = smooth(agg["mean"].to_numpy(float), smooth_win)
#     spread = smooth(agg["ci95" if band == "ci95" else "std"].to_numpy(float), smooth_win)
#     ax.plot(xv, mean, color=color, linewidth=2.15, label=label)
#     if fill:
#         ax.fill_between(xv, mean - spread, mean + spread, color=color, alpha=0.16, linewidth=0)
#     ax.set_title(title, pad=6)
#     ax.set_xlabel("Episode")
#     ax.set_ylabel(ylabel)
#     ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
#     axes_style(ax)


# def load_main_comparison(main_root: Path) -> pd.DataFrame:
#     files = sorted(main_root.glob("trainseed*_eval10/detailed_results.csv"))
#     if not files:
#         files = sorted(main_root.glob("matched_formal10_trainseed_eval10/trainseed*_eval10/detailed_results.csv"))
#     if not files:
#         raise FileNotFoundError(f"No detailed_results.csv under {main_root}")
#     parts = []
#     for f in files:
#         df = read_csv(f)
#         m = re.search(r"trainseed(\d+)_eval10", str(f.parent))
#         df["train_seed"] = int(m.group(1)) if m else -1
#         parts.append(df)
#     return pd.concat(parts, ignore_index=True)


# def dedupe_main(df: pd.DataFrame) -> pd.DataFrame:
#     no_train = {"Random", "Random_Refine4", "Greedy", "Greedy_Refine4"}
#     parts = []
#     for method, g in df.groupby("method"):
#         if method in no_train:
#             parts.append(g.drop_duplicates(subset=["method", "seed"]).copy())
#         else:
#             parts.append(g.copy())
#     return pd.concat(parts, ignore_index=True)


# def summarize_main(df: pd.DataFrame) -> pd.DataFrame:
#     metrics = [
#         "system_cost", "avg_delay", "avg_energy", "avg_deadline_violation", "feasible_ratio",
#         "ratio_mean", "ratio_std", "local_exec_ratio", "neighbor_exec_ratio", "decision_time_per_slot_sec",
#         "avg_battery_violation", "avg_move_violation", "avg_boundary_violation", "avg_collision_violation",
#         "avg_rate_violation", "avg_nan_count",
#     ]
#     rows = []
#     for method in METHOD_ORDER:
#         g = df[df["method"] == method]
#         if g.empty:
#             continue
#         row = {"method": method, "label": METHOD_LABELS.get(method, method), "n": len(g)}
#         for metric in metrics:
#             arr = pd.to_numeric(g[metric], errors="coerce").to_numpy(float)
#             arr = arr[np.isfinite(arr)]
#             row[f"{metric}_mean"] = float(np.mean(arr)) if len(arr) else np.nan
#             row[f"{metric}_std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
#             row[f"{metric}_ci95"] = float(1.96 * row[f"{metric}_std"] / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
#         rows.append(row)
#     return pd.DataFrame(rows)


# def bar_panel(ax, summary: pd.DataFrame, metric: str, title: str, ylabel: str, ci=True, rotate=22):
#     methods = summary["method"].tolist()
#     labels = [short_label(x) for x in summary["label"].tolist()]
#     x = np.arange(len(labels))
#     y = summary[f"{metric}_mean"].to_numpy(float)
#     e = summary[f"{metric}_{'ci95' if ci else 'std'}"].to_numpy(float)
#     colors = [METHOD_COLORS.get(m, PALETTE["gray"]) for m in methods]
#     ax.bar(x, y, yerr=e, capsize=3, color=colors, edgecolor=PALETTE["dark"], linewidth=0.75,
#            error_kw={"elinewidth": 1.0, "ecolor": PALETTE["dark"], "capthick": 1.0})
#     ax.set_xticks(x)
#     ax.set_xticklabels(labels, rotation=rotate, ha="right")
#     ax.set_title(title, pad=6)
#     ax.set_ylabel(ylabel)
#     axes_style(ax)
#     ax.grid(True, axis="y", color=PALETTE["grid"], linewidth=0.7, alpha=0.85)
#     ax.grid(False, axis="x")


# def write_table(summary: pd.DataFrame, out_dir: Path) -> None:
#     cols = [
#         "method", "n", "system_cost_mean", "system_cost_std", "avg_delay_mean", "avg_delay_std",
#         "avg_energy_mean", "avg_energy_std", "avg_deadline_violation_mean", "avg_deadline_violation_std",
#         "feasible_ratio_mean", "feasible_ratio_std", "neighbor_exec_ratio_mean", "neighbor_exec_ratio_std",
#         "decision_time_per_slot_sec_mean", "decision_time_per_slot_sec_std",
#     ]
#     summary[cols].to_csv(out_dir / "paper_main_table_values.csv", index=False)


# def plot_all(pg_train, np_train, pg_eval_refine, pg_eval_actor, summary, out_dir, smooth_train, smooth_eval, band):
#     train_dir = out_dir / "fig_training_convergence"
#     eval_dir = out_dir / "fig_eval_convergence"
#     main_dir = out_dir / "fig_main_comparison"
#     ensure_dir(train_dir); ensure_dir(eval_dir); ensure_dir(main_dir)

#     train_metrics = [
#         ("episode_reward", "Training reward", "Episode reward"),
#         ("avg_critic_loss", "Critic loss", "Loss"),
#         ("avg_td_abs_error", "TD absolute error", "TD error"),
#         ("avg_total_actor_loss", "Actor total loss", "Loss"),
#         ("avg_refined_ratio_bc_loss", "Ratio distillation loss", "Loss"),
#         ("avg_ratio_target_delta", "Ratio target delta", "Abs. delta"),
#         ("avg_refined_sched_ce_loss", "Scheduling CE loss", "Loss"),
#         ("avg_refined_sched_acc", "Scheduling accuracy", "Accuracy"),
#     ]
#     fig, axes = plt.subplots(2, 4, figsize=(14.2, 6.4))
#     for ax, (m, t, y) in zip(axes.ravel(), train_metrics):
#         curve(ax, pg_train, "episode", m, t, y, smooth_train, band, color=LINE_COLORS["wPG"])
#     fig.tight_layout()
#     save_fig(fig, train_dir, "training_convergence_pgboost_8panel")

#     eval_metrics = [
#         ("system_cost", "System cost", "Cost"),
#         ("avg_delay", "Average delay", "Delay"),
#         ("feasible_ratio", "Feasible ratio", "Ratio"),
#         ("avg_deadline_violation", "Deadline violation", "Violation"),
#         ("ratio_mean", "Offloading ratio mean", "Ratio"),
#         ("neighbor_exec_ratio", "Neighbor execution ratio", "Ratio"),
#     ]
#     fig, axes = plt.subplots(2, 3, figsize=(12.4, 6.2))
#     for ax, (m, t, y) in zip(axes.ravel(), eval_metrics):
#         curve(ax, pg_eval_actor, "eval_episode", m, t, y, smooth_eval, band, color=LINE_COLORS["actor"])
#     fig.tight_layout()
#     save_fig(fig, eval_dir, "eval_convergence_pgboost_actoronly_6panel")

#     fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.0))
#     metrics4 = [
#         ("system_cost", "System cost", "Cost"),
#         ("avg_delay", "Average delay", "Delay"),
#         ("avg_deadline_violation", "Deadline violation", "Violation"),
#         ("feasible_ratio", "Feasible ratio", "Ratio"),
#     ]
#     for ax, (m, t, y) in zip(axes.ravel(), metrics4):
#         bar_panel(ax, summary, m, t, y)
#     fig.tight_layout()
#     save_fig(fig, main_dir, "main_comparison_4panel")

#     abl_order = ["Proposed_woPG_ActorOnly", "Proposed_wPG_ActorOnly", "Proposed_wPG_Refine4"]
#     abl = summary[summary["method"].isin(abl_order)].copy()
#     abl["method"] = pd.Categorical(abl["method"], categories=abl_order, ordered=True)
#     abl = abl.sort_values("method")
#     fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.8))
#     for ax, (m, t, y) in zip(axes.ravel(), metrics4):
#         bar_panel(ax, abl, m, t, y, rotate=18)
#     fig.tight_layout()
#     save_fig(fig, main_dir, "ablation_proposed_4panel")

#     labels = [short_label(x) for x in summary["label"].tolist()]
#     x = np.arange(len(labels))
#     local = summary["local_exec_ratio_mean"].to_numpy(float)
#     neigh = summary["neighbor_exec_ratio_mean"].to_numpy(float)
#     fig, ax = plt.subplots(figsize=(8.6, 4.4))
#     ax.bar(x, local, label="Local execution", color="#DCE3E2", edgecolor=PALETTE["dark"], linewidth=0.75)
#     ax.bar(x, neigh, bottom=local, label="Neighbor execution", color=PALETTE["teal"], edgecolor=PALETTE["dark"], linewidth=0.75)
#     ax.set_xticks(x)
#     ax.set_xticklabels(labels, rotation=25, ha="right")
#     ax.set_ylim(0, 1.0)
#     ax.set_ylabel("Execution ratio")
#     ax.legend(frameon=False, loc="upper right")
#     axes_style(ax)
#     ax.grid(True, axis="y", color=PALETTE["grid"], linewidth=0.7, alpha=0.85)
#     ax.grid(False, axis="x")
#     fig.tight_layout()
#     save_fig(fig, main_dir, "execution_ratio_stacked")


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--pgboost-root", required=True)
#     parser.add_argument("--nopg-root", required=True)
#     parser.add_argument("--main-root", required=True)
#     parser.add_argument("--out", default="results/paper_figures_v8_pastel")
#     parser.add_argument("--seeds", default=",".join(str(x) for x in SEEDS_DEFAULT))
#     parser.add_argument("--smooth-train", type=int, default=21)
#     parser.add_argument("--smooth-eval", type=int, default=5)
#     parser.add_argument("--band", choices=["std", "ci95"], default="std")
#     parser.add_argument("--formats", default="png,pdf")
#     args = parser.parse_args()

#     global OUTPUT_FORMATS
#     OUTPUT_FORMATS = [x.strip().lower() for x in args.formats.split(",") if x.strip()]
#     setup_style()

#     seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
#     out_dir = Path(args.out)
#     ensure_dir(out_dir)

#     print("[LOAD] training logs")
#     pg_train, pg_eval_refine, pg_eval_actor = load_training_logs(Path(args.pgboost_root), "pgboost", seeds)
#     np_train, _, _ = load_training_logs(Path(args.nopg_root), "no_pg", seeds)

#     print("[LOAD] main comparison")
#     main_df = load_main_comparison(Path(args.main_root))
#     main_df2 = dedupe_main(main_df)
#     summary = summarize_main(main_df2)

#     pg_train.to_csv(out_dir / "merged_pgboost_train_log.csv", index=False)
#     pg_eval_actor.to_csv(out_dir / "merged_pgboost_eval_actoronly_log.csv", index=False)
#     main_df2.to_csv(out_dir / "main_comparison_detailed_dedup.csv", index=False)
#     summary.to_csv(out_dir / "main_comparison_summary_dedup.csv", index=False)
#     write_table(summary, out_dir)

#     print("[PLOT] compact SCI figures")
#     plot_all(pg_train, np_train, pg_eval_refine, pg_eval_actor, summary, out_dir, args.smooth_train, args.smooth_eval, args.band)

#     print("[DONE] saved to", out_dir)

# if __name__ == "__main__":
#     main()
