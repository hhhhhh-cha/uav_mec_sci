##总汇总脚本，把 10 个 train seed 的 summary 合成最终表。
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import glob
import pandas as pd
import numpy as np

# ROOT = "results/main_comparison_sci_v6/formal10_trainseed_eval10"
# OUT = "results/main_comparison_sci_v6/formal10_trainseed_eval10/final_aggregate"
ROOT = "results/main_comparison_sci_v6/formal10_trainseed_eval10_with_pure_refine"
OUT = "results/main_comparison_sci_v6/formal10_trainseed_eval10_with_pure_refine/final_aggregate"

TRAIN_SEEDS = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132]

METRICS = [
    "system_cost",
    "avg_delay",
    "avg_energy",
    "avg_deadline_violation",
    "feasible_ratio",
    "avg_ratio_violation",
    "avg_assoc_violation",
    "avg_schedule_violation",
    "avg_candidate_violation",
    "avg_bw_violation",
    "avg_cpu_violation",
    "avg_rate_violation",
    "avg_nan_count",
    "avg_move_violation",
    "avg_boundary_violation",
    "avg_collision_violation",
    "avg_battery_violation",
    "ratio_mean",
    "ratio_std",
    "local_exec_ratio",
    "neighbor_exec_ratio",
    "decision_time_per_slot_sec",
]

os.makedirs(OUT, exist_ok=True)

rows = []

for train_seed in TRAIN_SEEDS:
    path = os.path.join(ROOT, f"trainseed{train_seed}_eval10", "main_comparison_summary.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing summary file: {path}")

    df = pd.read_csv(path)
    for _, row in df.iterrows():
        item = {
            "train_seed": train_seed,
            "method": row["method"],
        }

        # Each row here is already averaged over 10 evaluation seeds.
        for metric in METRICS:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            ci_col = f"{metric}_ci95"

            if mean_col in row.index:
                item[metric] = row[mean_col]
            if std_col in row.index:
                item[f"{metric}_eval_std"] = row[std_col]
            if ci_col in row.index:
                item[f"{metric}_eval_ci95"] = row[ci_col]

        rows.append(item)

per_trainseed = pd.DataFrame(rows)
per_trainseed.to_csv(os.path.join(OUT, "per_trainseed_summary.csv"), index=False)

summary_rows = []
for method, g in per_trainseed.groupby("method"):
    out = {
        "method": method,
        "n_train_seeds": len(g),
    }

    for metric in METRICS:
        if metric not in g.columns:
            continue

        vals = pd.to_numeric(g[metric], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            out[f"{metric}_mean"] = np.nan
            out[f"{metric}_std_trainseed"] = np.nan
            out[f"{metric}_ci95_trainseed"] = np.nan
            continue

        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if vals.size >= 2 else 0.0
        ci95 = float(1.96 * std / math.sqrt(vals.size)) if vals.size >= 2 else 0.0

        out[f"{metric}_mean"] = mean
        out[f"{metric}_std_trainseed"] = std
        out[f"{metric}_ci95_trainseed"] = ci95

    summary_rows.append(out)

final_summary = pd.DataFrame(summary_rows)
final_summary = final_summary.sort_values("system_cost_mean", ascending=True)

best_cost = float(final_summary["system_cost_mean"].iloc[0])
final_summary["cost_gap_to_best_percent"] = (
    (final_summary["system_cost_mean"] - best_cost) / max(abs(best_cost), 1e-8) * 100.0
)

final_summary.to_csv(os.path.join(OUT, "final_main_comparison_summary_trainseed_mean.csv"), index=False)

# Also merge all detail rows for audit.
detail_rows = []
for train_seed in TRAIN_SEEDS:
    path = os.path.join(ROOT, f"trainseed{train_seed}_eval10", "main_comparison_detail.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing detail file: {path}")
    df = pd.read_csv(path)
    df.insert(0, "train_seed", train_seed)
    detail_rows.append(df)

all_detail = pd.concat(detail_rows, ignore_index=True)
all_detail.to_csv(os.path.join(OUT, "all_detail_10trainseed_10evalseed.csv"), index=False)

print("=" * 120)
print("Final formal main comparison summary")
print("=" * 120)
cols = [
    "method",
    "system_cost_mean",
    "system_cost_std_trainseed",
    "avg_delay_mean",
    "avg_delay_std_trainseed",
    "avg_deadline_violation_mean",
    "feasible_ratio_mean",
    "cost_gap_to_best_percent",
]
print(final_summary[cols].to_string(index=False))
print("=" * 120)
print("Saved:")
print(os.path.join(OUT, "per_trainseed_summary.csv"))
print(os.path.join(OUT, "final_main_comparison_summary_trainseed_mean.csv"))
print(os.path.join(OUT, "all_detail_10trainseed_10evalseed.csv"))