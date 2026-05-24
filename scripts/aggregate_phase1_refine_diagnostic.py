#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate phase1_refine_diagnostic_sci_v6.py outputs across train seeds.

Put this file at:
    ~/projects/uav_mec_sci/scripts/aggregate_phase1_refine_diagnostic.py

Run:
    python3 scripts/aggregate_phase1_refine_diagnostic.py \
      --root results/phase1_refine_diagnostic/formal10_trainseed_eval10 \
      --out results/phase1_refine_diagnostic/formal10_trainseed_eval10/final_aggregate
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

EPS = 1e-8


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None or x == "":
            return default
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def infer_train_seed(path: Path) -> int:
    m = re.search(r"trainseed(\d+)", str(path))
    if not m:
        return -1
    return int(m.group(1))


def summarize(rows: List[Dict[str, Any]], group_key: str) -> List[Dict[str, Any]]:
    if not rows:
        return []

    skip = {
        group_key, "method", "seed", "train_seed", "policy_kind", "stage2_prefix",
        "pure_prefix", "refine_enabled", "refine_max_tasks", "refine_ratio_search",
        "refine_schedule_search",
    }
    metrics: List[str] = []
    for row in rows:
        for k, v in row.items():
            if k in skip:
                continue
            if np.isfinite(safe_float(v)) and k not in metrics:
                metrics.append(k)

    groups = sorted(set(str(r[group_key]) for r in rows))
    out_rows: List[Dict[str, Any]] = []
    for g in groups:
        vals = [r for r in rows if str(r[group_key]) == g]
        out: Dict[str, Any] = {group_key: g, "n": len(vals)}
        for meta in ["policy_kind", "refine_enabled", "refine_max_tasks", "refine_ratio_search", "refine_schedule_search"]:
            uniq = sorted(set(str(v.get(meta, "")) for v in vals))
            out[meta] = uniq[0] if len(uniq) == 1 else ";".join(uniq)
        for metric in metrics:
            arr = np.asarray([safe_float(v.get(metric)) for v in vals], dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                out[f"{metric}_mean"] = float("nan")
                out[f"{metric}_std"] = float("nan")
                out[f"{metric}_ci95"] = float("nan")
            else:
                mean = float(np.mean(arr))
                std = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
                ci95 = float(1.96 * std / math.sqrt(arr.size)) if arr.size >= 2 else 0.0
                out[f"{metric}_mean"] = mean
                out[f"{metric}_std"] = std
                out[f"{metric}_ci95"] = ci95
        out_rows.append(out)

    out_rows.sort(key=lambda r: safe_float(r.get("system_cost_mean"), float("inf")))
    best = safe_float(out_rows[0].get("system_cost_mean"), float("nan")) if out_rows else float("nan")
    for row in out_rows:
        cost = safe_float(row.get("system_cost_mean"), float("nan"))
        row["cost_gap_to_best_percent"] = (cost - best) / max(abs(best), EPS) * 100.0 if np.isfinite(cost) and np.isfinite(best) else float("nan")
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    detail_paths = sorted(root.glob("trainseed*_eval*/phase1_refine_detail.csv"))
    if not detail_paths:
        raise FileNotFoundError(f"No phase1_refine_detail.csv found under {root}")

    all_detail: List[Dict[str, Any]] = []
    per_train_summary: List[Dict[str, Any]] = []

    for p in detail_paths:
        train_seed = infer_train_seed(p)
        rows = read_csv(p)
        for r in rows:
            r["train_seed"] = train_seed
        all_detail.extend(rows)

        s_rows = summarize(rows, group_key="method")
        for r in s_rows:
            r["train_seed"] = train_seed
        per_train_summary.extend(s_rows)

    final_summary = summarize(all_detail, group_key="method")

    write_csv(out_dir / "all_detail_phase1_trainseed_evalseed.csv", all_detail)
    write_csv(out_dir / "per_trainseed_phase1_summary.csv", per_train_summary)
    write_csv(out_dir / "final_phase1_summary_trainseed_evalseed_mean.csv", final_summary)

    print("Saved:")
    print(f"  {out_dir / 'all_detail_phase1_trainseed_evalseed.csv'}")
    print(f"  {out_dir / 'per_trainseed_phase1_summary.csv'}")
    print(f"  {out_dir / 'final_phase1_summary_trainseed_evalseed_mean.csv'}")

    print("\nTop methods by system_cost_mean:")
    for row in final_summary[:12]:
        print(
            f"  {row['method']:34s} cost={safe_float(row.get('system_cost_mean')):.2f} "
            f"delay={safe_float(row.get('avg_delay_mean')):.3f} "
            f"ratio={safe_float(row.get('ratio_mean_mean')):.4f} "
            f"feas={safe_float(row.get('feasible_ratio_mean')):.3f} "
            f"gap={safe_float(row.get('cost_gap_to_best_percent')):.2f}%"
        )


if __name__ == "__main__":
    main()
