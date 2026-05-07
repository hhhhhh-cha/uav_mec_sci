# -*- coding: utf-8 -*-
"""
SCI-style convergence plotting for multi-seed UAV MEC experiments.

Input:
results/convergence_training/
    500_proposed_full_stage2_seed42_run1/eval_log.csv
    500_proposed_full_stage2_seed52_run1/eval_log.csv
    ...

Output:
results/convergence_outputs/YYYYMMDD_HHMMSS/
"""

from __future__ import annotations

import os
import re
import glob
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter




@dataclass
class PlotConfig:
    root_dir: str = "results/convergence_training"
    out_base_dir: str = "results/convergence_outputs"

    # 严格指定这 10 组 seed，避免误读目录里其他 seed
    expected_seeds: Tuple[int, ...] = (
        42, 52, 62, 72, 82,
        92, 102, 112, 122, 132
    )

    single_seed: int = 72

    # 是否排除某些 seed。一般不要排除，除非某组数据确实不完整或异常。
    exclude_seeds: Tuple[int, ...] = ()

    # =========================
    # Data audit settings
    # =========================
    total_episodes: int = 500
    expected_last_episode: int = 499
    expected_train_rows: int = 500
    min_eval_rows: int = 80

    # True: 发现严重问题直接停止绘图
    strict_data_check: bool = True

    # 是否画每条 seed 的浅色曲线。
    # 论文主图建议 False，只画 mean ± 95% CI。
    draw_individual_seeds: bool = False

    # 是否平滑。
    # 严格按数据建议 1；如果老师想看更平滑曲线，可以改为 3 或 5。
    smooth_window: int = 1

    # ci95 或 std
    band_mode: str = "ci95"

    # 图片尺寸
    fig_width: float = 5.2
    fig_height: float = 3.6

    dpi: int = 600

    save_png: bool = True
    save_pdf: bool = True
    save_svg: bool = True


CFG = PlotConfig()
CFG.smooth_window = 5  # MA5

# metric column, y label, file key, color
EVAL_METRICS = [
    ("episode_reward", "Evaluation Reward ($\\times 10^3$)", "reward", "#0072B2", 1e-3,
     "Eval episode reward"),
    ("avg_delay", "Average Delay (ms)", "delay", "#D55E00", 1.0,
     "Eval average delay"),
    ("avg_energy", "Average Energy (J)", "energy", "#009E73", 1.0,
     "Eval average energy"),
    ("feasible_ratio", "Feasibility Ratio", "feasible_ratio", "#CC79A7", 1.0,
     "Eval feasible ratio"),
]

# =========================
# 手动坐标轴配置
# key 对应 EVAL_METRICS 里的第三项：
# reward / delay / energy / feasible_ratio
# 不想手动控制的图，直接不写或者写 None
# =========================
# 单种子 / 多种子 坐标轴配置
# key 对应 EVAL_METRICS 的第三项：
# reward / delay / energy / feasible_ratio
# 注意：reward 已经乘了 1e-3，所以这里填的是缩放后的值
# =========================

AXIS_CONFIG = {
    "single": {
        "xlim": (0, 500),
        "xticks": [0, 100, 200, 300, 400, 500],

        # "reward": {
        #     "ylim": (-186, -182),
        #     "yticks": [-185, -184, -183, -182],
        #     "fmt": "%.0f",
        # },

        # "reward": {
        #     "ylim": (-184.3, -182.8),
        #     "yticks": [-184.2, -183.9, -183.6, -183.3, -183.0],
        #     "fmt": "%.1f",
        # },

        "reward": {
            "ylim": (-184.55, -182.8),
            "yticks": [-184.75, -184.50, -184.25, -184.00, -183.75, -183.50,-183.25,-183.00],
            "fmt": "%.2f",
        },

        # "delay": {
        #     "ylim": (138, 142),
        #     "yticks": [138.5, 139.0, 139.5, 140.0, 140.5, 141],
        #     "fmt": "%.1f",
        # },

        "delay": {
            "ylim": (138.3, 140.2),
            "yticks": [138.4, 138.8, 139.2, 139.6, 140.0],
            "fmt": "%.1f",
        },

        # "energy": {
        #     "ylim": (500, 510),
        #     "yticks": [500, 502, 504, 506, 508, 510],
        #     "fmt": "%.0f",
        # },

        "energy": {
            "ylim": (504.5, 506.2),
            "yticks": [504.5, 505.0, 505.5, 506.0],
            "fmt": "%.1f",
        },

        "feasible_ratio": {
            "ylim": (0.60, 0.85),
            "yticks": [0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
            "fmt": "%.2f",
        },
    },

    "multi": {
        "xlim": (0, 500),
        "xticks": [0, 100, 200, 300, 400, 500],

        "reward": {
            "ylim": (-232, -180),
            "yticks": [-232, -222, -212, -202, -192, -182],
            "fmt": "%.0f",
        },
        "delay": {
            "ylim": (138, 198),
            "yticks": [148, 158, 168, 178, 188],
            "fmt": "%.0f",
        },
        "energy": {
            "ylim": (488, 508),
            "yticks": [488, 492, 496, 500, 504, 508],
            "fmt": "%.0f",
        },
        "feasible_ratio": {
            "ylim": (0.60, 0.80),
            "yticks": [0.60, 0.65, 0.70, 0.75, 0.80],
            "fmt": "%.2f",
        },
    },
}

def set_academic_style() -> None:
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    plt.rcParams.update({
        "figure.dpi": CFG.dpi,
        "savefig.dpi": CFG.dpi,

        # Linux 下没有 Times New Roman 时自动降级
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Liberation Serif",
            "Nimbus Roman",
            "DejaVu Serif",
        ],

        "mathtext.fontset": "stix",
        "font.size": 10.5,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,

        "axes.linewidth": 1.0,
        "lines.linewidth": 1.2,  # 全局线宽

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",

        "axes.unicode_minus": False,
    })


def extract_seed_from_path(path: str) -> Optional[int]:
    m = re.search(r"seed(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


def find_seed_dirs(root_dir: str) -> Dict[int, str]:
    """
    Strictly find 500-episode Stage-2 seed folders.

    Expected folder:
    results/convergence_training/
        500_proposed_full_stage2_seed42_run1/
        500_proposed_full_stage2_seed52_run1/
        ...
    """
    seed_to_dir: Dict[int, str] = {}

    for seed in CFG.expected_seeds:
        folder_name = f"500_proposed_full_stage2_seed{seed}_run1"
        folder_path = os.path.join(root_dir, folder_name)

        if os.path.isdir(folder_path):
            seed_to_dir[seed] = folder_path
        else:
            print(f"[ERROR] Missing expected seed folder: {folder_path}")

    return seed_to_dir


def check_numeric_finite(df: pd.DataFrame, cols: List[str]) -> List[str]:
    problems = []

    for col in cols:
        if col not in df.columns:
            problems.append(f"missing column: {col}")
            continue

        values = pd.to_numeric(df[col], errors="coerce")

        if values.isna().any():
            problems.append(f"column {col} contains NaN or non-numeric values")

        if not np.isfinite(values.dropna().to_numpy(dtype=float)).all():
            problems.append(f"column {col} contains Inf or -Inf")

    return problems


def audit_one_seed(seed: int, seed_dir: str) -> Dict[str, object]:
    """
    Audit eval_log.csv and train_log.csv for one seed.
    Return a report row.
    """
    eval_path = os.path.join(seed_dir, "eval_log.csv")
    train_path = os.path.join(seed_dir, "train_log.csv")

    report: Dict[str, object] = {
        "seed": seed,
        "seed_dir": seed_dir,
        "eval_path": eval_path,
        "train_path": train_path,
        "status": "PASS",
        "errors": "",
        "warnings": "",
    }

    errors = []
    warnings = []

    # =========================
    # Check file existence
    # =========================
    if not os.path.exists(eval_path):
        errors.append("eval_log.csv not found")

    if not os.path.exists(train_path):
        warnings.append("train_log.csv not found")

    if errors:
        report["status"] = "FAIL"
        report["errors"] = "; ".join(errors)
        report["warnings"] = "; ".join(warnings)
        return report

    # =========================
    # Check eval_log.csv
    # =========================
    eval_df = pd.read_csv(eval_path)

    eval_required_cols = [
        "eval_episode",
        "episode_reward",
        "avg_delay",
        "avg_energy",
        "feasible_ratio",
    ]

    eval_numeric_problems = check_numeric_finite(eval_df, eval_required_cols)
    errors.extend(eval_numeric_problems)

    if "eval_episode" in eval_df.columns:
        eval_episode = pd.to_numeric(eval_df["eval_episode"], errors="coerce")

        report["eval_rows"] = len(eval_df)
        report["eval_episode_min"] = eval_episode.min()
        report["eval_episode_max"] = eval_episode.max()
        report["eval_episode_unique"] = eval_episode.nunique()

        if eval_episode.isna().any():
            errors.append("eval_episode contains NaN")

        if eval_episode.duplicated().any():
            errors.append("eval_episode contains duplicated values")

        if not eval_episode.is_monotonic_increasing:
            errors.append("eval_episode is not monotonically increasing")

        if len(eval_df) < CFG.min_eval_rows:
            errors.append(
                f"too few eval rows: {len(eval_df)} < {CFG.min_eval_rows}"
            )

        if int(eval_episode.max()) < CFG.expected_last_episode:
            errors.append(
                f"incomplete eval log: max eval_episode={int(eval_episode.max())}, "
                f"expected at least {CFG.expected_last_episode}"
            )

        if int(eval_episode.max()) > CFG.expected_last_episode:
            warnings.append(
                f"eval_episode max={int(eval_episode.max())} is larger than "
                f"expected {CFG.expected_last_episode}"
            )

    # Metric statistics
    for col in ["episode_reward", "avg_delay", "avg_energy", "feasible_ratio"]:
        if col in eval_df.columns:
            values = pd.to_numeric(eval_df[col], errors="coerce")
            report[f"{col}_min"] = values.min()
            report[f"{col}_max"] = values.max()
            report[f"{col}_last"] = values.iloc[-1]

            if values.nunique(dropna=True) <= 1:
                warnings.append(f"{col} is constant in eval_log.csv")

    # Feasible ratio range check
    if "feasible_ratio" in eval_df.columns:
        feasible = pd.to_numeric(eval_df["feasible_ratio"], errors="coerce")
        if ((feasible < 0) | (feasible > 1)).any():
            errors.append("feasible_ratio has values outside [0, 1]")

    # =========================
    # Check train_log.csv
    # =========================
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path)

        report["train_rows"] = len(train_df)

        if "episode" not in train_df.columns:
            errors.append("train_log.csv missing column: episode")
        else:
            train_episode = pd.to_numeric(train_df["episode"], errors="coerce")

            report["train_episode_min"] = train_episode.min()
            report["train_episode_max"] = train_episode.max()
            report["train_episode_unique"] = train_episode.nunique()

            if train_episode.isna().any():
                errors.append("train episode contains NaN")

            if train_episode.duplicated().any():
                errors.append("train episode contains duplicated values")

            if not train_episode.is_monotonic_increasing:
                errors.append("train episode is not monotonically increasing")

            if len(train_df) < CFG.expected_train_rows:
                errors.append(
                    f"incomplete train_log.csv: rows={len(train_df)}, "
                    f"expected {CFG.expected_train_rows}"
                )

            if int(train_episode.max()) < CFG.expected_last_episode:
                errors.append(
                    f"incomplete train episodes: max episode={int(train_episode.max())}, "
                    f"expected {CFG.expected_last_episode}"
                )

        # Common training columns, only check if they exist
        # =========================
        # Non-critical training diagnostic columns
        # =========================
        # These columns are useful for diagnostics, but they should not block
        # evaluation-curve plotting. In DRL, loss columns can be NaN in early
        # episodes because the replay buffer is not ready for gradient updates.

        train_candidate_cols = [
            "episode_reward",
            "avg_delay",
            "avg_energy",
            "avg_deadline_violation",
            "feasible_ratio",
            "avg_actor_policy_loss",
            "avg_critic_loss",
            "avg_move_sched_bc_loss",
            "avg_ratio_bc_loss",
        ]

        existing_train_cols = [c for c in train_candidate_cols if c in train_df.columns]

        for col in existing_train_cols:
            values = pd.to_numeric(train_df[col], errors="coerce")

            nan_count = int(values.isna().sum())
            inf_count = int(np.isinf(values.dropna().to_numpy(dtype=float)).sum())

            report[f"train_{col}_nan_count"] = nan_count
            report[f"train_{col}_inf_count"] = inf_count

            if nan_count > 0:
                warnings.append(
                    f"train_log.csv diagnostic column {col} contains {nan_count} NaN values"
                )

            if inf_count > 0:
                warnings.append(
                    f"train_log.csv diagnostic column {col} contains {inf_count} Inf values"
                )

        existing_train_cols = [c for c in train_candidate_cols if c in train_df.columns]
        # train_numeric_problems = check_numeric_finite(train_df, existing_train_cols)
        # errors.extend([f"train_log.csv: {p}" for p in train_numeric_problems])

    # =========================
    # Final status
    # =========================
    if errors:
        report["status"] = "FAIL"
    elif warnings:
        report["status"] = "WARN"
    else:
        report["status"] = "PASS"

    report["errors"] = "; ".join(errors)
    report["warnings"] = "; ".join(warnings)

    return report


def audit_all_logs(out_dir: str) -> pd.DataFrame:
    """
    Audit all expected seeds before plotting.
    Save report to data_audit_report.csv.
    """
    print("\n" + "=" * 100)
    print("Data integrity audit")
    print("=" * 100)

    seed_to_dir = find_seed_dirs(CFG.root_dir)
    excluded = set(CFG.exclude_seeds)

    reports = []

    for seed in CFG.expected_seeds:
        if seed in excluded:
            reports.append({
                "seed": seed,
                "status": "SKIP",
                "errors": "",
                "warnings": "manually excluded",
            })
            print(f"[SKIP] seed={seed} manually excluded")
            continue

        if seed not in seed_to_dir:
            reports.append({
                "seed": seed,
                "status": "FAIL",
                "errors": "expected seed folder not found",
                "warnings": "",
            })
            print(f"[FAIL] seed={seed}: folder not found")
            continue

        report = audit_one_seed(seed, seed_to_dir[seed])
        reports.append(report)

        status = report["status"]
        eval_rows = report.get("eval_rows", "NA")
        eval_min = report.get("eval_episode_min", "NA")
        eval_max = report.get("eval_episode_max", "NA")
        train_rows = report.get("train_rows", "NA")

        print(
            f"[{status}] seed={seed:<4d} "
            f"eval_rows={eval_rows:<4} "
            f"eval_episode={eval_min}->{eval_max} "
            f"train_rows={train_rows}"
        )

        if report["errors"]:
            print(f"       errors  : {report['errors']}")
        if report["warnings"]:
            print(f"       warnings: {report['warnings']}")

    audit_df = pd.DataFrame(reports)

    os.makedirs(out_dir, exist_ok=True)
    audit_path = os.path.join(out_dir, "data_audit_report.csv")
    audit_df.to_csv(audit_path, index=False, encoding="utf-8-sig")

    print("-" * 100)
    print(f"[SAVE] Data audit report: {audit_path}")

    fail_df = audit_df[audit_df["status"] == "FAIL"]
    warn_df = audit_df[audit_df["status"] == "WARN"]

    print(f"[SUMMARY] PASS = {(audit_df['status'] == 'PASS').sum()}")
    print(f"[SUMMARY] WARN = {len(warn_df)}")
    print(f"[SUMMARY] FAIL = {len(fail_df)}")

    if CFG.strict_data_check and len(fail_df) > 0:
        raise RuntimeError(
            "Data audit failed. Stop plotting to avoid generating incorrect paper figures. "
            f"Please check: {audit_path}"
        )

    print("=" * 100)

    return audit_df

def load_eval_logs() -> Dict[int, pd.DataFrame]:
    seed_to_dir = find_seed_dirs(CFG.root_dir)

    expected = set(CFG.expected_seeds)
    excluded = set(CFG.exclude_seeds)

    logs: Dict[int, pd.DataFrame] = {}

    print("\n[INFO] Discovered seed folders:")
    for seed, d in sorted(seed_to_dir.items()):
        print(f"  seed={seed:<4d}  path={d}")

    print("\n[INFO] Loading selected seeds:")

    for seed in CFG.expected_seeds:
        if seed in excluded:
            print(f"  [SKIP] seed={seed} is excluded.")
            continue

        if seed not in seed_to_dir:
            print(f"  [WARN] seed={seed} folder not found.")
            continue

        csv_path = os.path.join(seed_to_dir[seed], "eval_log.csv")

        if not os.path.exists(csv_path):
            print(f"  [WARN] seed={seed} eval_log.csv not found.")
            continue

        df = pd.read_csv(csv_path)

        required_cols = ["eval_episode"] + [m[0] for m in EVAL_METRICS]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Seed {seed} missing columns: {missing}")

        df = df[required_cols].copy()
        df = df.sort_values("eval_episode").reset_index(drop=True)

        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        before = len(df)
        df = df.dropna(subset=required_cols)
        after = len(df)

        if after < before:
            print(f"  [WARN] seed={seed}: dropped {before - after} rows with NaN.")

        logs[seed] = df

        print(
            f"  [LOAD] seed={seed:<4d} rows={len(df):<4d} "
            f"episode={df['eval_episode'].min()} -> {df['eval_episode'].max()}"
        )

    if not logs:
        raise RuntimeError("No valid eval_log.csv loaded.")

    return logs


def print_data_quality_report(logs: Dict[int, pd.DataFrame]) -> None:
    print("\n" + "=" * 90)
    print("Data quality report")
    print("=" * 90)

    for metric, ylabel, key, color, scale, title in EVAL_METRICS:
        print(f"\n[{metric}]")
        for seed, df in sorted(logs.items()):
            y = df[metric].to_numpy(dtype=float)
            print(
                f"  seed={seed:<4d} "
                f"min={np.nanmin(y):>12.4f} "
                f"max={np.nanmax(y):>12.4f} "
                f"last={y[-1]:>12.4f}"
            )

    print("=" * 90)


def smooth_series(y: np.ndarray, window: int) -> np.ndarray:
    if window is None or window <= 1:
        return y.astype(float)

    return (
        pd.Series(y.astype(float))
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def build_metric_table(
    logs: Dict[int, pd.DataFrame],
    metric: str,
    scale: float,
) -> pd.DataFrame:
    series_list = []

    for seed, df in sorted(logs.items()):
        x = df["eval_episode"].to_numpy(dtype=float)
        y = df[metric].to_numpy(dtype=float) * scale
        y = smooth_series(y, CFG.smooth_window)

        s = pd.Series(y, index=x, name=seed)
        series_list.append(s)

    table = pd.concat(series_list, axis=1).sort_index()

    # 如果不同 seed 的 eval_episode 不完全一致，按 episode 插值对齐。
    table = table.interpolate(method="index", limit_direction="both")

    return table


def calc_band(values: np.ndarray, mode: str):
    mean = np.nanmean(values, axis=1)
    std = np.nanstd(values, axis=1, ddof=1)
    n = np.sum(~np.isnan(values), axis=1)

    if mode.lower() == "std":
        half_width = std
    elif mode.lower() == "ci95":
        # n=10 时 t_critical 约为 2.262
        # n=11 时约为 2.228，这里为了稳定统一采用近似。
        t_critical = np.where(n <= 10, 2.262, 2.228)
        half_width = t_critical * std / np.sqrt(np.maximum(n, 1))
    else:
        raise ValueError("band_mode must be 'ci95' or 'std'.")

    lower = mean - half_width
    upper = mean + half_width

    return mean, lower, upper


# def apply_axis_style(ax: plt.Axes, key: str) -> None:
#     ax.set_xlabel("Episode")

#     ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
#     ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

#     if key == "feasible_ratio":
#         ax.set_ylim(-0.02, 1.02)
#         ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

#     ax.grid(True, which="major", linestyle="--", linewidth=0.65, alpha=0.35)
#     ax.grid(True, which="minor", linestyle=":", linewidth=0.45, alpha=0.18)

#     # minor ticks 只显示刻度，不显示过多标签
#     ax.minorticks_on()

#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)

#     ax.tick_params(axis="both", which="major", direction="in", length=4.5, width=0.9)
#     ax.tick_params(axis="both", which="minor", direction="in", length=2.5, width=0.7)


def apply_axis_style(ax: plt.Axes, key: str, plot_type: str) -> None:
    ax.set_xlabel("Episode")

    cfg = AXIS_CONFIG.get(plot_type, {})
    y_cfg = cfg.get(key, {})

    # ========== X 轴 ==========
    if cfg.get("xlim") is not None:
        ax.set_xlim(*cfg["xlim"])

    if cfg.get("xticks") is not None:
        ax.set_xticks(cfg["xticks"])
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))

    # ========== Y 轴刻度 ==========
    if y_cfg.get("yticks") is not None:
        ax.set_yticks(y_cfg["yticks"])
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    if y_cfg.get("fmt") is not None:
        ax.yaxis.set_major_formatter(FormatStrFormatter(y_cfg["fmt"]))
    elif key == "feasible_ratio":
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    ax.grid(True, which="major", linestyle="--", linewidth=0.65, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.45, alpha=0.18)

    ax.minorticks_on()

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

    ax.tick_params(axis="both", which="major", direction="in", length=4.5, width=0.9)
    ax.tick_params(axis="both", which="minor", direction="in", length=2.5, width=0.7)

# def set_data_ylim(ax: plt.Axes, y_arrays: List[np.ndarray], key: str) -> None:
#     if key == "feasible_ratio":
#         return

#     y_all = np.concatenate([np.asarray(y).ravel() for y in y_arrays])
#     y_all = y_all[np.isfinite(y_all)]

#     if len(y_all) == 0:
#         return

#     ymin = float(np.min(y_all))
#     ymax = float(np.max(y_all))

#     if np.isclose(ymin, ymax):
#         pad = max(abs(ymax) * 0.02, 1.0)
#     else:
#         pad = 0.08 * (ymax - ymin)

#     ax.set_ylim(ymin - pad, ymax + pad)



def set_data_ylim(
    ax: plt.Axes,
    y_arrays: List[np.ndarray],
    key: str,
    plot_type: str,
) -> None:
    cfg = AXIS_CONFIG.get(plot_type, {})
    y_cfg = cfg.get(key, {})

    # 如果手动指定 ylim，优先使用手动范围
    if y_cfg.get("ylim") is not None:
        ax.set_ylim(*y_cfg["ylim"])
        return

    # feasible_ratio 没有手动指定时，默认 0~1
    if key == "feasible_ratio":
        ax.set_ylim(-0.02, 1.02)
        return

    # 其他指标默认根据数据自动设置
    y_all = np.concatenate([np.asarray(y).ravel() for y in y_arrays])
    y_all = y_all[np.isfinite(y_all)]

    if len(y_all) == 0:
        return

    ymin = float(np.min(y_all))
    ymax = float(np.max(y_all))

    if np.isclose(ymin, ymax):
        pad = max(abs(ymax) * 0.02, 1.0)
    else:
        pad = 0.08 * (ymax - ymin)

    ax.set_ylim(ymin - pad, ymax + pad)


def save_figure(fig: plt.Figure, out_dir: str, filename: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    if CFG.save_png:
        path = os.path.join(out_dir, filename + ".png")
        fig.savefig(path, bbox_inches="tight", dpi=CFG.dpi)
        print(f"[SAVE] {path}")

    if CFG.save_pdf:
        path = os.path.join(out_dir, filename + ".pdf")
        fig.savefig(path, bbox_inches="tight")
        print(f"[SAVE] {path}")

    if CFG.save_svg:
        path = os.path.join(out_dir, filename + ".svg")
        fig.savefig(path, bbox_inches="tight")
        print(f"[SAVE] {path}")


# def plot_single_seed(logs: Dict[int, pd.DataFrame], out_dir: str) -> None:
#     if CFG.single_seed not in logs:
#         raise ValueError(
#             f"single_seed={CFG.single_seed} not loaded. "
#             f"Available seeds: {sorted(logs.keys())}"
#         )

#     df = logs[CFG.single_seed]

#     for metric, ylabel, key, color, scale, title in EVAL_METRICS:
#         x = df["eval_episode"].to_numpy(dtype=float)
#         y = df[metric].to_numpy(dtype=float) * scale
#         y_plot = smooth_series(y, CFG.smooth_window)

#         fig, ax = plt.subplots(figsize=(CFG.fig_width, CFG.fig_height))

#         ax.plot(
#             x,
#             y_plot,
#             color=color,
#             linewidth=2.4,
#             label=f"Seed {CFG.single_seed}",
#             zorder=3,
#         )

#         ax.scatter(
#             x,
#             y,
#             s=15,
#             color=color,
#             alpha=0.35,
#             linewidths=0,
#             label="Evaluation points",
#             zorder=2,
#         )

#         ax.set_ylabel(ylabel)
#         apply_axis_style(ax, key, plot_type="single")
#         set_data_ylim(ax, [y_plot, y], key, plot_type="single")

#         ax.legend(frameon=False, loc="best")

#         filename = f"single_seed{CFG.single_seed}_{key}"
#         save_figure(fig, out_dir, filename)
#         plt.close(fig)

def plot_single_seed(logs: Dict[int, pd.DataFrame], out_dir: str) -> None:
    if CFG.single_seed not in logs:
        raise ValueError(f"single_seed={CFG.single_seed} not loaded.")

    df = logs[CFG.single_seed]

    # 循环画每个指标
    for metric, ylabel, key, color, scale, title in EVAL_METRICS:
        x = df["eval_episode"].to_numpy(dtype=float)
        y_raw = df[metric].to_numpy(dtype=float) * scale
        y_ma = smooth_series(y_raw, CFG.smooth_window)

        fig, ax = plt.subplots(figsize=(CFG.fig_width, CFG.fig_height))

        # raw 连续曲线
        ax.plot(
            x,
            y_raw,
            color="#999999",
            linewidth=1.0,
            linestyle="--",
            label="Raw curve",
            zorder=2,
        )

        # MA5 平滑曲线
        ax.plot(
            x,
            y_ma,
            color=color,
            linewidth=1.6,
            linestyle="-",
            label=f"MA{CFG.smooth_window} curve",
            zorder=3,
        )

        ax.set_ylabel(ylabel)
        ax.set_title(f"Seed {CFG.single_seed}: {title} (raw + MA{CFG.smooth_window})")
        apply_axis_style(ax, key, plot_type="single")
        set_data_ylim(ax, [y_raw, y_ma], key, plot_type="single")

        ax.legend(frameon=False, loc="best")

        filename = f"single_seed{CFG.single_seed}_{key}_raw_ma"
        save_figure(fig, out_dir, filename)
        plt.close(fig)


def plot_multi_seed(logs: Dict[int, pd.DataFrame], out_dir: str) -> None:
    seeds = sorted(logs.keys())

    for metric, ylabel, key, color, scale, title in EVAL_METRICS:
        table = build_metric_table(logs, metric, scale)
        x = table.index.to_numpy(dtype=float)
        values = table.to_numpy(dtype=float)

        mean, lower, upper = calc_band(values, CFG.band_mode)

        fig, ax = plt.subplots(figsize=(CFG.fig_width, CFG.fig_height))

        if CFG.draw_individual_seeds:
            for seed in seeds:
                ax.plot(
                    x,
                    table[seed].to_numpy(dtype=float),
                    color=color,
                    linewidth=0.8,
                    alpha=0.18,
                    zorder=1,
                )

        band_label = "95% CI" if CFG.band_mode.lower() == "ci95" else "Std."

        ax.fill_between(
            x,
            lower,
            upper,
            color=color,
            alpha=0.20,
            linewidth=0,
            label=band_label,
            zorder=2,
        )

        # ax.plot(
        #     x,
        #     mean,
        #     color=color,
        #     linewidth=2.6,
        #     label=f"Mean over {len(seeds)} seeds",
        #     zorder=3,
        # )
        ax.plot(
            x,
            mean,
            color=color,
            linewidth=1.8,
            label=f"Mean MA{CFG.smooth_window} over {len(seeds)} seeds",
            zorder=3,
        )

        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} over {len(seeds)} seeds")
        apply_axis_style(ax, key, plot_type="multi")
        set_data_ylim(ax, [lower, upper, mean], key, plot_type="multi")

        ax.legend(frameon=False, loc="best")

        filename = f"multi_seed_{key}_{CFG.band_mode}"
        save_figure(fig, out_dir, filename)
        plt.close(fig)


def main() -> None:
    set_academic_style()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(CFG.out_base_dir, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 90)
    print("SCI-style convergence plotting with data audit")
    print("=" * 90)
    print(f"Root dir        : {CFG.root_dir}")
    print(f"Output dir      : {out_dir}")
    print(f"Expected seeds  : {CFG.expected_seeds}")
    print(f"Excluded seeds  : {CFG.exclude_seeds}")
    print(f"Single seed     : {CFG.single_seed}")
    print(f"Band mode       : {CFG.band_mode}")
    print(f"Smooth window   : {CFG.smooth_window}")
    print(f"Strict check    : {CFG.strict_data_check}")
    print("=" * 90)

    # 1. 强制数据审计
    audit_all_logs(out_dir)

    # 2. 审计通过后再加载 eval_log
    logs = load_eval_logs()

    print(f"\n[INFO] Actually loaded seeds: {sorted(logs.keys())}")
    print(f"[INFO] Number of loaded seeds: {len(logs)}")

    # 3. 再次打印统计范围，方便人工核对
    print_data_quality_report(logs)

    # 4. 画图
    plot_single_seed(logs, out_dir)
    plot_multi_seed(logs, out_dir)

    print("\n[DONE] All figures have been generated.")
    print(f"[DONE] Saved to: {out_dir}")

if __name__ == "__main__":
    main()