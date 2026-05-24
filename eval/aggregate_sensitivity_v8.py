#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate v8 sensitivity experiments from folders containing detailed_results.csv."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group-cols", default="deadline_scale,method",
                    help="Comma-separated grouping columns, e.g. deadline_scale,method or model_refine_max_tasks,method")
    args = ap.parse_args()

    files = sorted(Path(args.root).glob("**/detailed_results.csv"))
    if not files:
        raise FileNotFoundError(f"No detailed_results.csv found under {args.root}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]

    metrics = [
        "system_cost", "avg_delay", "avg_energy", "avg_deadline_violation",
        "feasible_ratio", "neighbor_exec_ratio", "decision_time_per_slot_sec",
    ]
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {c: v for c, v in zip(group_cols, keys)}
        row["n"] = len(g)
        for m in metrics:
            arr = pd.to_numeric(g[m], errors="coerce").dropna().to_numpy(float)
            row[f"{m}_mean"] = float(np.mean(arr)) if len(arr) else np.nan
            row[f"{m}_std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            row[f"{m}_ci95"] = float(1.96 * row[f"{m}_std"] / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(group_cols).to_csv(out, index=False)
    print("[DONE]", out)

if __name__ == "__main__":
    main()


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """Aggregate v8 sensitivity experiments from folders containing detailed_results.csv."""
# from __future__ import annotations

# import argparse
# from pathlib import Path
# import pandas as pd
# import numpy as np

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--root", required=True)
#     ap.add_argument("--out", required=True)
#     ap.add_argument("--group-cols", default="deadline_scale,method",
#                     help="Comma-separated grouping columns, e.g. deadline_scale,method or model_refine_max_tasks,method")
#     args = ap.parse_args()

#     files = sorted(Path(args.root).glob("**/detailed_results.csv"))
#     if not files:
#         raise FileNotFoundError(f"No detailed_results.csv found under {args.root}")

#     df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
#     group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]

#     metrics = [
#         "system_cost", "avg_delay", "avg_energy", "avg_deadline_violation",
#         "feasible_ratio", "neighbor_exec_ratio", "decision_time_per_slot_sec",
#     ]
#     rows = []
#     for keys, g in df.groupby(group_cols, dropna=False):
#         if not isinstance(keys, tuple):
#             keys = (keys,)
#         row = {c: v for c, v in zip(group_cols, keys)}
#         row["n"] = len(g)
#         for m in metrics:
#             arr = pd.to_numeric(g[m], errors="coerce").dropna().to_numpy(float)
#             row[f"{m}_mean"] = float(np.mean(arr)) if len(arr) else np.nan
#             row[f"{m}_std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
#             row[f"{m}_ci95"] = float(1.96 * row[f"{m}_std"] / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
#         rows.append(row)

#     out = Path(args.out)
#     out.parent.mkdir(parents=True, exist_ok=True)
#     pd.DataFrame(rows).sort_values(group_cols).to_csv(out, index=False)
#     print("[DONE]", out)

# if __name__ == "__main__":
#     main()
