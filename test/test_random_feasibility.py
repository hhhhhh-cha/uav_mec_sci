import time

import numpy as np

from env.mec_env import random_generate_state
# from env.association import build_access_association, random_generate_high_action
from env.association import build_access_association, heuristic_generate_high_action
from solver.bandwidth_solver import solve_bandwidth_allocation
# from solver.cpu_solver import solve_cpu_allocation
from solver.cpu_solver import (
    solve_cpu_allocation,
    solve_cpu_allocation_proportional,
)
from solver.feasibility_check import check_feasibility, compute_delay_and_energy

CPU_MODE = "prop"
PROP_RHO = 1.0

def run_one_trial(seed=None, verbose=False):
    """
    Run one random single-slot feasibility test.
    """
    state = random_generate_state(M=3, K=8, seed=seed)
    access_assoc = build_access_association(state)
    # high_action = random_generate_high_action(state, access_assoc, seed=seed)
    high_action = heuristic_generate_high_action(state, access_assoc, seed=seed)


    offload_ratio = high_action["offload_ratio"]
    sched_beta = high_action["sched_beta"]

    # bandwidth allocation
    t0 = time.perf_counter()
    bw_alloc = solve_bandwidth_allocation(
        state=state,
        access_assoc=access_assoc,
        offload_ratio=offload_ratio,
    )
    t1 = time.perf_counter()

    # cpu allocation
    if CPU_MODE == "kkt":
        cpu_alloc, total_bisect_iters = solve_cpu_allocation(
            state=state,
            access_assoc=access_assoc,
            sched_beta=sched_beta,
            offload_ratio=offload_ratio,
            omega1=100.0,
            omega2=1.0,
        )
    elif CPU_MODE == "prop":
        cpu_alloc = solve_cpu_allocation_proportional(
            state=state,
            access_assoc=access_assoc,
            sched_beta=sched_beta,
            offload_ratio=offload_ratio,
            rho=PROP_RHO,
        )
        total_bisect_iters = 0
    else:
        raise ValueError(f"Unknown CPU_MODE: {CPU_MODE}")
    t2 = time.perf_counter()
        # cpu_alloc, total_bisect_iters = solve_cpu_allocation(
    #     state=state,
    #     access_assoc=access_assoc,
    #     sched_beta=sched_beta,
    #     offload_ratio=offload_ratio,
    #     omega1=100.0,
    #     omega2=1.0,
    # )
    

    metrics = compute_delay_and_energy(
        state=state,
        access_assoc=access_assoc,
        offload_ratio=offload_ratio,
        sched_beta=sched_beta,
        bw_alloc=bw_alloc,
        cpu_alloc=cpu_alloc,
    )

    if verbose:
        print("\nPer-task debug:")
        for k in range(state["K"]):
            access_m = int(np.argmax(access_assoc[:, k]))
            chosen_js = np.where(sched_beta[k, access_m, :] > 0.5)[0]
            exec_j = int(chosen_js[0]) if len(chosen_js) == 1 else -1

            print(
                f"Task {k}: "
                f"assoc={access_m}, exec={exec_j}, "
                f"lambda={offload_ratio[k]:.3f}, "
                f"deadline={state['task_deadline'][k]:.3f}, "
                f"Tloc={metrics['delay_local'][k]:.3f}, "
                f"Tup={metrics['delay_up'][k]:.3f}, "
                f"Tbh={metrics['delay_bh'][k]:.3f}, "
                f"Texe={metrics['delay_exec'][k]:.3f}, "
                f"Tedge={metrics['delay_edge'][k]:.3f}, "
                f"Ttotal={metrics['delay_total'][k]:.3f}"
            )

    report = check_feasibility(
        state=state,
        access_assoc=access_assoc,
        offload_ratio=offload_ratio,
        sched_beta=sched_beta,
        bw_alloc=bw_alloc,
        cpu_alloc=cpu_alloc,
        metrics=metrics,
    )

    result = {
        "state": state,
        "ok": report["ok"],
        "report": report,
        "metrics": metrics,
        "bw_time_ms": (t1 - t0) * 1000.0,
        "cpu_time_ms": (t2 - t1) * 1000.0,
        "total_bisect_iters": total_bisect_iters,
    }

    if verbose:
        print("=" * 70)
        print("Single Random Feasibility Trial")
        print("=" * 70)
        print(f"Feasible: {result['ok']}")
        print(f"Bandwidth solver time: {result['bw_time_ms']:.4f} ms")
        print(f"CPU solver time: {result['cpu_time_ms']:.4f} ms")
        print(f"Total bisection iterations: {result['total_bisect_iters']}")
        print("Feasibility report:")
        for k, v in report.items():
            print(f"  {k}: {v}")
        print(f"System delay:  {metrics['delay_sys']:.6f}")
        print(f"System energy: {metrics['energy_sys']:.6f}")
        print("=" * 70)

    return result


def monte_carlo_test(num_trials=100, base_seed=1000):
    feasible_count = 0
    max_ratio_violation = 0.0
    max_assoc_violation = 0.0
    max_schedule_violation = 0.0
    max_candidate_violation = 0.0
    max_bw_violation = 0.0
    max_cpu_violation = 0.0
    total_nan = 0

    num_task_total = 0
    num_task_deadline_ok = 0

    total_bw_time = 0.0
    total_cpu_time = 0.0
    total_bisect_iters = 0

    total_delay_sys = 0.0
    total_energy_sys = 0.0

    for i in range(num_trials):
        result = run_one_trial(seed=base_seed + i, verbose=False)
        state = result["state"]
        report = result["report"]
        metrics = result["metrics"]

        if result["ok"]:
            feasible_count += 1

        max_ratio_violation = max(max_ratio_violation, report["ratio_violation"])
        max_assoc_violation = max(max_assoc_violation, report["assoc_violation"])
        max_schedule_violation = max(max_schedule_violation, report["schedule_violation"])
        max_candidate_violation = max(max_candidate_violation, report["candidate_violation"])
        max_bw_violation = max(max_bw_violation, report["bw_violation"])
        max_cpu_violation = max(max_cpu_violation, report["cpu_violation"])
        total_nan += report["nan_count"]

        total_bw_time += result["bw_time_ms"]
        total_cpu_time += result["cpu_time_ms"]
        total_bisect_iters += result["total_bisect_iters"]

        total_delay_sys += metrics["delay_sys"]
        total_energy_sys += metrics["energy_sys"]

        delay_total = metrics["delay_total"]
        deadline = state["task_deadline"]

        num_task_total += len(delay_total)
        num_task_deadline_ok += int(np.sum(delay_total <= deadline + 1e-8))

    summary = {
        "num_trials": num_trials,
        "feasible_ratio": feasible_count / num_trials,
        "avg_bw_time_ms": total_bw_time / num_trials,
        "avg_cpu_time_ms": total_cpu_time / num_trials,
        "avg_bisection_iters": total_bisect_iters / num_trials,
        "max_ratio_violation": max_ratio_violation,
        "max_assoc_violation": max_assoc_violation,
        "max_schedule_violation": max_schedule_violation,
        "max_candidate_violation": max_candidate_violation,
        "max_bw_violation": max_bw_violation,
        "max_cpu_violation": max_cpu_violation,
        "total_nan_count": total_nan,
        "avg_delay_sys": total_delay_sys / num_trials,
        "avg_energy_sys": total_energy_sys / num_trials,
        "task_deadline_satisfaction_ratio": num_task_deadline_ok / num_task_total,
    }

    print("\n" + "=" * 80)
    print("Monte Carlo Feasibility Test Summary")
    print("=" * 80)
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("=" * 80)

    return summary

def sweep_prop_rho(rho_list=(0.25, 0.5, 0.75, 1.0), num_trials=100, base_seed=1000):
    global CPU_MODE, PROP_RHO

    CPU_MODE = "prop"

    print("\n" + "=" * 90)
    print("Proportional CPU rho Sweep")
    print("=" * 90)

    all_results = []

    for rho in rho_list:
        PROP_RHO = float(rho)
        print(f"\n>>> Running proportional CPU with rho = {PROP_RHO:.2f}")
        summary = monte_carlo_test(num_trials=num_trials, base_seed=base_seed)

        row = {
            "rho": PROP_RHO,
            "feasible_ratio": summary["feasible_ratio"],
            "task_deadline_satisfaction_ratio": summary["task_deadline_satisfaction_ratio"],
            "avg_delay_sys": summary["avg_delay_sys"],
            "avg_energy_sys": summary["avg_energy_sys"],
            "avg_cpu_time_ms": summary["avg_cpu_time_ms"],
        }
        all_results.append(row)

    print("\n" + "=" * 90)
    print("rho Sweep Summary")
    print("=" * 90)
    for row in all_results:
        print(
            f"rho={row['rho']:.2f} | "
            f"feasible_ratio={row['feasible_ratio']:.4f} | "
            f"task_deadline_satisfaction={row['task_deadline_satisfaction_ratio']:.4f} | "
            f"avg_delay_sys={row['avg_delay_sys']:.4f} | "
            f"avg_energy_sys={row['avg_energy_sys']:.4f} | "
            f"avg_cpu_time_ms={row['avg_cpu_time_ms']:.4f}"
        )
    print("=" * 90)

    return all_results

# def main_test():
#     # one verbose run
#     run_one_trial(seed=42, verbose=True)

#     # batch test
#     monte_carlo_test(num_trials=100, base_seed=1000)
def main_test():
    sweep_prop_rho(rho_list=(0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5,
            0.6, 0.7, 0.8, 0.9, 1.0), num_trials=100, base_seed=1000)
