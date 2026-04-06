"""
Pilot script: run ALNS with different k values for 50 000 iterations
and plot convergence curves side-by-side to determine a good iteration
budget for the full k-scan.

Usage:
    cd src && python pilot_convergence.py
"""

import numpy as np
import time
import json
import sys
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, ".")

_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

from data_loader import load_data
from q1b_construction import construct_solution
from q1c_alns import alns_lahc
from solution import evaluate


# ── Configuration ────────────────────────────────────────────────────────
PILOT_K_VALUES = [80, 100, 120, 140, 170]   # 5 representative k values
MAX_ITER       = 50_000
SEED           = 42


def _pilot_worker(args):
    """Run one ALNS trial for a given k. Top-level for pickling."""
    k, seed, max_iter = args
    from data_loader import load_data
    from q1b_construction import construct_solution
    from q1c_alns import alns_lahc
    from solution import evaluate

    data = load_data()
    rng_c = np.random.default_rng(seed)
    sol_init = construct_solution(data, k=k, rng=rng_c)
    init_cost = evaluate(sol_init, data)

    rng_alns = np.random.default_rng(seed + 10000)
    best, best_cost, history, _k_hist, _w_hist, stats = alns_lahc(
        sol_init, data, max_iter=max_iter, rng=rng_alns, verbose=False,
    )
    final_cost = evaluate(best, data)

    return {
        "k": k,
        "init_cost": init_cost.total,
        "final_cost": final_cost.total,
        "history": history.tolist(),   # full curve
        "time": stats["time"],
        "n_stations": best.n_stations,
        "improvement": (init_cost.total - final_cost.total)
                       / init_cost.total * 100,
    }


def main():
    print("Pilot: ALNS Convergence Test")
    print("=" * 60)
    print(f"  k values:   {PILOT_K_VALUES}")
    print(f"  Iterations: {MAX_ITER}")
    print(f"  Seed:       {SEED}")
    print(f"  Tasks:      {len(PILOT_K_VALUES)}  (parallel)")
    print()

    args_list = [(k, SEED, MAX_ITER) for k in PILOT_K_VALUES]

    t0 = time.time()
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(_pilot_worker, args_list))
    elapsed = time.time() - t0

    # ── Print table ──────────────────────────────────────────────────
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"\n  {'k':>5} {'Init Cost':>12} {'Final Cost':>12} "
          f"{'Improv':>8} {'Time':>8} {'Stations':>8}")
    print("  " + "-" * 60)
    for r in results:
        print(f"  {r['k']:>5} £{r['init_cost']:>10,.0f} "
              f"£{r['final_cost']:>10,.0f} "
              f"{r['improvement']:>7.2f}% "
              f"{r['time']:>7.1f}s "
              f"{r['n_stations']:>8}")

    # ── Save raw data ────────────────────────────────────────────────
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_data = {}
    for r in results:
        save_data[f"k={r['k']}"] = r
    with open(_RESULTS_DIR / "pilot_convergence.json", "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nRaw data  → {_RESULTS_DIR / 'pilot_convergence.json'}")

    # ── Plot ─────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 10),
                             gridspec_kw={"height_ratios": [3, 1]})

    # --- Top: full convergence curves ---
    ax = axes[0]
    cmap = plt.cm.viridis
    colors = [cmap(i / (len(results) - 1)) for i in range(len(results))]

    for r, c in zip(results, colors):
        h = np.array(r["history"])
        ax.plot(h, color=c, linewidth=1.2, label=f"k={r['k']}")

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Best Cost (£)", fontsize=12)
    ax.set_title("ALNS Convergence by Station Count k  "
                 f"(1072 robots, {MAX_ITER} iters)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # --- Bottom: tail zoom (last 60%) to see fine differences ---
    ax2 = axes[1]
    tail_start = int(MAX_ITER * 0.2)     # from 20% onward
    for r, c in zip(results, colors):
        h = np.array(r["history"])
        iters = np.arange(len(h))
        ax2.plot(iters[tail_start:], h[tail_start:],
                 color=c, linewidth=1.2, label=f"k={r['k']}")

    ax2.set_xlabel("Iteration", fontsize=12)
    ax2.set_ylabel("Best Cost (£)", fontsize=12)
    ax2.set_title("Tail Zoom (iteration 10 000+)", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = str(_RESULTS_DIR / "pilot_convergence.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure    → {_RESULTS_DIR / 'pilot_convergence.png'}")

    # ── Convergence analysis: find "elbow" per k ─────────────────────
    print(f"\n{'=' * 60}")
    print("Convergence Analysis")
    print("=" * 60)
    print(f"  Checking where 95% / 99% of total improvement is reached:\n")
    print(f"  {'k':>5} {'95% at iter':>12} {'99% at iter':>12} "
          f"{'Total Improv':>14}")
    print("  " + "-" * 50)

    for r in results:
        h = np.array(r["history"])
        total_drop = h[0] - h[-1]
        if total_drop <= 0:
            print(f"  {r['k']:>5} {'N/A':>12} {'N/A':>12} "
                  f"{'No improvement':>14}")
            continue

        target_95 = h[0] - 0.95 * total_drop
        target_99 = h[0] - 0.99 * total_drop

        idx_95 = int(np.argmax(h <= target_95))
        idx_99 = int(np.argmax(h <= target_99))

        # If never reached, set to max
        if h[idx_95] > target_95:
            idx_95 = len(h) - 1
        if h[idx_99] > target_99:
            idx_99 = len(h) - 1

        print(f"  {r['k']:>5} {idx_95:>12,} {idx_99:>12,} "
              f"£{total_drop:>12,.0f}")

    print(f"\n  Recommendation: set MAX_ITER_SCAN to the largest "
          f"'99% iter' value above,")
    print(f"  rounded up to a convenient number.")


if __name__ == "__main__":
    main()
