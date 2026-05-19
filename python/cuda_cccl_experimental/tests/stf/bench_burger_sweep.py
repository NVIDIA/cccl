# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Driver that sweeps the Burger variants across a set of problem sizes
and produces a comparison plot (ms / step vs N).

Why subprocesses?
    * clean CUDA state per run  (no stale primary context, no static stream
      pool leakage between variants, no framework cross-contamination with
      torch/numba in the same process -- see the nvJitLink note in CLAUDE)
    * trivial to parse: each test prints a single ``BENCH ...`` line that
      we grep out of stdout.

Usage:
    python bench_burger_sweep.py                # default sizes & nsteps
    python bench_burger_sweep.py --sizes 640,1280,2560,5120
    python bench_burger_sweep.py --only stf_pytorch,stf_numba
    python bench_burger_sweep.py --nsteps 60 --substeps 10
    BURGER_PLOT_OUT=burger_scaling.png python bench_burger_sweep.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Map variant name -> (test file, pytest node id)
VARIANTS: dict[str, tuple[str, str]] = {
    "optimized": (
        "test_burger_pytorch_optimized.py",
        "test_burger_pytorch_optimized",
    ),
    "optimized_hop": (
        "test_burger_pytorch_optimized.py",
        "test_burger_pytorch_optimized_hop",
    ),
    "stf_pytorch": (
        "test_burger_stackable.py",
        "test_burger",
    ),
    "stf_numba": (
        "test_burger_stackable_fast.py",
        "test_burger_fast",
    ),
}

BENCH_RE = re.compile(
    r"^BENCH\s+variant=(?P<variant>\S+)\s+N=(?P<N>\d+)\s+nsteps=(?P<nsteps>\d+)"
    r"\s+total_s=(?P<total_s>[\d.eE+-]+)\s+ms_per_step=(?P<mps>[\d.eE+-]+)"
)


def run_one(variant: str, N: int, nsteps: int, substeps: int, timeout: int = 600):
    test_file, node = VARIANTS[variant]
    # project root = cuda_cccl_experimental/ (grandparent of stf/)
    project_root = HERE.parents[1]
    node_id = f"tests/stf/{test_file}::{node}"
    env = os.environ.copy()
    env["BURGER_N"] = str(N)
    env["BURGER_NSTEPS"] = str(nsteps)
    env["BURGER_SUBSTEPS"] = str(substeps)
    env.pop("BURGER_PLOT", None)

    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", node_id, "-q", "-s"],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    wall = time.perf_counter() - t0

    stdout = proc.stdout
    stderr = proc.stderr

    match = None
    for line in stdout.splitlines():
        m = BENCH_RE.match(line.strip())
        if m and m.group("variant") == variant and int(m.group("N")) == N:
            match = m
            break

    if proc.returncode != 0 or match is None:
        print(f"  !! {variant} N={N} FAILED (exit={proc.returncode}, wall={wall:.1f}s)")
        if stdout:
            print("     stdout tail:")
            for line in stdout.splitlines()[-25:]:
                print(f"       {line}")
        if stderr:
            print("     stderr tail:")
            for line in stderr.splitlines()[-10:]:
                print(f"       {line}")
        return None

    return {
        "variant": variant,
        "N": N,
        "nsteps": int(match.group("nsteps")),
        "total_s": float(match.group("total_s")),
        "ms_per_step": float(match.group("mps")),
        "wall_s": wall,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sizes",
        default="640,1280,2560,5120,10240",
        help="comma-separated N values",
    )
    p.add_argument(
        "--only",
        default=",".join(VARIANTS.keys()),
        help=f"comma-separated subset of: {list(VARIANTS.keys())}",
    )
    p.add_argument("--nsteps", type=int, default=100)
    p.add_argument("--substeps", type=int, default=10)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--out-json", default="burger_scaling.json")
    p.add_argument(
        "--out-plot", default=os.environ.get("BURGER_PLOT_OUT", "burger_scaling.png")
    )
    p.add_argument(
        "--skip-run", action="store_true", help="only re-plot from existing JSON"
    )
    p.add_argument(
        "--merge",
        action="store_true",
        help="load existing JSON and append/overwrite points keyed by (variant, N)",
    )
    args = p.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    variants = [v.strip() for v in args.only.split(",") if v.strip()]
    for v in variants:
        if v not in VARIANTS:
            sys.exit(f"Unknown variant: {v!r}.  Choices: {list(VARIANTS)}")

    out_json = HERE / args.out_json
    results: list[dict] = []

    if args.skip_run:
        if not out_json.exists():
            sys.exit(f"--skip-run but {out_json} does not exist")
        results = json.loads(out_json.read_text())
    else:
        if args.merge and out_json.exists():
            existing = json.loads(out_json.read_text())
            print(f"Merge mode: {len(existing)} existing results in {out_json.name}")
        else:
            existing = []
        by_key = {(r["variant"], r["N"]): r for r in existing}

        print(f"Sweep: variants={variants}, sizes={sizes}, nsteps={args.nsteps}")
        for variant in variants:
            for N in sizes:
                print(f"  running {variant:<12s} N={N:<6d} ...", flush=True)
                r = run_one(variant, N, args.nsteps, args.substeps, args.timeout)
                if r is not None:
                    print(
                        f"    -> {r['ms_per_step']:.2f} ms/step  "
                        f"(total {r['total_s']:.2f}s, wall {r['wall_s']:.1f}s)"
                    )
                    by_key[(variant, N)] = r

        results = sorted(by_key.values(), key=lambda r: (r["variant"], r["N"]))
        out_json.write_text(json.dumps(results, indent=2))
        print(f"Wrote {out_json}  ({len(results)} points)")

    _plot(results, HERE / args.out_plot)


def _plot(results: list[dict], out_path: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    by_variant: dict[str, list[tuple[int, float]]] = {}
    for r in results:
        by_variant.setdefault(r["variant"], []).append((r["N"], r["ms_per_step"]))

    style = {
        "optimized": {
            "marker": "s",
            "color": "#1f77b4",
            "label": "optimized (torch.compile + K=4 sync)",
        },
        "optimized_hop": {
            "marker": "P",
            "color": "#9467bd",
            "label": "optimized + HOP while_loop CG",
        },
        "stf_pytorch": {
            "marker": "^",
            "color": "#2ca02c",
            "label": "STF + PyTorch tasks",
        },
        "stf_numba": {
            "marker": "D",
            "color": "#d62728",
            "label": "STF + Numba kernels",
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # (1) absolute ms/step
    ax = axes[0]
    for variant, pts in by_variant.items():
        pts = sorted(pts)
        xs = [n for n, _ in pts]
        ys = [ms for _, ms in pts]
        s = style.get(variant, {"marker": "x", "color": "k", "label": variant})
        ax.plot(
            xs, ys, marker=s["marker"], color=s["color"], label=s["label"], linewidth=2
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("grid size N")
    ax.set_ylabel("time per step (ms)")
    ax.set_title("Burger solver: time per step vs problem size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize="small")

    # (2) speedup relative to the optimized PyTorch variant at each N
    ax = axes[1]
    if "optimized" in by_variant:
        ref = {n: ms for n, ms in by_variant["optimized"]}
        for variant, pts in by_variant.items():
            if variant == "optimized":
                continue
            pts = sorted(pts)
            xs = []
            ys = []
            for n, ms in pts:
                if n in ref and ms > 0:
                    xs.append(n)
                    ys.append(ref[n] / ms)
            if not xs:
                continue
            s = style.get(variant, {"marker": "x", "color": "k", "label": variant})
            ax.plot(
                xs,
                ys,
                marker=s["marker"],
                color=s["color"],
                label=s["label"],
                linewidth=2,
            )
        ax.axhline(
            1.0, color="#888888", linestyle="--", linewidth=1, label="optimized (1.0x)"
        )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("grid size N")
        ax.set_ylabel("speedup vs optimized PyTorch")
        ax.set_title("Speedup over optimized PyTorch baseline")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize="small")
    else:
        ax.text(
            0.5, 0.5, "(optimized variant not in results)", ha="center", va="center"
        )
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote plot to {out_path}")

    # ASCII summary table
    print()
    print("Summary (ms/step):")
    variants_in_order = [
        v
        for v in ("optimized", "optimized_hop", "stf_pytorch", "stf_numba")
        if v in by_variant
    ]
    all_ns = sorted({n for pts in by_variant.values() for n, _ in pts})
    header = f"  {'N':>8}  " + "  ".join(f"{v:>14s}" for v in variants_in_order)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for n in all_ns:
        cells = []
        for v in variants_in_order:
            d = dict(by_variant[v])
            cells.append(f"{d[n]:>14.2f}" if n in d else f"{'-':>14s}")
        print(f"  {n:>8}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
