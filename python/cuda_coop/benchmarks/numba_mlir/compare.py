# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tabulate Python/C++ GPU-time ratios from repeated nvbench JSON runs.

Each run is one process. Per run and state, the statistic is the median cold
GPU time. Across runs, the table reports the median of those medians, their
range, and the mean within-run relative standard deviation ("noise").

Pairings (Python benchmark ``base`` vs C++):

    sum, full         vs  warp_reduce_sum_method (Sum), Unroll=full
    sum, loop         vs  warp_reduce_sum_method (Sum), Unroll=loop
    sum, loop         vs  warp_reduce_sum (Reduce(plus), upstream, loop)
    reduce_add, loop  vs  warp_reduce_sum (Reduce(plus), upstream, loop)
"""

import argparse
import collections
import json
import statistics
import sys

STATISTICS = {
    "median": "nv/cold/time/gpu/median",
    "mean": "nv/cold/time/gpu/mean",
    "noise": "nv/cold/time/gpu/stdev/relative",
    "cpu_mean": "nv/cold/time/cpu/mean",
    "sm_clock": "nv/cold/sm_clock_rate/mean",
}


def load_states(paths):
    """Return {axis-tuple: [per-run statistics]} for the given run files."""
    runs = collections.defaultdict(list)
    for path in paths:
        with open(path) as f:
            document = json.load(f)
        for benchmark in document["benchmarks"]:
            for state in benchmark["states"]:
                if state.get("is_skipped"):
                    continue
                axes = tuple(
                    sorted(
                        (a["name"], str(a["value"]))
                        for a in state["axis_values"]
                    )
                )
                summaries = {
                    s["tag"]: float(
                        next(
                            d["value"]
                            for d in s["data"]
                            if d["name"] == "value"
                        )
                    )
                    for s in state["summaries"]
                    if s["tag"] in STATISTICS.values()
                }
                runs[axes].append(
                    {k: summaries.get(tag) for k, tag in STATISTICS.items()}
                )
    return runs


def lookup(runs, **axes):
    for key, values in runs.items():
        named = dict(key)
        if all(named.get(name) == value for name, value in axes.items()):
            return values
    return None


def aggregate(values):
    medians = [v["median"] for v in values]
    return {
        "runs": len(values),
        "median": statistics.median(medians),
        "min": min(medians),
        "max": max(medians),
        "noise": statistics.mean(v["noise"] for v in values),
        "cpu_mean": statistics.median(v["cpu_mean"] for v in values),
        "sm_clock": statistics.median(v["sm_clock"] for v in values),
    }


def rows(python, cpp_sum, cpp_reduce, types, unroll_factor):
    pairings = [
        ("sum", "full", cpp_sum, "Sum", "full"),
        ("sum", "loop", cpp_sum, "Sum", "loop"),
        ("sum", "loop", cpp_reduce, "Reduce(plus)", "loop"),
        ("reduce_add", "loop", cpp_reduce, "Reduce(plus)", "loop"),
    ]
    for algorithm, py_unroll, cpp_runs, cpp_name, cpp_unroll in pairings:
        for type_name in types:
            py = lookup(
                python,
                **{"T{ct}": type_name, "Algorithm{ct}": algorithm},
                UnrollFactor=str(unroll_factor),
                Unroll=py_unroll,
            )
            if cpp_name == "Sum":
                cpp = lookup(
                    cpp_runs,
                    **{"T{ct}": type_name},
                    UnrollFactor=str(unroll_factor),
                    Unroll=cpp_unroll,
                )
            else:
                cpp = lookup(cpp_runs, **{"T{ct}": type_name})
            yield {
                "python": f"{algorithm}/{py_unroll}",
                "cpp": f"{cpp_name}/{cpp_unroll}",
                "type": type_name,
                "py": aggregate(py) if py else None,
                "cxx": aggregate(cpp) if cpp else None,
            }


def format_us(seconds):
    return f"{seconds * 1e6:.3f}"


def format_range(side):
    median, low, high = (format_us(side[k]) for k in ("median", "min", "max"))
    return f"{median} [{low}, {high}]"


def markdown(table):
    header = [
        "Python",
        "C++",
        "T",
        "Python GPU median us [min, max]",
        "C++ GPU median us [min, max]",
        "Python/C++",
        "ratio range",
        "noise py / C++",
        "runs",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for row in table:
        py, cxx = row["py"], row["cxx"]
        if py is None or cxx is None:
            missing = "Python" if py is None else "C++"
            lines.append(
                f"| {row['python']} | {row['cpp']} | {row['type']} | "
                f"{'n/a' if py is None else format_us(py['median'])} | "
                f"{'n/a' if cxx is None else format_us(cxx['median'])} | "
                f"n/a | no {missing} data | | |"
            )
            continue
        cells = [
            row["python"],
            row["cpp"],
            row["type"],
            format_range(py),
            format_range(cxx),
            f"{py['median'] / cxx['median']:.3f}",
            f"{py['min'] / cxx['max']:.3f} - {py['max'] / cxx['min']:.3f}",
            f"{py['noise'] * 100:.1f}% / {cxx['noise'] * 100:.1f}%",
            f"{py['runs']} / {cxx['runs']}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--python", nargs="+", required=True)
    parser.add_argument("--cpp-sum", nargs="+", required=True)
    parser.add_argument("--cpp-reduce", nargs="+", required=True)
    parser.add_argument("--types", default="I8,I16,I32,I64,F16,F32,F64")
    parser.add_argument("--unroll-factor", type=int, default=128)
    parser.add_argument("--json", help="also write the aggregated rows here")
    args = parser.parse_args(argv)

    table = list(
        rows(
            load_states(args.python),
            load_states(args.cpp_sum),
            load_states(args.cpp_reduce),
            args.types.split(","),
            args.unroll_factor,
        )
    )
    print(markdown(table))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(table, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
