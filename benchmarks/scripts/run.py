#!/usr/bin/env python3

import json
import math
import os
import sys
import time

import cccl.bench

TIMING_REPORT_PATH = "cccl_meta_bench_timing.json"


def elapsed_time_looks_good(x):
    if isinstance(x, float):
        if math.isfinite(x):
            return True
    return False


def get_largest_problem_size(rt_values):
    # Small problem sizes do not utilize entire GPU.
    # Benchmarking small problem sizes in environments where we do not control
    # distributions comparison, e.g. CI, is not useful because of stability issues.
    elements = []
    for element in rt_values:
        if element.isdigit():
            elements.append(int(element))
    return [str(max(elements))]


def filter_runtime_workloads_for_ci(rt_values):
    for subbench in rt_values:
        for axis in rt_values[subbench]:
            if axis.startswith("Elements") and axis.endswith("[pow2]"):
                rt_values[subbench][axis] = get_largest_problem_size(
                    rt_values[subbench][axis]
                )

    return rt_values


class BaseRunner:
    def __init__(self):
        self.estimator = cccl.bench.MedianCenterEstimator()
        self.timings = {}

    def __call__(self, algname, ct_workload_space, rt_values):
        rt_values = filter_runtime_workloads_for_ci(rt_values)
        self.timings[algname] = []

        bench = cccl.bench.BaseBench(algname)
        for ct_workload in ct_workload_space:
            begin = time.perf_counter()
            try:
                results = bench.run(ct_workload, rt_values, self.estimator, False)
                for subbench in results:
                    for point in results[subbench]:
                        bench_name = "{}.{}-{}".format(
                            bench.algorithm_name(), subbench, point
                        )
                        bench_name = bench_name.replace(" ", "___")
                        bench_name = "".join(
                            c if c.isalnum() else "_" for c in bench_name
                        )
                        elapsed_time = results[subbench][point]
                        if elapsed_time_looks_good(elapsed_time):
                            print(
                                "&&&& PERF {} {} -sec".format(bench_name, elapsed_time)
                            )
            finally:
                self.timings[algname].append(
                    {
                        "compile_time_axes": list(ct_workload),
                        "elapsed_seconds": time.perf_counter() - begin,
                    }
                )


def create_timing_report(algorithm_results, runner_timings, total_elapsed):
    return {
        "schema_version": 1,
        "total_elapsed_seconds": total_elapsed,
        "algorithms": [
            {
                **timing,
                "workloads": runner_timings.get(timing["algorithm"], []),
            }
            for timing in algorithm_results
        ],
    }


def write_timing_report(report, path=TIMING_REPORT_PATH):
    with open(path, "w", encoding="utf-8") as timing_file:
        json.dump(report, timing_file, indent=2)
        timing_file.write("\n")


def print_timing_summary(report):
    print("\n### Benchmark timing summary")
    for algorithm in report["algorithms"]:
        benchmark_seconds = sum(
            workload["elapsed_seconds"] for workload in algorithm["workloads"]
        )
        print(
            "  * {}: {:.1f}s benchmark, {} workload(s)".format(
                algorithm["algorithm"],
                benchmark_seconds,
                len(algorithm["workloads"]),
            )
        )
        for workload in algorithm["workloads"]:
            axes = " ".join(workload["compile_time_axes"])
            print("    * {}: {:.1f}s".format(axes, workload["elapsed_seconds"]))
    print("  * total: {:.1f}s".format(report["total_elapsed_seconds"]))
    print("  * timing report: {}".format(TIMING_REPORT_PATH))


def main():
    print("&&&& RUNNING bench")
    os.environ["CUDA_MODULE_LOADING"] = "EAGER"
    runner = BaseRunner()
    begin = time.perf_counter()
    algorithm_results = cccl.bench.search(runner)
    total_elapsed = time.perf_counter() - begin

    if algorithm_results is not None:
        report = create_timing_report(algorithm_results, runner.timings, total_elapsed)
        write_timing_report(report)
        print_timing_summary(report)

        failed = [
            timing["algorithm"]
            for timing in algorithm_results
            if not timing["succeeded"]
        ]
        if failed:
            print("&&&& FAILED {}".format(", ".join(failed)))
            sys.exit(1)

    print("&&&& PASSED bench")


if __name__ == "__main__":
    main()
