import argparse
import re

import numpy as np

from .bench import BaseBench, Bench
from .cmake import CMake
from .config import Config
from .smoke import (
    SMOKE_BENCHMARKS,
    SMOKE_SUBBENCHES,
    SMOKE_WORKLOADS,
    filter_smoke_subbench_inputs,
    validate_smoke_benchmarks,
)
from .storage import Storage


def benchmark_group(algname):
    """Return the project and algorithm family"""
    parts = algname.split(".")
    if len(parts) >= 3 and parts[1] == "bench":
        return parts[0], parts[2]
    return parts[0], "other"


def list_benches(algnames, config=None, smoke=False):
    """Print the benchmarks grouped by project and algorithm"""
    print("### Benchmarks ({})".format(len(algnames)))

    if config is None:
        config = Config()

    current_group = None
    for algname in algnames:
        group = benchmark_group(algname)
        if group != current_group:
            print("\n#### {} / {}".format(*group))
            current_group = group

        space_size = config.variant_space_size(algname)
        variant_suffix = "" if space_size == 1 else "s"
        print("  * `{}`: {} variant{}".format(algname, space_size, variant_suffix))

        if smoke:
            workload = SMOKE_WORKLOADS[algname]
            values = [
                "`{}={}`".format(axis, ",".join(axis_values))
                for axis, axis_values in workload.items()
            ]
            print("    * workload: {}".format(", ".join(values)))
            if algname in SMOKE_SUBBENCHES:
                print(
                    "    * subbenchmarks: {}".format(
                        ", ".join(SMOKE_SUBBENCHES[algname])
                    )
                )
        else:
            for param_space in config.benchmarks[algname]:
                param_name = param_space.label
                param_rng = (param_space.low, param_space.high, param_space.step)
                print("    * `{}`: {}".format(param_name, param_rng))


def parse_sub_space(args):
    sub_space = {}
    for axis in args:
        name, value = axis.split("=")

        if "[" in value:
            value = value.replace("[", "").replace("]", "")
            values = value.split(",")
        else:
            values = [value]
        sub_space[name] = values

    return sub_space


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Runs benchmarks and stores results in a database."
    )
    parser.add_argument(
        "-R", type=str, default=".*", help="Regex for benchmarks selection."
    )
    parser.add_argument(
        "-a",
        "--args",
        action="append",
        type=str,
        help="Parameter in the format `Param=Value`.",
    )
    parser.add_argument(
        "-l", "--list-benches", action="store_true", help="Show available benchmarks."
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split benchmarks into NUM_SHARDS pieces and only run one",
    )
    parser.add_argument(
        "--run-shard",
        type=int,
        default=0,
        help="Run benchmark shard RUN_SHARD from NUM_SHARDS pieces",
    )
    priority = parser.add_mutually_exclusive_group()
    priority.add_argument("-P0", action="store_true", help="Run P0 benchmarks")
    priority.add_argument(
        "--smoke",
        action="store_true",
        help="Run representative workloads for critical CUB benchmarks.",
    )
    return parser.parse_args()


def filter_benchmark_space_for_p0(algname, ct_space, rt_values):
    if algname in [
        "cub.bench.merge_sort.pairs",
        "cub.bench.radix_sort.pairs",
        "cub.bench.select.unique_by_key",
    ]:
        ct_space = list(
            filter(
                lambda variant: (
                    not (
                        ("OffsetT{ct}=I64" in variant)
                        or ("KeyT{ct}=I16" in variant)
                        or ("ValueT{ct}=I16" in variant)
                        or ("KeyT{ct}=I128" in variant)
                        or ("ValueT{ct}=I128" in variant)
                    )
                ),
                ct_space,
            )
        )

    # Scanning 2^32 elements of I128 allocates too much memory on DGX Spark. See NVBug 6683746
    if algname in [
        "cub.bench.scan.exclusive.sum",
        "cub.bench.scan.exclusive.deterministic",
    ]:
        ct_space = list(filter(lambda variant: "T{ct}=I128" not in variant, ct_space))

    if algname == "cub.bench.merge_sort.pairs":
        for subbench in rt_values:
            for axis in rt_values[subbench]:
                if axis == "Entropy":
                    rt_values[subbench][axis] = ["1.000"]

    return ct_space, rt_values


def run_benches(algnames, sub_space, seeker, args):
    results = []
    for algname in algnames:
        succeeded = True
        try:
            bench = BaseBench(algname)
            algorithm_sub_space = SMOKE_WORKLOADS[algname] if args.smoke else sub_space

            ct_space = bench.ct_workload_space(algorithm_sub_space)
            rt_values = bench.rt_axes_values(algorithm_sub_space)
            if args.smoke:
                rt_values = filter_smoke_subbench_inputs(algname, rt_values)
            if args.P0:
                ct_space, rt_values = filter_benchmark_space_for_p0(
                    algname, ct_space, rt_values
                )
            seeker(algname, ct_space, rt_values)
        except Exception as e:
            succeeded = False
            print(
                "#### ERROR exception occurred while running {}: '{}'".format(
                    algname, e
                )
            )
        results.append({"algorithm": algname, "succeeded": succeeded})

    return results


def filter_benchmarks_by_regex(benchmarks, R):
    pattern = re.compile(R)
    return list(filter(lambda x: pattern.match(x), benchmarks))


def filter_benchmarks(benchmarks, args):
    if args.run_shard >= args.num_shards:
        raise ValueError("run-shard must be less than num-shards")

    cub_p0_benchmarks = [
        "cub.bench.merge_sort.keys",
        "cub.bench.merge_sort.pairs",
        "cub.bench.radix_sort.keys",
        "cub.bench.radix_sort.pairs",
        "cub.bench.reduce.by_key",
        "cub.bench.reduce.custom",
        "cub.bench.reduce.sum",
        "cub.bench.scan.exclusive.deterministic",
        "cub.bench.scan.exclusive.sum",
        "cub.bench.select.flagged",
        "cub.bench.select.if",
        "cub.bench.select.unique",
        "cub.bench.select.unique_by_key",
        "cub.bench.transform.babelstream",
        "cub.bench.transform.fill",
    ]

    algnames = filter_benchmarks_by_regex(benchmarks.keys(), args.R)
    if args.smoke:
        validate_smoke_benchmarks(benchmarks)
        algnames = [name for name in SMOKE_BENCHMARKS if name in algnames]
    elif args.P0:
        non_cub_algnames = [name for name in algnames if not name.startswith("cub.")]
        non_cub_algnames = filter_benchmarks_by_regex(
            non_cub_algnames,
            r"^(?!.*segmented).*(scan|reduce|select|sort|transform\.(babelstream|fill)).*",
        )
        non_cub_algnames = filter_benchmarks_by_regex(
            non_cub_algnames, r"^(?!.*P[123456789]\d*).*"
        )

        algnames = [name for name in cub_p0_benchmarks if name in algnames]
        algnames.extend(non_cub_algnames)
    if not args.smoke:
        algnames.sort()

    if args.num_shards > 1:
        algnames = np.array_split(algnames, args.num_shards)[args.run_shard].tolist()
        return algnames

    return algnames


def search(seeker):
    args = parse_arguments()

    # This guard has never fired: Storage() calls sqlite3.connect(), which
    # creates the DB file before exists() is checked. Making it fire would
    # wipe all built benchmark binaries on every fresh campaign; current
    # workflows rely on fresh runs reusing existing binaries (see #11096).
    if not Storage().exists():
        CMake().clean()

    config = Config()
    print(" ctk: ", config.ctk)
    print("cccl: ", config.cccl)

    workload_sub_space = {}

    if args.args:
        workload_sub_space = parse_sub_space(args.args)

    algnames = filter_benchmarks(config.benchmarks, args)
    if args.list_benches:
        list_benches(algnames, config, args.smoke)
        return

    return run_benches(algnames, workload_sub_space, seeker, args)


class MedianCenterEstimator:
    def __init__(self):
        pass

    def __call__(self, samples):
        if len(samples) == 0:
            return float("inf")

        return float(np.median(samples))


class BruteForceSeeker:
    def __init__(self, base_center_estimator, variant_center_estimator):
        self.base_center_estimator = base_center_estimator
        self.variant_center_estimator = variant_center_estimator

    def __call__(self, algname, ct_workload_space, rt_values):
        variants = Config().variant_space(algname)

        for ct_workload in ct_workload_space:
            for variant in variants:
                bench = Bench(algname, variant, list(ct_workload))
                if bench.build():
                    score = bench.score(
                        ct_workload,
                        rt_values,
                        self.base_center_estimator,
                        self.variant_center_estimator,
                    )

                    print(bench.label(), score)
