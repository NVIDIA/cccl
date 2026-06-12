# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import argparse
import time

import cuda.compute as cc

from common import (
    TimingResult,
    add_case_filter,
    add_json_output,
    print_results,
    select_cases,
    synchronize,
    write_results_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure cold cuda.compute make_* build time."
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=10,
        help="Number of cold build samples. Defaults to 10.",
    )
    add_case_filter(parser)
    add_json_output(parser)
    args = parser.parse_args()

    results = []
    for case in select_cases(args.case):
        state = case.setup()
        synchronize()

        samples_ns = []
        for _ in range(args.repeat):
            cc.clear_all_caches()
            start = time.perf_counter_ns()
            case.make_wrapper(state)
            end = time.perf_counter_ns()
            samples_ns.append(end - start)

        results.append(TimingResult(case.name, samples_ns=samples_ns, number=1))

    print_results(results)
    if args.json is not None:
        write_results_json(
            args.json,
            benchmark="build_time",
            results=results,
            config={"repeat": args.repeat, "case": args.case},
        )


if __name__ == "__main__":
    main()
