# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import argparse

import cuda.compute as cc

from common import (
    add_case_filter,
    add_json_output,
    measure_call,
    patch_wrapper_to_skip_native_compute,
    print_results,
    select_cases,
    synchronize,
    write_results_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure cached cuda.compute public one-shot host overhead."
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=20,
        help="Number of timing samples.",
    )
    parser.add_argument(
        "--number",
        type=int,
        default=100,
        help="Number of calls per timing sample.",
    )
    add_case_filter(parser)
    add_json_output(parser)
    args = parser.parse_args()

    results = []
    for case in select_cases(args.case):
        cc.clear_all_caches()
        state = case.setup()
        wrapper = case.make_wrapper(state)
        patch_wrapper_to_skip_native_compute(wrapper, case.noop_return_kind)
        synchronize()

        results.append(
            measure_call(
                case.name,
                lambda case=case, state=state: case.oneshot(state),
                repeat=args.repeat,
                number=args.number,
            )
        )

    print_results(results)
    if args.json is not None:
        write_results_json(
            args.json,
            benchmark="oneshot_cached",
            results=results,
            config={
                "repeat": args.repeat,
                "number": args.number,
                "case": args.case,
            },
        )


if __name__ == "__main__":
    main()
