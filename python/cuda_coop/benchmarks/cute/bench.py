# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Small cuda.coop.cutlass benchmark harness.

Run from the source tree with:

    python -m benchmarks.cute.bench --measure-iters 16
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from examples.cutlass import (
    cute_kmeans_assign_gemm_argmin,
    cute_kmeans_assign_topk,
    cute_legacy_reduce_compare,
    cute_mma_amax_sm100,
    cute_mma_topk_sm100,
    cute_run_length_decode_window,
    cute_scheduler_prefix,
    cute_sort_and_segment,
    cute_sort_and_segment_thread_data,
    cute_sort_register_fragment,
    cute_thread_group_descriptor_reduce,
    cute_thread_group_query,
    cute_thread_group_reduce,
    cute_thread_hierarchy_reduce,
    cute_topk_score_window,
    cute_warp_merge_sort,
    cute_warp_prefix_reduce,
    mixed_payload_factory_sort_topk,
    mixed_payload_sort_topk,
    mixed_tensor_vector_scan,
    prims_vector_block_exchange,
    prims_vector_block_prefix_segment,
    prims_vector_histogram_run_length,
    prims_vector_pair_sort_topk,
    prims_vector_rank_merge,
    prims_vector_sort_topk,
    prims_vector_warp_merge_sort,
    prims_vector_warp_prefix,
)

from . import kmeans_assign_reference

CUPTI_TIMER_UNSUPPORTED_MESSAGE = (
    "CUPTI timing requires a nvidia-cutlass-dsl build that provides "
    "cutlass.cute.testing.JitArguments and benchmark(..., use_cupti=True)"
)

DEFAULT_SCENARIOS = (
    "cute_kmeans_assign_topk",
    "cute_kmeans_assign_topk_batched",
    "cute_kmeans_assign_topk_feature_split_batched",
    "cute_kmeans_assign_topk_feature_split_score_batched",
    "cute_kmeans_assign_topk_feature_split_top1_score_batched",
    "cute_kmeans_assign_topk_feature_split_top1_score_warp_batched",
    "cute_kmeans_assign_topk_wide_batched",
    "cute_kmeans_assign_gemm_argmin",
    "kmeans_assign_torch_gemm_argmin_reference",
    "kmeans_assign_cute_gemm_coop_argmin_reference",
    "cute_legacy_reduce_compare",
    "cute_run_length_decode_window",
    "cute_scheduler_prefix",
    "cute_sort_register_fragment",
    "cute_sort_and_segment",
    "cute_sort_and_segment_thread_data",
    "cute_thread_group_descriptor_reduce",
    "cute_thread_group_query",
    "cute_thread_group_reduce",
    "cute_thread_hierarchy_reduce",
    "cute_topk_score_window",
    "cute_warp_merge_sort",
    "cute_warp_prefix_reduce",
)

EXPLICIT_SCENARIOS = (
    "mixed_payload_factory_sort_topk",
    "mixed_payload_sort_topk",
    "mixed_tensor_vector_scan",
    "prims_vector_block_exchange",
    "prims_vector_block_prefix_segment",
    "prims_vector_histogram_run_length",
    "prims_vector_pair_sort_topk",
    "prims_vector_rank_merge",
    "prims_vector_sort_topk",
    "prims_vector_warp_merge_sort",
    "prims_vector_warp_prefix",
)

OPTIONAL_SM100_SCENARIOS = (
    "cute_mma_amax_sm100_batched",
    "cute_mma_topk_sm100_block_merge_batched",
    "cute_mma_topk_sm100_warp_merge_batched",
)


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    first_launch_us: float
    steady_launch_us: float
    steady_kernel_us: float | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "name": self.name,
            "first_launch_us": self.first_launch_us,
            "steady_launch_us": self.steady_launch_us,
        }
        if self.steady_kernel_us is not None:
            result["steady_kernel_us"] = self.steady_kernel_us
        return result


def available_scenarios() -> tuple[str, ...]:
    return (
        *DEFAULT_SCENARIOS,
        *OPTIONAL_SM100_SCENARIOS,
        *EXPLICIT_SCENARIOS,
    )


def default_scenarios() -> tuple[str, ...]:
    return DEFAULT_SCENARIOS


def _scenario_callable(name: str) -> Callable[[], Any]:
    if name == "cute_kmeans_assign_topk":
        return cute_kmeans_assign_topk.prepare_example
    if name == "cute_kmeans_assign_topk_batched":
        return cute_kmeans_assign_topk.prepare_batched_example
    if name == "cute_kmeans_assign_topk_feature_split_batched":
        return cute_kmeans_assign_topk.prepare_feature_split_batched_example
    if name == "cute_kmeans_assign_topk_feature_split_score_batched":
        return cute_kmeans_assign_topk.prepare_feature_split_score_batched_example
    if name == "cute_kmeans_assign_topk_feature_split_top1_score_batched":
        return cute_kmeans_assign_topk.prepare_feature_split_top1_score_batched_example
    if name == "cute_kmeans_assign_topk_feature_split_top1_score_warp_batched":
        return cute_kmeans_assign_topk.prepare_feature_split_top1_score_warp_batched_example
    if name == "cute_kmeans_assign_topk_wide_batched":
        return cute_kmeans_assign_topk.prepare_wide_batched_example
    if name == "cute_kmeans_assign_gemm_argmin":
        return cute_kmeans_assign_gemm_argmin.prepare_example
    if name == "kmeans_assign_torch_gemm_argmin_reference":
        return kmeans_assign_reference.prepare_torch_gemm_argmin_reference
    if name == "kmeans_assign_cute_gemm_coop_argmin_reference":
        return kmeans_assign_reference.prepare_cute_gemm_coop_argmin_reference
    if name == "cute_legacy_reduce_compare":
        return cute_legacy_reduce_compare.prepare_example
    if name == "cute_mma_amax_sm100_batched":
        return cute_mma_amax_sm100.prepare_benchmark
    if name == "cute_mma_topk_sm100_block_merge_batched":
        return cute_mma_topk_sm100.prepare_block_merge_benchmark
    if name == "cute_mma_topk_sm100_warp_merge_batched":
        return cute_mma_topk_sm100.prepare_warp_merge_benchmark
    if name == "cute_run_length_decode_window":
        return cute_run_length_decode_window.prepare_example
    if name == "cute_scheduler_prefix":
        return cute_scheduler_prefix.prepare_example
    if name == "cute_sort_register_fragment":
        return cute_sort_register_fragment.prepare_example
    if name == "cute_sort_and_segment":
        return cute_sort_and_segment.prepare_example
    if name == "cute_sort_and_segment_thread_data":
        return cute_sort_and_segment_thread_data.prepare_example
    if name == "cute_thread_group_descriptor_reduce":
        return cute_thread_group_descriptor_reduce.prepare_example
    if name == "cute_thread_group_query":
        return cute_thread_group_query.prepare_example
    if name == "cute_thread_group_reduce":
        return cute_thread_group_reduce.prepare_example
    if name == "cute_thread_hierarchy_reduce":
        return cute_thread_hierarchy_reduce.prepare_example
    if name == "cute_topk_score_window":
        return cute_topk_score_window.prepare_example
    if name == "cute_warp_merge_sort":
        return cute_warp_merge_sort.prepare_example
    if name == "cute_warp_prefix_reduce":
        return cute_warp_prefix_reduce.prepare_example
    if name == "mixed_payload_sort_topk":
        return mixed_payload_sort_topk.prepare_example
    if name == "mixed_payload_factory_sort_topk":
        return mixed_payload_factory_sort_topk.prepare_example
    if name == "mixed_tensor_vector_scan":
        return mixed_tensor_vector_scan.prepare_example
    if name == "prims_vector_block_exchange":
        return prims_vector_block_exchange.prepare_example
    if name == "prims_vector_block_prefix_segment":
        return prims_vector_block_prefix_segment.prepare_example
    if name == "prims_vector_histogram_run_length":
        return prims_vector_histogram_run_length.prepare_example
    if name == "prims_vector_pair_sort_topk":
        return prims_vector_pair_sort_topk.prepare_example
    if name == "prims_vector_rank_merge":
        return prims_vector_rank_merge.prepare_example
    if name == "prims_vector_sort_topk":
        return prims_vector_sort_topk.prepare_example
    if name == "prims_vector_warp_merge_sort":
        return prims_vector_warp_merge_sort.prepare_example
    if name == "prims_vector_warp_prefix":
        return prims_vector_warp_prefix.prepare_example
    raise ValueError(f"Unknown cuda.coop.cutlass benchmark scenario: {name}")


def _measure_kernel_activity_us(
    step: Callable[[], Any],
    *,
    warmup_iters: int,
    measure_iters: int,
) -> float:
    last_exc: Exception | None = None
    for module_name in ("cutlass.cute.testing", "cutlass.testing"):
        try:
            testing = importlib.import_module(module_name)
            JitArguments = testing.JitArguments
            benchmark = testing.benchmark
            break
        except (AttributeError, ImportError, ModuleNotFoundError) as exc:
            last_exc = exc
    else:
        if last_exc is not None:
            raise RuntimeError(CUPTI_TIMER_UNSUPPORTED_MESSAGE) from last_exc
        raise RuntimeError(CUPTI_TIMER_UNSUPPORTED_MESSAGE)

    try:
        benchmark_parameters = inspect.signature(benchmark).parameters
    except (TypeError, ValueError) as exc:
        raise RuntimeError(CUPTI_TIMER_UNSUPPORTED_MESSAGE) from exc

    if (
        "kernel_arguments" not in benchmark_parameters
        or "use_cupti" not in benchmark_parameters
    ):
        raise RuntimeError(CUPTI_TIMER_UNSUPPORTED_MESSAGE)

    # cutlass.testing.benchmark reports per-iteration CUPTI kernel activity
    # in microseconds.
    return float(
        benchmark(
            step,
            warmup_iterations=max(1, int(warmup_iters)),
            iterations=max(1, int(measure_iters)),
            kernel_arguments=JitArguments(),
            use_cupti=True,
        )
    )


def _measure_scenario(
    name: str,
    *,
    warmup_iters: int,
    measure_iters: int,
    timer: str,
) -> ScenarioResult:
    prepared = _scenario_callable(name)()

    start = time.perf_counter()
    prepared.step()
    prepared.synchronize()
    first_launch_us = (time.perf_counter() - start) * 1.0e6
    prepared.validate()

    for _ in range(warmup_iters):
        prepared.step()
    prepared.synchronize()

    start = time.perf_counter()
    for _ in range(measure_iters):
        prepared.step()
    prepared.synchronize()
    steady_launch_us = (time.perf_counter() - start) * 1.0e6 / measure_iters
    prepared.validate()

    steady_kernel_us = None
    if timer == "cupti":
        custom_measure = getattr(prepared, "measure_cuda_event_us", None)
        if callable(custom_measure):
            steady_kernel_us = float(
                custom_measure(
                    warmup_iters=warmup_iters,
                    measure_iters=measure_iters,
                )
            )
        else:
            steady_kernel_us = _measure_kernel_activity_us(
                prepared.step,
                warmup_iters=warmup_iters,
                measure_iters=measure_iters,
            )
        prepared.validate()

    return ScenarioResult(
        name=name,
        first_launch_us=float(first_launch_us),
        steady_launch_us=float(steady_launch_us),
        steady_kernel_us=steady_kernel_us,
    )


def run_sanity_suite(
    *,
    scenarios: list[str] | None = None,
    warmup_iters: int = 2,
    measure_iters: int = 8,
    timer: str = "wall",
) -> list[dict[str, Any]]:
    if timer not in {"wall", "cupti"}:
        raise ValueError("timer must be 'wall' or 'cupti'")

    requested = scenarios or list(default_scenarios())
    results = [
        _measure_scenario(
            name,
            warmup_iters=max(0, int(warmup_iters)),
            measure_iters=max(1, int(measure_iters)),
            timer=timer,
        )
        for name in requested
    ]
    return [result.to_dict() for result in results]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        action="append",
        choices=available_scenarios(),
        help="Scenario to run. May be specified more than once.",
    )
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--measure-iters", type=int, default=8)
    parser.add_argument(
        "--timer",
        choices=("wall", "cupti"),
        default="wall",
        help=(
            "Include CUPTI kernel activity timing through cutlass.cute.testing. "
            "Wall timing remains useful for DSL runner overhead; CUPTI timing "
            "is the steady GPU-kernel metric to compare with upstream kernel "
            "claims and is measured in a separate benchmark loop."
        ),
    )
    args = parser.parse_args(argv)

    print(
        json.dumps(
            run_sanity_suite(
                scenarios=args.scenario,
                warmup_iters=args.warmup_iters,
                measure_iters=args.measure_iters,
                timer=args.timer,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
