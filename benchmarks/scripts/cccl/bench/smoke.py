# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Configuration and validation for the CUB performance smoke profile."""

from collections.abc import Collection

# { subbench_name: { "FeatureNameA": ["ValueA1", "ValueA2", ...]},
#                    "FeatureNameB": ["ValueB1", "ValueB2", ...]}, ... }
# Example: {"small": {"Elements{io}[pow2]":   ["26"],
#                     "MaxSegmentSize[pow2]": ["8"]}}
RuntimeBenchmarkInputs = dict[str, dict[str, list[str]]]

SMOKE_WORKLOADS: dict[str, dict[str, list[str]]] = {
    "cub.bench.radix_sort.keys": {
        "T{ct}": ["I8", "I16", "I32", "I64"],
        "OffsetT{ct}": ["I32"],
        "Elements{io}[pow2]": ["28"],
        "Entropy": ["0.544"],
    },
    "cub.bench.radix_sort.pairs": {
        "KeyT{ct}": ["I32"],
        "ValueT{ct}": ["I8", "I16", "I32", "I64"],
        "OffsetT{ct}": ["I32"],
        "Elements{io}[pow2]": ["28"],
        "Entropy": ["1.000"],
    },
    "cub.bench.segmented_radix_sort.keys": {
        "T{ct}": ["I8", "I16", "I32", "I64"],
        "OffsetT{ct}": ["I32"],
        "Elements{io}[pow2]": ["26"],
        "Segments{io}[pow2]": ["16"],
        "Entropy": ["1.000"],
    },
    "cub.bench.segmented_sort.keys": {
        "T{ct}": ["I32"],
        "OffsetT{ct}": ["I32"],
        "Elements{io}[pow2]": ["26"],
    },
    "cub.bench.reduce.sum": {
        "T{ct}": ["I8", "I16", "F32", "F64", "C32"],
        "OffsetT{ct}": ["I32"],
        "Elements{io}[pow2]": ["28"],
    },
    "cub.bench.reduce.by_key": {
        "KeyT{ct}": ["I32"],
        "ValueT{ct}": ["I8", "I16", "I32", "I64", "C32"],
        "OffsetT{ct}": ["I32"],
        "Elements{io}[pow2]": ["28"],
        "MaxSegSize[pow2]": ["4"],
    },
    "cub.bench.segmented_reduce.sum": {
        "T{ct}": ["I32", "F32"],
        "Elements{io}[pow2]": ["28"],
    },
    "cub.bench.scan.exclusive.sum": {
        "T{ct}": ["I8", "I16", "I32", "I64"],
        "OffsetT{ct}": ["I32"],
        "Elements{io}[pow2]": ["28"],
    },
    "cub.bench.scan.exclusive.by_key": {
        "KeyT{ct}": ["I32"],
        "ValueT{ct}": ["I8", "I16", "I32", "I64"],
        "OffsetT{ct}": ["I32"],
        "Elements{io}[pow2]": ["28"],
    },
    "cub.bench.scan.exclusive.deterministic": {
        "T{ct}": ["F32", "F64"],
        "OffsetT{ct}": ["I64"],
        "Elements{io}[pow2]": ["28"],
    },
    "cub.bench.segmented_scan.sum": {
        "T{ct}": ["I32", "F32"],
        "OffsetT{ct}": ["I32"],
        "Elements{io}[pow2]": ["26"],
        "SegmentSize{io}": ["233"],
    },
    "cub.bench.topk.keys": {
        "KeyT{ct}": ["F32"],
        "OffsetT{ct}": ["I32"],
        "OutOffsetT{ct}": ["I32"],
        "Elements{io}[pow2]": ["28"],
        "SelectedElements[pow2]": ["3", "7", "15", "23"],
        "Entropy": ["0.544"],
    },
    "cub.bench.topk.pairs": {
        "KeyT{ct}": ["I32"],
        "ValueT{ct}": ["I32"],
        "OffsetT{ct}": ["I32"],
        "OutOffsetT{ct}": ["I32"],
        "Elements{io}[pow2]": ["28"],
        "SelectedElements[pow2]": ["3", "7", "15", "23"],
        "Entropy": ["0.544"],
    },
    "cub.bench.segmented_topk.fixed.keys": {
        "KeyT{ct}": ["F32"],
        "MaxSegmentSize{ct}": ["1024"],
        "MaxNumSelected{ct}": ["32"],
        "Elements{io}[pow2]": ["28"],
        "Entropy": ["0.544"],
    },
    "cub.bench.segmented_topk.variable.keys": {
        "KeyT{ct}": ["F32"],
        "MaxSegmentSize{ct}": ["8192"],
        "K{ct}": ["512"],
        "NumSegments": ["32"],
        "Pattern": ["relu_quantized"],
    },
    "cub.bench.transform.babelstream": {
        "T{ct}": ["I8", "I16", "F32", "F64"],
        "Aligned": ["yes"],
        "Elements{io}[pow2]": ["28"],
    },
    "cub.bench.transform.fill": {
        "T{ct}": ["I8", "I16", "I32", "I64"],
        "Aligned": ["yes"],
        "Elements{io}[pow2]": ["28"],
    },
    "cub.bench.merge_sort.keys": {
        "T{ct}": ["I32", "F64"],
        "OffsetT{ct}": ["I32"],
        "Elements{io}[pow2]": ["28"],
        "Entropy": ["1.000"],
    },
    "cub.bench.merge_sort.pairs": {
        "KeyT{ct}": ["I32"],
        "ValueT{ct}": ["I32"],
        "OffsetT{ct}": ["I32"],
        "Elements{io}[pow2]": ["28"],
        "Entropy": ["1.000"],
    },
    "cub.bench.adjacent_difference.subtract_left": {
        "T{ct}": ["I32"],
        "OffsetT{ct}": ["I32"],
        "Elements{io}[pow2]": ["28"],
    },
    "cub.bench.find_if.base": {
        "T{ct}": ["I32"],
        "OffsetT{ct}": ["I32"],
        "Elements{io}[pow2]": ["28"],
        "MismatchAt": ["1", "0.5", "0"],
    },
    "cub.bench.run_length_encode.encode": {
        "T{ct}": ["I32"],
        "OffsetT{ct}": ["I32"],
        "RunLengthT{ct}": ["I32"],
        "Elements{io}[pow2]": ["28"],
        "MaxSegSize[pow2]": ["4"],
    },
    "cub.bench.select.flagged": {
        "T{ct}": ["I32", "F64"],
        "InPlace{ct}": ["false"],
        "Elements{io}[pow2]": ["28"],
        "Entropy": ["0.544"],
    },
    "cub.bench.select.unique": {
        "T{ct}": ["F32"],
        "InPlace{ct}": ["false"],
        "Elements{io}[pow2]": ["28"],
        "MaxSegSize[pow2]": ["4"],
    },
    "cub.bench.select.unique_by_key": {
        "KeyT{ct}": ["I32"],
        "ValueT{ct}": ["F32"],
        "OffsetT{ct}": ["I32"],
        "Elements{io}[pow2]": ["28"],
        "MaxSegSize[pow2]": ["4"],
    },
    "cub.bench.partition.three_way": {
        "T{ct}": ["F32"],
        "OffsetT{ct}": ["I32"],
        "Elements{io}[pow2]": ["28"],
        "Entropy": ["0.544"],
    },
    "cub.bench.histogram.even": {
        "SampleT{ct}": ["I32"],
        "CounterT{ct}": ["I32"],
        "OffsetT{ct}": ["I32"],
        "Elements{io}[pow2]": ["28"],
        "Bins": ["128"],
        "Entropy": ["0.201"],
    },
    "cub.bench.histogram.range": {
        "SampleT{ct}": ["F32"],
        "CounterT{ct}": ["I32"],
        "OffsetT{ct}": ["I32"],
        "Elements{io}[pow2]": ["28"],
        "Bins": ["2048"],
        "Entropy": ["1.000"],
    },
}

SMOKE_BENCHMARKS = tuple(SMOKE_WORKLOADS)

SMOKE_SUBBENCHES: dict[str, RuntimeBenchmarkInputs] = {
    "cub.bench.segmented_radix_sort.keys": {
        "power": {},
        "small": {"MaxSegmentSize[pow2]": ["8"]},
    },
    "cub.bench.segmented_sort.keys": {
        "small": {"MaxSegmentSize[pow2]": ["8"]},
    },
    "cub.bench.segmented_reduce.sum": {
        "small": {"SegmentSize[pow2]": ["4"]},
        "medium": {"SegmentSize[pow2]": ["8"]},
        "large": {"SegmentSize[pow2]": ["12"]},
    },
    "cub.bench.segmented_scan.sum": {
        "fixed_size_segments": {},
    },
    "cub.bench.segmented_topk.variable.keys": {
        "decode_style_variable_topk_keys": {},
    },
}


def validate_smoke_benchmarks(available_benchmarks: Collection[str]) -> None:
    """Reject stale smoke entries instead of silently dropping them."""
    missing = [name for name in SMOKE_BENCHMARKS if name not in available_benchmarks]
    if missing:
        raise ValueError(
            "Smoke benchmarks are not registered in this build: {}".format(
                ", ".join(missing)
            )
        )


def smoke_runtime_bench_inputs(
    alg_name: str, runtime_benchmark_inputs: RuntimeBenchmarkInputs
) -> RuntimeBenchmarkInputs:
    """Limit algorithms with multiple subbenchmarks to representative cases."""
    selected_subbenches = SMOKE_SUBBENCHES.get(alg_name)
    if selected_subbenches is None:
        return runtime_benchmark_inputs

    # extract runtime values for selected subbenchmarks
    return {
        subbench_name: {
            **runtime_benchmark_inputs[subbench_name],
            **overrides,
        }
        for subbench_name, overrides in selected_subbenches.items()
    }
