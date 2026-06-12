# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Literal

import cupy as cp
import numpy as np

import cuda.compute as cc
from cuda.compute._cpp_compile import compile_cpp_op_code
from cuda.compute.op import RawOp

NOOP_TEMP_STORAGE_BYTES = 1
NUM_ITEMS = 128
NUM_SEGMENTS = 4
MIN_SAMPLES_FOR_NOISE_ESTIMATE = 5

NoopReturnKind = Literal["none", "temp_storage_bytes", "temp_storage_and_selector"]


@dataclass(frozen=True)
class TimingResult:
    name: str
    samples_ns: list[float]
    number: int

    @property
    def min_ns(self) -> float:
        return min(self.samples_ns)

    @property
    def median_ns(self) -> float:
        return statistics.median(self.samples_ns)

    @property
    def mean_ns(self) -> float:
        return statistics.mean(self.samples_ns)

    @property
    def stdev_ns(self) -> float | None:
        if len(self.samples_ns) < MIN_SAMPLES_FOR_NOISE_ESTIMATE:
            return None
        return statistics.stdev(self.samples_ns)

    @property
    def relative_noise(self) -> float | None:
        stdev_ns = self.stdev_ns
        mean_ns = self.mean_ns
        if stdev_ns is None or mean_ns <= 0:
            return None
        return stdev_ns / mean_ns

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "unit": "ns",
            "number": self.number,
            "samples": self.samples_ns,
            "min": self.min_ns,
            "median": self.median_ns,
            "mean": self.mean_ns,
            "stdev": self.stdev_ns,
            "relative_noise": self.relative_noise,
        }


@dataclass(frozen=True)
class HostBenchmarkCase:
    name: str
    setup: Callable[[], SimpleNamespace]
    make_wrapper: Callable[[SimpleNamespace], Any]
    oneshot: Callable[[SimpleNamespace], None]
    twoshot: Callable[[SimpleNamespace, Any], None]
    noop_return_kind: NoopReturnKind
    skip_reason: str | None = None


class NoopBuildResult:
    """Proxy that skips native compute while preserving wrapper host work."""

    def __init__(self, real_build_result: Any, return_kind: NoopReturnKind):
        self._real_build_result = real_build_result
        self._return_kind = return_kind

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real_build_result, name)

    def compute(self, *args, **kwargs):
        return _noop_return(self._return_kind)

    def compute_even(self, *args, **kwargs):
        return _noop_return(self._return_kind)


def _noop_return(return_kind: NoopReturnKind):
    if return_kind == "none":
        return None
    if return_kind == "temp_storage_bytes":
        return NOOP_TEMP_STORAGE_BYTES
    if return_kind == "temp_storage_and_selector":
        return NOOP_TEMP_STORAGE_BYTES, -1
    raise ValueError(f"Unsupported no-op return kind: {return_kind}")


def patch_wrapper_to_skip_native_compute(
    wrapper: Any, return_kind: NoopReturnKind
) -> None:
    """Patch a cached wrapper so measured calls skip native compute."""
    if hasattr(wrapper, "build_result"):
        wrapper.build_result = NoopBuildResult(wrapper.build_result, return_kind)

    if hasattr(wrapper, "device_reduce_fn"):
        wrapper.device_reduce_fn = lambda *args, **kwargs: _noop_return(return_kind)

    if hasattr(wrapper, "device_scan_fn"):
        wrapper.device_scan_fn = lambda *args, **kwargs: _noop_return(return_kind)

    if hasattr(wrapper, "partitioner"):
        patch_wrapper_to_skip_native_compute(wrapper.partitioner, return_kind)


def make_tiny_temp_storage() -> cp.ndarray:
    return cp.empty(NOOP_TEMP_STORAGE_BYTES, dtype=cp.uint8)


def synchronize() -> None:
    cp.cuda.Device().synchronize()


def measure_call(
    name: str,
    fn: Callable[[], None],
    *,
    repeat: int,
    number: int,
) -> TimingResult:
    samples_ns = []
    for _ in range(repeat):
        start = time.perf_counter_ns()
        for _ in range(number):
            fn()
        end = time.perf_counter_ns()
        samples_ns.append((end - start) / number)
    return TimingResult(name=name, samples_ns=samples_ns, number=number)


def print_results(results: Iterable[TimingResult]) -> None:
    rows = list(results)
    name_width = max((len(row.name) for row in rows), default=4)
    print(
        f"{'case':<{name_width}}  {'median':>12}  {'min':>12}  "
        f"{'mean':>12}  {'noise':>8}  {'repeat':>6}  {'number':>6}"
    )
    print("-" * (name_width + 68))
    for result in rows:
        print(
            f"{result.name:<{name_width}}  "
            f"{_format_ns(result.median_ns):>12}  "
            f"{_format_ns(result.min_ns):>12}  "
            f"{_format_ns(result.mean_ns):>12}  "
            f"{_format_percentage(result.relative_noise):>8}  "
            f"{len(result.samples_ns):>6}  "
            f"{result.number:>6}"
        )


def _format_ns(ns: float) -> str:
    if ns < 1_000:
        return f"{ns:.1f} ns"
    if ns < 1_000_000:
        return f"{ns / 1_000:.2f} us"
    return f"{ns / 1_000_000:.2f} ms"


def _format_percentage(value: float | None) -> str:
    if value is None:
        return "inf"
    return f"{value * 100.0:.2f}%"


def add_case_filter(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--case",
        action="append",
        choices=[case.name for case in CASES],
        help="Benchmark case to run. May be passed multiple times.",
    )


def add_json_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json",
        type=Path,
        help="Write structured benchmark results to this JSON file.",
    )


def write_results_json(
    path: Path,
    *,
    benchmark: str,
    results: Iterable[TimingResult],
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "cuda.compute.host_benchmark.v1",
        "benchmark": benchmark,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "environment": _environment_info(),
        "results": [result.as_json() for result in results],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _environment_info() -> dict[str, Any]:
    device_count = cp.cuda.runtime.getDeviceCount()
    devices = []
    for device_id in range(device_count):
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        name = props["name"]
        if isinstance(name, bytes):
            name = name.decode()
        devices.append(
            {
                "id": device_id,
                "name": name,
                "compute_capability": [
                    int(props["major"]),
                    int(props["minor"]),
                ],
            }
        )

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "devices": devices,
    }


def select_cases(case_names: list[str] | None) -> list[HostBenchmarkCase]:
    if not case_names:
        selected_cases = CASES
    else:
        selected = set(case_names)
        selected_cases = [case for case in CASES if case.name in selected]

    runnable = []
    skipped_by_reason: dict[str, list[str]] = {}
    for case in selected_cases:
        if case.skip_reason is None:
            runnable.append(case)
        else:
            skipped_by_reason.setdefault(case.skip_reason, []).append(case.name)

    for reason, names in skipped_by_reason.items():
        print(f"Skipping {len(names)} benchmark case(s): {', '.join(names)}")
        print(f"  Reason: {reason}")

    return runnable


def _numba_cuda_skip_reason() -> str | None:
    try:
        import numba.cuda  # noqa: F401
    except Exception as exc:
        return f"numba.cuda is not available: {exc}"
    return None


_NUMBA_CUDA_SKIP_REASON = _numba_cuda_skip_reason()


def _raw_predicate_i32(name: str) -> RawOp:
    source = f"""
extern "C" __device__ void {name}(void* x, void* result) {{
    int value = *static_cast<int*>(x);
    *static_cast<bool*>(result) = value < {NUM_ITEMS // 2};
}}
"""
    return RawOp(ltoir=compile_cpp_op_code(source), name=name)


def _raw_plus_i32() -> RawOp:
    source = """
extern "C" __device__ void host_bench_plus_i32(
    void* lhs,
    void* rhs,
    void* result
) {
    *static_cast<int*>(result) =
        *static_cast<int*>(lhs) + *static_cast<int*>(rhs);
}
"""
    return RawOp(ltoir=compile_cpp_op_code(source), name="host_bench_plus_i32")


def _raw_identity_i32() -> RawOp:
    source = """
extern "C" __device__ void host_bench_identity_i32(void* x, void* result) {
    *static_cast<int*>(result) = *static_cast<int*>(x);
}
"""
    return RawOp(ltoir=compile_cpp_op_code(source), name="host_bench_identity_i32")


def _raw_less_i32() -> RawOp:
    source = """
extern "C" __device__ void host_bench_less_i32(
    void* lhs,
    void* rhs,
    void* result
) {
    *static_cast<bool*>(result) =
        *static_cast<int*>(lhs) < *static_cast<int*>(rhs);
}
"""
    return RawOp(ltoir=compile_cpp_op_code(source), name="host_bench_less_i32")


def _raw_equal_i32() -> RawOp:
    source = """
extern "C" __device__ void host_bench_equal_i32(
    void* lhs,
    void* rhs,
    void* result
) {
    *static_cast<bool*>(result) =
        *static_cast<int*>(lhs) == *static_cast<int*>(rhs);
}
"""
    return RawOp(ltoir=compile_cpp_op_code(source), name="host_bench_equal_i32")


def _py_plus_i32(lhs, rhs):
    return lhs + rhs


def _py_identity_i32(x):
    return x


def _py_less_i32(lhs, rhs):
    return lhs < rhs


def _py_equal_i32(lhs, rhs):
    return lhs == rhs


def _py_predicate_i32(x):
    return x < NUM_ITEMS // 2


def _setup_unary_input_output() -> SimpleNamespace:
    d_in = cp.arange(NUM_ITEMS, dtype=cp.int32)
    d_out = cp.empty_like(d_in)
    return SimpleNamespace(d_in=d_in, d_out=d_out, num_items=NUM_ITEMS)


def _setup_binary_input_output() -> SimpleNamespace:
    d_in1 = cp.arange(NUM_ITEMS, dtype=cp.int32)
    d_in2 = cp.arange(NUM_ITEMS, dtype=cp.int32)
    d_out = cp.empty_like(d_in1)
    return SimpleNamespace(d_in1=d_in1, d_in2=d_in2, d_out=d_out, num_items=NUM_ITEMS)


def _setup_reduce() -> SimpleNamespace:
    state = _setup_unary_input_output()
    state.h_init = np.array([0], dtype=np.int32)
    state.op = cc.OpKind.PLUS
    state.temp_storage = make_tiny_temp_storage()
    return state


def _make_reduce(state: SimpleNamespace):
    return cc.make_reduce_into(
        d_in=state.d_in,
        d_out=state.d_out[:1],
        op=state.op,
        h_init=state.h_init,
    )


def _oneshot_reduce(state: SimpleNamespace) -> None:
    cc.reduce_into(
        d_in=state.d_in,
        d_out=state.d_out[:1],
        num_items=state.num_items,
        op=state.op,
        h_init=state.h_init,
    )


def _twoshot_reduce(state: SimpleNamespace, wrapper) -> None:
    wrapper(
        temp_storage=state.temp_storage,
        d_in=state.d_in,
        d_out=state.d_out[:1],
        num_items=state.num_items,
        op=state.op,
        h_init=state.h_init,
    )


def _setup_scan() -> SimpleNamespace:
    state = _setup_unary_input_output()
    state.h_init = np.array([0], dtype=np.int32)
    state.op = cc.OpKind.PLUS
    state.temp_storage = make_tiny_temp_storage()
    return state


def _make_scan(state: SimpleNamespace):
    return cc.make_exclusive_scan(
        d_in=state.d_in,
        d_out=state.d_out,
        op=state.op,
        init_value=state.h_init,
    )


def _oneshot_scan(state: SimpleNamespace) -> None:
    cc.exclusive_scan(
        d_in=state.d_in,
        d_out=state.d_out,
        op=state.op,
        init_value=state.h_init,
        num_items=state.num_items,
    )


def _twoshot_scan(state: SimpleNamespace, wrapper) -> None:
    wrapper(
        temp_storage=state.temp_storage,
        d_in=state.d_in,
        d_out=state.d_out,
        op=state.op,
        init_value=state.h_init,
        num_items=state.num_items,
    )


def _setup_segmented_reduce() -> SimpleNamespace:
    d_in = cp.arange(NUM_ITEMS, dtype=cp.int32)
    d_out = cp.empty(NUM_SEGMENTS, dtype=cp.int32)
    offsets = cp.asarray(
        np.linspace(0, NUM_ITEMS, NUM_SEGMENTS + 1, dtype=np.int64)
    )
    return SimpleNamespace(
        d_in=d_in,
        d_out=d_out,
        start_offsets=offsets[:-1],
        end_offsets=offsets[1:],
        num_segments=NUM_SEGMENTS,
        h_init=np.array([0], dtype=np.int32),
        op=cc.OpKind.PLUS,
        temp_storage=make_tiny_temp_storage(),
    )


def _make_segmented_reduce(state: SimpleNamespace):
    return cc.make_segmented_reduce(
        d_in=state.d_in,
        d_out=state.d_out,
        start_offsets_in=state.start_offsets,
        end_offsets_in=state.end_offsets,
        op=state.op,
        h_init=state.h_init,
    )


def _oneshot_segmented_reduce(state: SimpleNamespace) -> None:
    cc.segmented_reduce(
        d_in=state.d_in,
        d_out=state.d_out,
        num_segments=state.num_segments,
        start_offsets_in=state.start_offsets,
        end_offsets_in=state.end_offsets,
        op=state.op,
        h_init=state.h_init,
    )


def _twoshot_segmented_reduce(state: SimpleNamespace, wrapper) -> None:
    wrapper(
        temp_storage=state.temp_storage,
        d_in=state.d_in,
        d_out=state.d_out,
        num_segments=state.num_segments,
        start_offsets_in=state.start_offsets,
        end_offsets_in=state.end_offsets,
        op=state.op,
        h_init=state.h_init,
    )


def _make_unary_transform(state: SimpleNamespace):
    return cc.make_unary_transform(
        d_in=state.d_in,
        d_out=state.d_out,
        op=state.op,
    )


def _oneshot_unary_transform(state: SimpleNamespace) -> None:
    cc.unary_transform(
        d_in=state.d_in,
        d_out=state.d_out,
        op=state.op,
        num_items=state.num_items,
    )


def _twoshot_unary_transform(state: SimpleNamespace, wrapper) -> None:
    wrapper(
        d_in=state.d_in,
        d_out=state.d_out,
        op=state.op,
        num_items=state.num_items,
    )


def _make_binary_transform(state: SimpleNamespace):
    return cc.make_binary_transform(
        d_in1=state.d_in1,
        d_in2=state.d_in2,
        d_out=state.d_out,
        op=state.op,
    )


def _oneshot_binary_transform(state: SimpleNamespace) -> None:
    cc.binary_transform(
        d_in1=state.d_in1,
        d_in2=state.d_in2,
        d_out=state.d_out,
        op=state.op,
        num_items=state.num_items,
    )


def _twoshot_binary_transform(state: SimpleNamespace, wrapper) -> None:
    wrapper(
        d_in1=state.d_in1,
        d_in2=state.d_in2,
        d_out=state.d_out,
        op=state.op,
        num_items=state.num_items,
    )


def _setup_histogram() -> SimpleNamespace:
    d_samples = cp.arange(NUM_ITEMS, dtype=cp.int32)
    num_output_levels = 17
    d_histogram = cp.empty(num_output_levels - 1, dtype=cp.int32)
    lower_level = np.int32(0)
    upper_level = np.int32(NUM_ITEMS)
    return SimpleNamespace(
        d_samples=d_samples,
        d_histogram=d_histogram,
        num_output_levels=num_output_levels,
        h_num_output_levels=np.array([num_output_levels], dtype=np.int32),
        lower_level=lower_level,
        upper_level=upper_level,
        h_lower_level=np.array([lower_level], dtype=np.int32),
        h_upper_level=np.array([upper_level], dtype=np.int32),
        num_samples=NUM_ITEMS,
        temp_storage=make_tiny_temp_storage(),
    )


def _make_histogram(state: SimpleNamespace):
    return cc.make_histogram_even(
        d_samples=state.d_samples,
        d_histogram=state.d_histogram,
        h_num_output_levels=state.h_num_output_levels,
        h_lower_level=state.h_lower_level,
        h_upper_level=state.h_upper_level,
        num_samples=state.num_samples,
    )


def _oneshot_histogram(state: SimpleNamespace) -> None:
    cc.histogram_even(
        d_samples=state.d_samples,
        d_histogram=state.d_histogram,
        num_output_levels=state.num_output_levels,
        lower_level=state.lower_level,
        upper_level=state.upper_level,
        num_samples=state.num_samples,
    )


def _twoshot_histogram(state: SimpleNamespace, wrapper) -> None:
    wrapper(
        temp_storage=state.temp_storage,
        d_samples=state.d_samples,
        d_histogram=state.d_histogram,
        h_num_output_levels=state.h_num_output_levels,
        h_lower_level=state.h_lower_level,
        h_upper_level=state.h_upper_level,
        num_samples=state.num_samples,
    )


def _setup_binary_search() -> SimpleNamespace:
    d_data = cp.arange(NUM_ITEMS, dtype=cp.int32)
    d_values = cp.arange(0, NUM_ITEMS, 2, dtype=cp.int32)
    d_out = cp.empty(d_values.size, dtype=np.uintp)
    return SimpleNamespace(
        d_data=d_data,
        d_values=d_values,
        d_out=d_out,
        num_items=NUM_ITEMS,
        num_values=int(d_values.size),
        comp=cc.OpKind.LESS,
    )


def _make_lower_bound(state: SimpleNamespace):
    return cc.make_lower_bound(
        d_data=state.d_data,
        d_values=state.d_values,
        d_out=state.d_out,
        comp=state.comp,
    )


def _oneshot_lower_bound(state: SimpleNamespace) -> None:
    cc.lower_bound(
        d_data=state.d_data,
        num_items=state.num_items,
        d_values=state.d_values,
        num_values=state.num_values,
        d_out=state.d_out,
        comp=state.comp,
    )


def _twoshot_lower_bound(state: SimpleNamespace, wrapper) -> None:
    wrapper(
        d_data=state.d_data,
        num_items=state.num_items,
        d_values=state.d_values,
        num_values=state.num_values,
        d_out=state.d_out,
        comp=state.comp,
    )


def _setup_select() -> SimpleNamespace:
    state = _setup_unary_input_output()
    state.d_num_selected = cp.empty(1, dtype=np.uint64)
    state.cond = cc.OpKind.LOGICAL_NOT
    state.temp_storage = make_tiny_temp_storage()
    return state


def _make_select(state: SimpleNamespace):
    return cc.make_select(
        d_in=state.d_in,
        d_out=state.d_out,
        d_num_selected_out=state.d_num_selected,
        cond=state.cond,
    )


def _oneshot_select(state: SimpleNamespace) -> None:
    cc.select(
        d_in=state.d_in,
        d_out=state.d_out,
        d_num_selected_out=state.d_num_selected,
        cond=state.cond,
        num_items=state.num_items,
    )


def _twoshot_select(state: SimpleNamespace, wrapper) -> None:
    wrapper(
        temp_storage=state.temp_storage,
        d_in=state.d_in,
        d_out=state.d_out,
        d_num_selected_out=state.d_num_selected,
        cond=state.cond,
        num_items=state.num_items,
    )


def _setup_three_way_partition() -> SimpleNamespace:
    state = _setup_unary_input_output()
    state.d_first = cp.empty_like(state.d_in)
    state.d_second = cp.empty_like(state.d_in)
    state.d_unselected = cp.empty_like(state.d_in)
    state.d_num_selected = cp.empty(2, dtype=np.uint64)
    state.first_op = cc.OpKind.LOGICAL_NOT
    state.second_op = cc.OpKind.LOGICAL_NOT
    state.temp_storage = make_tiny_temp_storage()
    return state


def _make_three_way_partition(state: SimpleNamespace):
    return cc.make_three_way_partition(
        d_in=state.d_in,
        d_first_part_out=state.d_first,
        d_second_part_out=state.d_second,
        d_unselected_out=state.d_unselected,
        d_num_selected_out=state.d_num_selected,
        select_first_part_op=state.first_op,
        select_second_part_op=state.second_op,
    )


def _oneshot_three_way_partition(state: SimpleNamespace) -> None:
    cc.three_way_partition(
        d_in=state.d_in,
        d_first_part_out=state.d_first,
        d_second_part_out=state.d_second,
        d_unselected_out=state.d_unselected,
        d_num_selected_out=state.d_num_selected,
        select_first_part_op=state.first_op,
        select_second_part_op=state.second_op,
        num_items=state.num_items,
    )


def _twoshot_three_way_partition(state: SimpleNamespace, wrapper) -> None:
    wrapper(
        temp_storage=state.temp_storage,
        d_in=state.d_in,
        d_first_part_out=state.d_first,
        d_second_part_out=state.d_second,
        d_unselected_out=state.d_unselected,
        d_num_selected_out=state.d_num_selected,
        select_first_part_op=state.first_op,
        select_second_part_op=state.second_op,
        num_items=state.num_items,
    )


def _setup_unique_by_key() -> SimpleNamespace:
    d_keys = cp.arange(NUM_ITEMS, dtype=cp.int32)
    d_items = cp.arange(NUM_ITEMS, dtype=cp.int32)
    return SimpleNamespace(
        d_in_keys=d_keys,
        d_in_items=d_items,
        d_out_keys=cp.empty_like(d_keys),
        d_out_items=cp.empty_like(d_items),
        d_num_selected=cp.empty(1, dtype=np.uint64),
        op=cc.OpKind.EQUAL_TO,
        num_items=NUM_ITEMS,
        temp_storage=make_tiny_temp_storage(),
    )


def _make_unique_by_key(state: SimpleNamespace):
    return cc.make_unique_by_key(
        d_in_keys=state.d_in_keys,
        d_in_items=state.d_in_items,
        d_out_keys=state.d_out_keys,
        d_out_items=state.d_out_items,
        d_out_num_selected=state.d_num_selected,
        op=state.op,
    )


def _oneshot_unique_by_key(state: SimpleNamespace) -> None:
    cc.unique_by_key(
        d_in_keys=state.d_in_keys,
        d_in_items=state.d_in_items,
        d_out_keys=state.d_out_keys,
        d_out_items=state.d_out_items,
        d_out_num_selected=state.d_num_selected,
        op=state.op,
        num_items=state.num_items,
    )


def _twoshot_unique_by_key(state: SimpleNamespace, wrapper) -> None:
    wrapper(
        temp_storage=state.temp_storage,
        d_in_keys=state.d_in_keys,
        d_in_items=state.d_in_items,
        d_out_keys=state.d_out_keys,
        d_out_items=state.d_out_items,
        d_out_num_selected=state.d_num_selected,
        op=state.op,
        num_items=state.num_items,
    )


def _setup_sort() -> SimpleNamespace:
    d_in_keys = cp.arange(NUM_ITEMS, 0, -1, dtype=cp.int32)
    d_out_keys = cp.empty_like(d_in_keys)
    return SimpleNamespace(
        d_in_keys=d_in_keys,
        d_out_keys=d_out_keys,
        op=cc.OpKind.LESS,
        num_items=NUM_ITEMS,
        temp_storage=make_tiny_temp_storage(),
    )


def _make_merge_sort(state: SimpleNamespace):
    return cc.make_merge_sort(
        d_in_keys=state.d_in_keys,
        d_out_keys=state.d_out_keys,
        op=state.op,
    )


def _oneshot_merge_sort(state: SimpleNamespace) -> None:
    cc.merge_sort(
        d_in_keys=state.d_in_keys,
        d_out_keys=state.d_out_keys,
        num_items=state.num_items,
        op=state.op,
    )


def _twoshot_merge_sort(state: SimpleNamespace, wrapper) -> None:
    wrapper(
        temp_storage=state.temp_storage,
        d_in_keys=state.d_in_keys,
        d_in_values=None,
        d_out_keys=state.d_out_keys,
        d_out_values=None,
        num_items=state.num_items,
        op=state.op,
    )


def _make_radix_sort(state: SimpleNamespace):
    return cc.make_radix_sort(
        d_in_keys=state.d_in_keys,
        d_out_keys=state.d_out_keys,
        d_in_values=None,
        d_out_values=None,
        order=cc.SortOrder.ASCENDING,
    )


def _oneshot_radix_sort(state: SimpleNamespace) -> None:
    cc.radix_sort(
        d_in_keys=state.d_in_keys,
        d_out_keys=state.d_out_keys,
        num_items=state.num_items,
        order=cc.SortOrder.ASCENDING,
    )


def _twoshot_radix_sort(state: SimpleNamespace, wrapper) -> None:
    wrapper(
        temp_storage=state.temp_storage,
        d_in_keys=state.d_in_keys,
        d_out_keys=state.d_out_keys,
        d_in_values=None,
        d_out_values=None,
        num_items=state.num_items,
    )


def _setup_segmented_sort() -> SimpleNamespace:
    state = _setup_sort()
    offsets = cp.asarray(
        np.linspace(0, NUM_ITEMS, NUM_SEGMENTS + 1, dtype=np.int64)
    )
    state.start_offsets = offsets[:-1]
    state.end_offsets = offsets[1:]
    state.num_segments = NUM_SEGMENTS
    return state


def _make_segmented_sort(state: SimpleNamespace):
    return cc.make_segmented_sort(
        d_in_keys=state.d_in_keys,
        d_out_keys=state.d_out_keys,
        d_in_values=None,
        d_out_values=None,
        start_offsets_in=state.start_offsets,
        end_offsets_in=state.end_offsets,
        order=cc.SortOrder.ASCENDING,
    )


def _oneshot_segmented_sort(state: SimpleNamespace) -> None:
    cc.segmented_sort(
        d_in_keys=state.d_in_keys,
        d_out_keys=state.d_out_keys,
        d_in_values=None,
        d_out_values=None,
        num_items=state.num_items,
        num_segments=state.num_segments,
        start_offsets_in=state.start_offsets,
        end_offsets_in=state.end_offsets,
        order=cc.SortOrder.ASCENDING,
    )


def _twoshot_segmented_sort(state: SimpleNamespace, wrapper) -> None:
    wrapper(
        temp_storage=state.temp_storage,
        d_in_keys=state.d_in_keys,
        d_out_keys=state.d_out_keys,
        d_in_values=None,
        d_out_values=None,
        num_items=state.num_items,
        num_segments=state.num_segments,
        start_offsets_in=state.start_offsets,
        end_offsets_in=state.end_offsets,
    )


def _setup_with_values(
    setup_fn: Callable[[], SimpleNamespace], **values: Any
) -> Callable[[], SimpleNamespace]:
    def setup() -> SimpleNamespace:
        state = setup_fn()
        for name, value in values.items():
            setattr(state, name, value)
        return state

    return setup


def _setup_with_factories(
    setup_fn: Callable[[], SimpleNamespace], **factories: Callable[[], Any]
) -> Callable[[], SimpleNamespace]:
    def setup() -> SimpleNamespace:
        state = setup_fn()
        for name, factory in factories.items():
            setattr(state, name, factory())
        return state

    return setup


def _make_case(
    name: str,
    setup: Callable[[], SimpleNamespace],
    make_wrapper: Callable[[SimpleNamespace], Any],
    oneshot: Callable[[SimpleNamespace], None],
    twoshot: Callable[[SimpleNamespace, Any], None],
    noop_return_kind: NoopReturnKind,
    skip_reason: str | None = None,
) -> HostBenchmarkCase:
    return HostBenchmarkCase(
        name,
        setup,
        make_wrapper,
        oneshot,
        twoshot,
        noop_return_kind,
        skip_reason,
    )


CASES = [
    _make_case(
        "reduce.plus",
        _setup_with_values(_setup_reduce, op=cc.OpKind.PLUS),
        _make_reduce,
        _oneshot_reduce,
        _twoshot_reduce,
        "temp_storage_bytes",
    ),
    _make_case(
        "reduce.raw_cpp",
        _setup_with_factories(_setup_reduce, op=_raw_plus_i32),
        _make_reduce,
        _oneshot_reduce,
        _twoshot_reduce,
        "temp_storage_bytes",
    ),
    _make_case(
        "reduce.python",
        _setup_with_values(_setup_reduce, op=_py_plus_i32),
        _make_reduce,
        _oneshot_reduce,
        _twoshot_reduce,
        "temp_storage_bytes",
        _NUMBA_CUDA_SKIP_REASON,
    ),
    _make_case(
        "exclusive_scan.plus",
        _setup_with_values(_setup_scan, op=cc.OpKind.PLUS),
        _make_scan,
        _oneshot_scan,
        _twoshot_scan,
        "temp_storage_bytes",
    ),
    _make_case(
        "exclusive_scan.raw_cpp",
        _setup_with_factories(_setup_scan, op=_raw_plus_i32),
        _make_scan,
        _oneshot_scan,
        _twoshot_scan,
        "temp_storage_bytes",
    ),
    _make_case(
        "exclusive_scan.python",
        _setup_with_values(_setup_scan, op=_py_plus_i32),
        _make_scan,
        _oneshot_scan,
        _twoshot_scan,
        "temp_storage_bytes",
        _NUMBA_CUDA_SKIP_REASON,
    ),
    _make_case(
        "segmented_reduce.plus",
        _setup_with_values(_setup_segmented_reduce, op=cc.OpKind.PLUS),
        _make_segmented_reduce,
        _oneshot_segmented_reduce,
        _twoshot_segmented_reduce,
        "temp_storage_bytes",
    ),
    _make_case(
        "segmented_reduce.raw_cpp",
        _setup_with_factories(_setup_segmented_reduce, op=_raw_plus_i32),
        _make_segmented_reduce,
        _oneshot_segmented_reduce,
        _twoshot_segmented_reduce,
        "temp_storage_bytes",
    ),
    _make_case(
        "segmented_reduce.python",
        _setup_with_values(_setup_segmented_reduce, op=_py_plus_i32),
        _make_segmented_reduce,
        _oneshot_segmented_reduce,
        _twoshot_segmented_reduce,
        "temp_storage_bytes",
        _NUMBA_CUDA_SKIP_REASON,
    ),
    _make_case(
        "unary_transform.identity",
        _setup_with_values(_setup_unary_input_output, op=cc.OpKind.IDENTITY),
        _make_unary_transform,
        _oneshot_unary_transform,
        _twoshot_unary_transform,
        "none",
    ),
    _make_case(
        "unary_transform.raw_cpp",
        _setup_with_factories(_setup_unary_input_output, op=_raw_identity_i32),
        _make_unary_transform,
        _oneshot_unary_transform,
        _twoshot_unary_transform,
        "none",
    ),
    _make_case(
        "unary_transform.python",
        _setup_with_values(_setup_unary_input_output, op=_py_identity_i32),
        _make_unary_transform,
        _oneshot_unary_transform,
        _twoshot_unary_transform,
        "none",
        _NUMBA_CUDA_SKIP_REASON,
    ),
    _make_case(
        "binary_transform.plus",
        _setup_with_values(_setup_binary_input_output, op=cc.OpKind.PLUS),
        _make_binary_transform,
        _oneshot_binary_transform,
        _twoshot_binary_transform,
        "none",
    ),
    _make_case(
        "binary_transform.raw_cpp",
        _setup_with_factories(_setup_binary_input_output, op=_raw_plus_i32),
        _make_binary_transform,
        _oneshot_binary_transform,
        _twoshot_binary_transform,
        "none",
    ),
    _make_case(
        "binary_transform.python",
        _setup_with_values(_setup_binary_input_output, op=_py_plus_i32),
        _make_binary_transform,
        _oneshot_binary_transform,
        _twoshot_binary_transform,
        "none",
        _NUMBA_CUDA_SKIP_REASON,
    ),
    _make_case(
        "histogram_even",
        _setup_histogram,
        _make_histogram,
        _oneshot_histogram,
        _twoshot_histogram,
        "temp_storage_bytes",
    ),
    _make_case(
        "lower_bound.less",
        _setup_with_values(_setup_binary_search, comp=cc.OpKind.LESS),
        _make_lower_bound,
        _oneshot_lower_bound,
        _twoshot_lower_bound,
        "none",
    ),
    _make_case(
        "lower_bound.raw_cpp",
        _setup_with_factories(_setup_binary_search, comp=_raw_less_i32),
        _make_lower_bound,
        _oneshot_lower_bound,
        _twoshot_lower_bound,
        "none",
    ),
    _make_case(
        "lower_bound.python",
        _setup_with_values(_setup_binary_search, comp=_py_less_i32),
        _make_lower_bound,
        _oneshot_lower_bound,
        _twoshot_lower_bound,
        "none",
        _NUMBA_CUDA_SKIP_REASON,
    ),
    _make_case(
        "select.logical_not",
        _setup_with_values(_setup_select, cond=cc.OpKind.LOGICAL_NOT),
        _make_select,
        _oneshot_select,
        _twoshot_select,
        "temp_storage_bytes",
    ),
    _make_case(
        "select.raw_cpp",
        _setup_with_factories(
            _setup_select,
            cond=lambda: _raw_predicate_i32("host_bench_select_predicate_i32"),
        ),
        _make_select,
        _oneshot_select,
        _twoshot_select,
        "temp_storage_bytes",
    ),
    _make_case(
        "select.python",
        _setup_with_values(_setup_select, cond=_py_predicate_i32),
        _make_select,
        _oneshot_select,
        _twoshot_select,
        "temp_storage_bytes",
        _NUMBA_CUDA_SKIP_REASON,
    ),
    _make_case(
        "three_way_partition.logical_not",
        _setup_with_values(
            _setup_three_way_partition,
            first_op=cc.OpKind.LOGICAL_NOT,
            second_op=cc.OpKind.LOGICAL_NOT,
        ),
        _make_three_way_partition,
        _oneshot_three_way_partition,
        _twoshot_three_way_partition,
        "temp_storage_bytes",
    ),
    _make_case(
        "three_way_partition.raw_cpp",
        _setup_with_factories(
            _setup_three_way_partition,
            first_op=lambda: _raw_predicate_i32("host_bench_partition_first_i32"),
            second_op=lambda: _raw_predicate_i32("host_bench_partition_second_i32"),
        ),
        _make_three_way_partition,
        _oneshot_three_way_partition,
        _twoshot_three_way_partition,
        "temp_storage_bytes",
    ),
    _make_case(
        "three_way_partition.python",
        _setup_with_values(
            _setup_three_way_partition,
            first_op=_py_predicate_i32,
            second_op=_py_predicate_i32,
        ),
        _make_three_way_partition,
        _oneshot_three_way_partition,
        _twoshot_three_way_partition,
        "temp_storage_bytes",
        _NUMBA_CUDA_SKIP_REASON,
    ),
    _make_case(
        "unique_by_key.equal",
        _setup_with_values(_setup_unique_by_key, op=cc.OpKind.EQUAL_TO),
        _make_unique_by_key,
        _oneshot_unique_by_key,
        _twoshot_unique_by_key,
        "temp_storage_bytes",
    ),
    _make_case(
        "unique_by_key.raw_cpp",
        _setup_with_factories(_setup_unique_by_key, op=_raw_equal_i32),
        _make_unique_by_key,
        _oneshot_unique_by_key,
        _twoshot_unique_by_key,
        "temp_storage_bytes",
    ),
    _make_case(
        "unique_by_key.python",
        _setup_with_values(_setup_unique_by_key, op=_py_equal_i32),
        _make_unique_by_key,
        _oneshot_unique_by_key,
        _twoshot_unique_by_key,
        "temp_storage_bytes",
        _NUMBA_CUDA_SKIP_REASON,
    ),
    _make_case(
        "merge_sort.less",
        _setup_with_values(_setup_sort, op=cc.OpKind.LESS),
        _make_merge_sort,
        _oneshot_merge_sort,
        _twoshot_merge_sort,
        "temp_storage_bytes",
    ),
    _make_case(
        "merge_sort.raw_cpp",
        _setup_with_factories(_setup_sort, op=_raw_less_i32),
        _make_merge_sort,
        _oneshot_merge_sort,
        _twoshot_merge_sort,
        "temp_storage_bytes",
    ),
    _make_case(
        "merge_sort.python",
        _setup_with_values(_setup_sort, op=_py_less_i32),
        _make_merge_sort,
        _oneshot_merge_sort,
        _twoshot_merge_sort,
        "temp_storage_bytes",
        _NUMBA_CUDA_SKIP_REASON,
    ),
    _make_case(
        "radix_sort",
        _setup_sort,
        _make_radix_sort,
        _oneshot_radix_sort,
        _twoshot_radix_sort,
        "temp_storage_and_selector",
    ),
    _make_case(
        "segmented_sort",
        _setup_segmented_sort,
        _make_segmented_sort,
        _oneshot_segmented_sort,
        _twoshot_segmented_sort,
        "temp_storage_and_selector",
    ),
]
