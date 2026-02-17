import cupy as cp
import numpy as np
import pytest

import cuda.compute
from cuda.compute import (
    CacheModifiedInputIterator,
    gpu_struct,
)


def select_pointer(inp, out, num_selected, build_only):
    size = len(inp)

    def even_op(x):
        return x % 2 == 0

    selector = cuda.compute.make_select(inp, out, num_selected, even_op)
    if not build_only:
        temp_bytes = selector(None, inp, out, num_selected, even_op, size)
        temp_storage = cp.empty(temp_bytes, dtype=np.uint8)
        selector(temp_storage, inp, out, num_selected, even_op, size)

    cp.cuda.runtime.deviceSynchronize()


def select_iterator(size, d_in, out, num_selected, build_only):
    d_in_iter = CacheModifiedInputIterator(d_in, modifier="stream")

    def less_than_50(x):
        return x < 50

    selector = cuda.compute.make_select(d_in_iter, out, num_selected, less_than_50)
    if not build_only:
        temp_bytes = selector(None, d_in_iter, out, num_selected, less_than_50, size)
        temp_storage = cp.empty(temp_bytes, dtype=np.uint8)
        selector(temp_storage, d_in_iter, out, num_selected, less_than_50, size)

    cp.cuda.runtime.deviceSynchronize()


@gpu_struct
class Point:
    x: np.int32
    y: np.int32


def select_struct(inp, out, num_selected, build_only):
    size = len(inp)

    def in_first_quadrant(p: Point) -> np.uint8:
        return (p.x > 50) and (p.y > 50)

    selector = cuda.compute.make_select(inp, out, num_selected, in_first_quadrant)
    if not build_only:
        temp_bytes = selector(None, inp, out, num_selected, in_first_quadrant, size)
        temp_storage = cp.empty(temp_bytes, dtype=np.uint8)
        selector(temp_storage, inp, out, num_selected, in_first_quadrant, size)

    cp.cuda.runtime.deviceSynchronize()


def select_stateful(inp, out, num_selected, threshold_state, build_only):
    size = len(inp)

    def threshold_select(x):
        return x > threshold_state[0]

    selector = cuda.compute.make_select(inp, out, num_selected, threshold_select)
    if not build_only:
        temp_bytes = selector(None, inp, out, num_selected, threshold_select, size)
        temp_storage = cp.empty(temp_bytes, dtype=np.uint8)
        selector(temp_storage, inp, out, num_selected, threshold_select, size)

    cp.cuda.runtime.deviceSynchronize()


@pytest.mark.parametrize("bench_fixture", ["compile_benchmark", "benchmark"])
def bench_select_pointer(bench_fixture, request, size):
    actual_size = 100 if bench_fixture == "compile_benchmark" else size
    inp = cp.random.randint(0, 100, actual_size, dtype=np.int32)
    out = cp.empty_like(inp)
    num_selected = cp.empty(2, dtype=np.uint64)

    def run():
        select_pointer(
            inp, out, num_selected, build_only=(bench_fixture == "compile_benchmark")
        )

    fixture = request.getfixturevalue(bench_fixture)
    fixture(run)


@pytest.mark.parametrize("bench_fixture", ["compile_benchmark", "benchmark"])
def bench_select_iterator(bench_fixture, request, size):
    actual_size = 100 if bench_fixture == "compile_benchmark" else size
    d_in = cp.random.randint(0, 100, actual_size, dtype=np.int32)
    out = cp.empty(actual_size, dtype=np.int32)
    num_selected = cp.empty(2, dtype=np.uint64)

    def run():
        select_iterator(
            actual_size,
            d_in,
            out,
            num_selected,
            build_only=(bench_fixture == "compile_benchmark"),
        )

    fixture = request.getfixturevalue(bench_fixture)
    fixture(run)


@pytest.mark.parametrize("bench_fixture", ["compile_benchmark", "benchmark"])
def bench_select_struct(bench_fixture, request, size):
    actual_size = 100 if bench_fixture == "compile_benchmark" else size
    inp = cp.random.randint(0, 100, (actual_size, 2), dtype=np.int32).view(Point.dtype)
    out = cp.empty_like(inp)
    num_selected = cp.empty(2, dtype=np.uint64)

    def run():
        select_struct(
            inp, out, num_selected, build_only=(bench_fixture == "compile_benchmark")
        )

    fixture = request.getfixturevalue(bench_fixture)
    fixture(run)


@pytest.mark.parametrize("bench_fixture", ["compile_benchmark", "benchmark"])
def bench_select_stateful(bench_fixture, request, size):
    actual_size = 100 if bench_fixture == "compile_benchmark" else size
    inp = cp.random.randint(0, 100, actual_size, dtype=np.int32)
    out = cp.empty_like(inp)
    num_selected = cp.empty(2, dtype=np.uint64)
    threshold_state = cp.array([50], dtype=np.int32)

    def run():
        select_stateful(
            inp,
            out,
            num_selected,
            threshold_state,
            build_only=(bench_fixture == "compile_benchmark"),
        )

    fixture = request.getfixturevalue(bench_fixture)
    fixture(run)
