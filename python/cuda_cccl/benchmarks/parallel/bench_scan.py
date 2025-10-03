import cupy as cp
import numpy as np
import pytest

import cuda.compute as cc


def scan_pointer(input_array, build_only, scan_type):
    size = len(input_array)
    res = cp.empty_like(input_array)
    h_init = np.zeros(tuple(), dtype=input_array.dtype)

    if scan_type == "exclusive":
        alg = cc.make_exclusive_scan(input_array, res, cc.OpKind.PLUS, h_init)
    else:  # inclusive
        alg = cc.make_inclusive_scan(input_array, res, cc.OpKind.PLUS, h_init)

    if not build_only:
        temp_storage_bytes = alg(None, input_array, res, size, h_init)
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
        alg(temp_storage, input_array, res, size, h_init)

    cp.cuda.runtime.deviceSynchronize()


def scan_pointer_custom_op(input_array, build_only, scan_type):
    size = len(input_array)
    res = cp.empty_like(input_array)
    h_init = np.zeros(tuple(), dtype=input_array.dtype)

    def my_add(a, b):
        return a + b

    if scan_type == "exclusive":
        alg = cc.make_exclusive_scan(input_array, res, my_add, h_init)
    else:  # inclusive
        alg = cc.make_inclusive_scan(input_array, res, my_add, h_init)

    if not build_only:
        temp_storage_bytes = alg(None, input_array, res, size, h_init)
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
        alg(temp_storage, input_array, res, size, h_init)

    cp.cuda.runtime.deviceSynchronize()


def scan_struct(input_array, build_only, scan_type):
    size = len(input_array)
    res = cp.empty_like(input_array)
    h_init = MyStruct(0, 0)

    def my_add(a, b):
        return MyStruct(a.x + b.x, a.y + b.y)

    if scan_type == "exclusive":
        alg = cc.make_exclusive_scan(input_array, res, my_add, h_init)
    else:  # inclusive
        alg = cc.make_inclusive_scan(input_array, res, my_add, h_init)

    if not build_only:
        temp_storage_bytes = alg(None, input_array, res, size, h_init)
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
        alg(temp_storage, input_array, res, size, h_init)

    cp.cuda.runtime.deviceSynchronize()


def scan_iterator(inp, size, build_only, scan_type):
    res = cp.empty(size, dtype=np.int32)
    h_init = np.zeros(tuple(), dtype=np.int32)

    if scan_type == "exclusive":
        alg = cc.make_exclusive_scan(inp, res, cc.OpKind.PLUS, h_init)
    else:  # inclusive
        alg = cc.make_inclusive_scan(inp, res, cc.OpKind.PLUS, h_init)

    if not build_only:
        temp_storage_bytes = alg(None, inp, res, size, h_init)
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
        alg(temp_storage, inp, res, size, h_init)

    cp.cuda.runtime.deviceSynchronize()


@cc.gpu_struct
class MyStruct:
    x: np.int32
    y: np.int32


@pytest.mark.parametrize("scan_type", ["exclusive", "inclusive"])
@pytest.mark.parametrize("bench_fixture", ["compile_benchmark", "benchmark"])
def bench_scan_pointer(bench_fixture, request, size, scan_type):
    # Use small size for compile benchmarks, parameterized size for runtime benchmarks
    actual_size = 10 if bench_fixture == "compile_benchmark" else size
    input_array = cp.random.randint(0, 10, actual_size)

    def run():
        scan_pointer(
            input_array,
            build_only=(bench_fixture == "compile_benchmark"),
            scan_type=scan_type,
        )

    fixture = request.getfixturevalue(bench_fixture)
    if bench_fixture == "compile_benchmark":
        if scan_type == "exclusive":
            fixture(cc.make_exclusive_scan, run)
        else:
            fixture(cc.make_inclusive_scan, run)
    else:
        fixture(run)


@pytest.mark.parametrize("scan_type", ["exclusive", "inclusive"])
@pytest.mark.parametrize("bench_fixture", ["benchmark"])
def bench_scan_pointer_custom_op(bench_fixture, request, size, scan_type):
    input_array = cp.random.randint(0, 10, size)

    def run():
        scan_pointer_custom_op(input_array, build_only=False, scan_type=scan_type)

    fixture = request.getfixturevalue(bench_fixture)
    fixture(run)


@pytest.mark.parametrize("scan_type", ["exclusive", "inclusive"])
@pytest.mark.parametrize("bench_fixture", ["compile_benchmark", "benchmark"])
def bench_scan_iterator(bench_fixture, request, size, scan_type):
    inp = cc.CountingIterator(np.int32(0))
    # Use small size for compile benchmarks, parameterized size for runtime benchmarks
    actual_size = 10 if bench_fixture == "compile_benchmark" else size

    def run():
        scan_iterator(
            inp,
            actual_size,
            build_only=(bench_fixture == "compile_benchmark"),
            scan_type=scan_type,
        )

    fixture = request.getfixturevalue(bench_fixture)
    if bench_fixture == "compile_benchmark":
        if scan_type == "exclusive":
            fixture(cc.make_exclusive_scan, run)
        else:
            fixture(cc.make_inclusive_scan, run)
    else:
        fixture(run)


@pytest.mark.parametrize("scan_type", ["exclusive", "inclusive"])
@pytest.mark.parametrize("bench_fixture", ["compile_benchmark", "benchmark"])
def bench_scan_struct(bench_fixture, request, size, scan_type):
    # Use small size for compile benchmarks, parameterized size for runtime benchmarks
    actual_size = 10 if bench_fixture == "compile_benchmark" else size
    input_array = cp.random.randint(0, 10, (actual_size, 2), dtype="int32").view(
        MyStruct
    )

    def run():
        scan_struct(
            input_array,
            build_only=(bench_fixture == "compile_benchmark"),
            scan_type=scan_type,
        )

    fixture = request.getfixturevalue(bench_fixture)
    if bench_fixture == "compile_benchmark":
        if scan_type == "exclusive":
            fixture(cc.make_exclusive_scan, run)
        else:
            fixture(cc.make_inclusive_scan, run)
    else:
        fixture(run)


def scan_pointer_single_phase(input_array, build_only, scan_type):
    size = len(input_array)
    res = cp.empty_like(input_array)
    h_init = np.zeros(tuple(), dtype=input_array.dtype)

    def my_add(a, b):
        return a + b

    if scan_type == "exclusive":
        cc.exclusive_scan(input_array, res, my_add, h_init, size)
    else:  # inclusive
        cc.inclusive_scan(input_array, res, my_add, h_init, size)

    cp.cuda.runtime.deviceSynchronize()


def scan_struct_single_phase(input_array, build_only, scan_type):
    size = len(input_array)
    res = cp.empty_like(input_array)
    h_init = MyStruct(0, 0)

    def my_add(a, b):
        return MyStruct(a.x + b.x, a.y + b.y)

    if scan_type == "exclusive":
        cc.exclusive_scan(input_array, res, my_add, h_init, size)
    else:  # inclusive
        cc.inclusive_scan(input_array, res, my_add, h_init, size)

    cp.cuda.runtime.deviceSynchronize()


def scan_iterator_single_phase(inp, size, build_only, scan_type):
    res = cp.empty(size, dtype=np.int32)
    h_init = np.zeros(tuple(), dtype=np.int32)

    def my_add(a, b):
        return a + b

    if scan_type == "exclusive":
        cc.exclusive_scan(inp, res, my_add, h_init, size)
    else:  # inclusive
        cc.inclusive_scan(inp, res, my_add, h_init, size)

    cp.cuda.runtime.deviceSynchronize()


@pytest.mark.parametrize("scan_type", ["exclusive", "inclusive"])
@pytest.mark.parametrize("bench_fixture", ["benchmark"])
def bench_scan_pointer_single_phase(bench_fixture, request, size, scan_type):
    input_array = cp.random.randint(0, 10, size)

    # warm up run
    scan_pointer_single_phase(input_array, build_only=False, scan_type=scan_type)

    # benchmark run
    def run():
        scan_pointer_single_phase(input_array, build_only=False, scan_type=scan_type)

    fixture = request.getfixturevalue(bench_fixture)
    fixture(run)


@pytest.mark.parametrize("scan_type", ["exclusive", "inclusive"])
@pytest.mark.parametrize("bench_fixture", ["benchmark"])
def bench_scan_iterator_single_phase(bench_fixture, request, size, scan_type):
    inp = cc.CountingIterator(np.int32(0))

    # warm up run
    scan_iterator_single_phase(inp, size, build_only=False, scan_type=scan_type)

    # benchmark run
    def run():
        scan_iterator_single_phase(inp, size, build_only=False, scan_type=scan_type)

    fixture = request.getfixturevalue(bench_fixture)
    fixture(run)


@pytest.mark.parametrize("scan_type", ["exclusive", "inclusive"])
@pytest.mark.parametrize("bench_fixture", ["benchmark"])
def bench_scan_struct_single_phase(bench_fixture, request, size, scan_type):
    input_array = cp.random.randint(0, 10, (size, 2), dtype="int32").view(MyStruct)

    # warm up run
    scan_struct_single_phase(input_array, build_only=False, scan_type=scan_type)

    # benchmark run
    def run():
        scan_struct_single_phase(input_array, build_only=False, scan_type=scan_type)

    fixture = request.getfixturevalue(bench_fixture)
    fixture(run)
