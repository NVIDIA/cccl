import cupy as cp
import numpy as np
import pytest

import cuda.compute
from cuda.compute import CountingIterator, gpu_struct


def three_way_partition_pointer(
    inp, first_out, second_out, unselected_out, num_selected, build_only
):
    size = len(inp)

    def less_than_op(x):
        return x < 42

    def greater_equal_op(x):
        return x >= 42

    partitioner = cuda.compute.make_three_way_partition(
        inp,
        first_out,
        second_out,
        unselected_out,
        num_selected,
        less_than_op,
        greater_equal_op,
    )

    if not build_only:
        temp_storage_bytes = partitioner(
            None,
            inp,
            first_out,
            second_out,
            unselected_out,
            num_selected,
            less_than_op,
            greater_equal_op,
            size,
        )
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
        partitioner(
            temp_storage,
            inp,
            first_out,
            second_out,
            unselected_out,
            num_selected,
            less_than_op,
            greater_equal_op,
            size,
        )

    cp.cuda.runtime.deviceSynchronize()


def three_way_partition_iterator(
    size, first_out, second_out, unselected_out, num_selected, build_only
):
    in_it = CountingIterator(np.int32(0))

    def less_than_op(x):
        return x < 42

    def greater_equal_op(x):
        return x >= 42

    partitioner = cuda.compute.make_three_way_partition(
        in_it,
        first_out,
        second_out,
        unselected_out,
        num_selected,
        less_than_op,
        greater_equal_op,
    )

    if not build_only:
        temp_storage_bytes = partitioner(
            None,
            in_it,
            first_out,
            second_out,
            unselected_out,
            num_selected,
            less_than_op,
            greater_equal_op,
            size,
        )
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
        partitioner(
            temp_storage,
            in_it,
            first_out,
            second_out,
            unselected_out,
            num_selected,
            less_than_op,
            greater_equal_op,
            size,
        )

    cp.cuda.runtime.deviceSynchronize()


@gpu_struct
class MyStruct:
    x: np.int32
    y: np.int32


def three_way_partition_struct(
    inp, first_out, second_out, unselected_out, num_selected, build_only
):
    size = len(inp)

    def less_than_op(a: MyStruct):
        return (a.x < 42) & (a.y < 42)

    def greater_equal_op(a: MyStruct):
        return (a.x >= 42) & (a.y >= 42)

    partitioner = cuda.compute.make_three_way_partition(
        inp,
        first_out,
        second_out,
        unselected_out,
        num_selected,
        less_than_op,
        greater_equal_op,
    )

    if not build_only:
        temp_storage_bytes = partitioner(
            None,
            inp,
            first_out,
            second_out,
            unselected_out,
            num_selected,
            less_than_op,
            greater_equal_op,
            size,
        )
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
        partitioner(
            temp_storage,
            inp,
            first_out,
            second_out,
            unselected_out,
            num_selected,
            less_than_op,
            greater_equal_op,
            size,
        )

    cp.cuda.runtime.deviceSynchronize()


@pytest.mark.parametrize("bench_fixture", ["compile_benchmark", "benchmark"])
def bench_three_way_partition_pointer(bench_fixture, request, size):
    actual_size = 100 if bench_fixture == "compile_benchmark" else size
    d_in = cp.random.randint(0, 100, actual_size)
    d_first = cp.empty_like(d_in)
    d_second = cp.empty_like(d_in)
    d_unselected = cp.empty_like(d_in)
    d_num_selected = cp.empty(2, dtype=np.int32)

    def run():
        three_way_partition_pointer(
            d_in,
            d_first,
            d_second,
            d_unselected,
            d_num_selected,
            build_only=(bench_fixture == "compile_benchmark"),
        )

    fixture = request.getfixturevalue(bench_fixture)
    fixture(run)


@pytest.mark.parametrize("bench_fixture", ["compile_benchmark", "benchmark"])
def bench_three_way_partition_iterator(bench_fixture, request, size):
    actual_size = 100 if bench_fixture == "compile_benchmark" else size
    d_first = cp.empty(actual_size, dtype=np.int32)
    d_second = cp.empty(actual_size, dtype=np.int32)
    d_unselected = cp.empty(actual_size, dtype=np.int32)
    d_num_selected = cp.empty(2, dtype=np.int32)

    def run():
        three_way_partition_iterator(
            actual_size,
            d_first,
            d_second,
            d_unselected,
            d_num_selected,
            build_only=(bench_fixture == "compile_benchmark"),
        )

    fixture = request.getfixturevalue(bench_fixture)
    fixture(run)


@pytest.mark.parametrize("bench_fixture", ["compile_benchmark", "benchmark"])
def bench_three_way_partition_struct(bench_fixture, request, size):
    actual_size = 100 if bench_fixture == "compile_benchmark" else size
    d_in = cp.random.randint(0, 100, (actual_size, 2)).view(MyStruct.dtype)
    d_first = cp.empty_like(d_in)
    d_second = cp.empty_like(d_in)
    d_unselected = cp.empty_like(d_in)
    d_num_selected = cp.empty(2, dtype=np.int32)

    def run():
        three_way_partition_struct(
            d_in,
            d_first,
            d_second,
            d_unselected,
            d_num_selected,
            build_only=(bench_fixture == "compile_benchmark"),
        )

    fixture = request.getfixturevalue(bench_fixture)
    fixture(run)


def three_way_partition_pointer_single_phase(inp):
    size = len(inp)
    d_first = cp.empty_like(inp)
    d_second = cp.empty_like(inp)
    d_unselected = cp.empty_like(inp)
    d_num_selected = cp.empty(2, dtype=np.int32)

    def less_than_op(x):
        return x < 42

    def greater_equal_op(x):
        return x >= 42

    cuda.compute.three_way_partition(
        inp,
        d_first,
        d_second,
        d_unselected,
        d_num_selected,
        less_than_op,
        greater_equal_op,
        size,
    )
    cp.cuda.runtime.deviceSynchronize()


def three_way_partition_iterator_single_phase(size):
    in_it = CountingIterator(np.int32(0))
    d_first = cp.empty(size, dtype=np.int32)
    d_second = cp.empty(size, dtype=np.int32)
    d_unselected = cp.empty(size, dtype=np.int32)
    d_num_selected = cp.empty(2, dtype=np.int32)

    def less_than_op(x):
        return x < 42

    def greater_equal_op(x):
        return x >= 42

    cuda.compute.three_way_partition(
        in_it,
        d_first,
        d_second,
        d_unselected,
        d_num_selected,
        less_than_op,
        greater_equal_op,
        size,
    )
    cp.cuda.runtime.deviceSynchronize()


def three_way_partition_struct_single_phase(size):
    d_in = cp.random.randint(0, 100, (size, 2)).view(MyStruct.dtype)
    d_first = cp.empty_like(d_in)
    d_second = cp.empty_like(d_in)
    d_unselected = cp.empty_like(d_in)
    d_num_selected = cp.empty(2, dtype=np.int32)

    def less_than_op(a: MyStruct):
        return (a.x < 42) & (a.y < 42)

    def greater_equal_op(a: MyStruct):
        return (a.x >= 42) & (a.y >= 42)

    cuda.compute.three_way_partition(
        d_in,
        d_first,
        d_second,
        d_unselected,
        d_num_selected,
        less_than_op,
        greater_equal_op,
        size,
    )
    cp.cuda.runtime.deviceSynchronize()


@pytest.mark.parametrize("bench_fixture", ["benchmark"])
def bench_three_way_partition_pointer_single_phase(bench_fixture, request, size):
    d_in = cp.random.randint(0, 100, size)

    # warm up run
    three_way_partition_pointer_single_phase(d_in)

    def run():
        three_way_partition_pointer_single_phase(d_in)

    fixture = request.getfixturevalue(bench_fixture)
    fixture(run)


@pytest.mark.parametrize("bench_fixture", ["benchmark"])
def bench_three_way_partition_iterator_single_phase(bench_fixture, request, size):
    # warm up run
    three_way_partition_iterator_single_phase(size)

    def run():
        three_way_partition_iterator_single_phase(size)

    fixture = request.getfixturevalue(bench_fixture)
    fixture(run)


@pytest.mark.parametrize("bench_fixture", ["benchmark"])
def bench_three_way_partition_struct_single_phase(bench_fixture, request, size):
    # warm up run
    three_way_partition_struct_single_phase(size)

    def run():
        three_way_partition_struct_single_phase(size)

    fixture = request.getfixturevalue(bench_fixture)
    fixture(run)
