# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy
import pytest
import numba
from numba import cuda
import cuda.parallel.experimental as cudax


def random_int(shape, dtype):
    return numpy.random.randint(0, 5, size=shape).astype(dtype)


def type_to_problem_sizes(dtype):
    if dtype in [numpy.uint8, numpy.int8]:
        return [2, 4, 5, 6]
    elif dtype in [numpy.uint16, numpy.int16]:
        return [4, 8, 12, 14]
    elif dtype in [numpy.uint32, numpy.int32]:
        return [16, 20, 24, 28]
    elif dtype in [numpy.uint64, numpy.int64]:
        return [16, 20, 24, 28]
    else:
        raise ValueError("Unsupported dtype")


@pytest.mark.parametrize('dtype', [numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64])
def test_device_reduce(dtype):
    def op(a, b):
        return a + b

    init_value = 42
    h_init = numpy.array([init_value], dtype=dtype)
    d_output = cuda.device_array(1, dtype=dtype)
    reduce_into = cudax.reduce_into(d_output, d_output, op, h_init)

    for num_items_pow2 in type_to_problem_sizes(dtype):
        num_items = 2 ** num_items_pow2
        h_input = random_int(num_items, dtype)
        d_input = cuda.to_device(h_input)
        temp_storage_size = reduce_into(None, None, d_input, d_output, h_init)
        d_temp_storage = cuda.device_array(
            temp_storage_size, dtype=numpy.uint8)
        reduce_into(d_temp_storage, None, d_input, d_output, h_init)
        h_output = d_output.copy_to_host()
        assert h_output[0] == sum(h_input) + init_value


def test_complex_device_reduce():
    def op(a, b):
        return a + b

    h_init = numpy.array([40.0 + 2.0j], dtype=complex)
    d_output = cuda.device_array(1, dtype=complex)
    reduce_into = cudax.reduce_into(d_output, d_output, op, h_init)

    for num_items in [42, 420000]:
        h_input = numpy.random.random(
            num_items) + 1j * numpy.random.random(num_items)
        d_input = cuda.to_device(h_input)
        temp_storage_bytes = reduce_into(None, None, d_input, d_output, h_init)
        d_temp_storage = cuda.device_array(temp_storage_bytes, numpy.uint8)
        reduce_into(d_temp_storage, None, d_input, d_output, h_init)

        result = d_output.copy_to_host()[0]
        expected = numpy.sum(h_input, initial=h_init[0])
        assert result == pytest.approx(expected)


def test_device_reduce_dtype_mismatch():
    def min_op(a, b):
        return a if a < b else b

    dtypes = [numpy.int32, numpy.int64]
    h_inits = [numpy.array([], dt) for dt in dtypes]
    h_inputs = [numpy.array([], dt) for dt in dtypes]
    d_outputs = [cuda.device_array(1, dt) for dt in dtypes]
    d_inputs = [cuda.to_device(h_inp) for h_inp in h_inputs]

    reduce_into = cudax.reduce_into(d_inputs[0], d_outputs[0], min_op, h_inits[0])

    for ix in range(3):
        with pytest.raises(TypeError, match=r"^dtype mismatch: __init__=int32, __call__=int64$"):
          reduce_into(None, None, d_inputs[int(ix == 0)], d_outputs[int(ix == 1)], h_inits[int(ix == 2)])


def cu_repeat(value):
    def input_unary_op(unused_distance):
        data = (value + 2, value + 1)
        return data[0] - data[1]
    return input_unary_op


def cu_count(start):
    def input_unary_op(distance):
        return distance
    return input_unary_op


def cu_arbitrary(permutation):
    def input_unary_op(distance):
        return permutation[distance % len(permutation)]
    return input_unary_op


@pytest.mark.parametrize("use_numpy_array", [True, False])
@pytest.mark.parametrize("input_generator", ["constant", "counting", "arbitrary"])
def test_device_sum_input_unary_op(use_numpy_array, input_generator, num_items=19, start_sum_with=1000):
    def add_op(a, b):
        return a + b

    if input_generator == "constant":
        input_unary_op = cu_repeat(3)
    elif input_generator == "counting":
        input_unary_op = cu_count(4)
    elif input_generator == "arbitrary":
        input_unary_op = cu_arbitrary((4, 2, 0, 3, 1))
    else:
        raise RuntimeError("Unexpected input_generator")

    l_input = [input_unary_op(distance) for distance in range(num_items)]
    expected_result = start_sum_with
    for v in l_input:
        expected_result = add_op(expected_result, v)

    dtype = numpy.int32 # TODO: Replace hard-wired dtype in production code.

    if use_numpy_array:
        h_input = numpy.array(l_input, dtype)
        d_input = cuda.to_device(h_input)
    else:
        d_input = input_unary_op

    d_output = cuda.device_array(1, dtype) # to store device sum

    h_init = numpy.array([start_sum_with], dtype)

    reduce_into = cudax.reduce_into(d_in=d_input, d_out=d_output, op=add_op, init=h_init)

    temp_storage_size = reduce_into(None, num_items, d_in=d_input, d_out=d_output, init=h_init)
    d_temp_storage = cuda.device_array(temp_storage_size, dtype=numpy.uint8)

    reduce_into(d_temp_storage, num_items, d_input, d_output, h_init)

    h_output = d_output.copy_to_host()
    assert h_output[0] == expected_result
