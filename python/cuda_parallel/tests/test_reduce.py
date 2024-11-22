# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy
import pytest
import random
import numba.cuda
import numba.types
import cuda.parallel.experimental as cudax
from cuda.parallel.experimental import iterators


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
    d_output = numba.cuda.device_array(1, dtype=dtype)
    reduce_into = cudax.reduce_into(d_output, d_output, op, h_init)

    for num_items_pow2 in type_to_problem_sizes(dtype):
        num_items = 2 ** num_items_pow2
        h_input = random_int(num_items, dtype)
        d_input = numba.cuda.to_device(h_input)
        temp_storage_size = reduce_into(None, None, d_input, d_output, h_init)
        d_temp_storage = numba.cuda.device_array(
            temp_storage_size, dtype=numpy.uint8)
        reduce_into(d_temp_storage, None, d_input, d_output, h_init)
        h_output = d_output.copy_to_host()
        assert h_output[0] == sum(h_input) + init_value


def test_complex_device_reduce():
    def op(a, b):
        return a + b

    h_init = numpy.array([40.0 + 2.0j], dtype=complex)
    d_output = numba.cuda.device_array(1, dtype=complex)
    reduce_into = cudax.reduce_into(d_output, d_output, op, h_init)

    for num_items in [42, 420000]:
        h_input = numpy.random.random(
            num_items) + 1j * numpy.random.random(num_items)
        d_input = numba.cuda.to_device(h_input)
        temp_storage_bytes = reduce_into(None, None, d_input, d_output, h_init)
        d_temp_storage = numba.cuda.device_array(temp_storage_bytes, numpy.uint8)
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
    d_outputs = [numba.cuda.device_array(1, dt) for dt in dtypes]
    d_inputs = [numba.cuda.to_device(h_inp) for h_inp in h_inputs]

    reduce_into = cudax.reduce_into(d_inputs[0], d_outputs[0], min_op, h_inits[0])

    for ix in range(3):
        with pytest.raises(TypeError, match=r"^dtype mismatch: __init__=int32, __call__=int64$"):
          reduce_into(None, None, d_inputs[int(ix == 0)], d_outputs[int(ix == 1)], h_inits[int(ix == 2)])


def mul2(val):
    return 2 * val


@pytest.mark.parametrize("use_numpy_array", [True, False])
@pytest.mark.parametrize("input_generator", ["raw_pointer_int16",
                                             "raw_pointer_uint16",
                                             "raw_pointer_int32",
                                             "raw_pointer_uint32",
                                             "raw_pointer_int64",
                                             "raw_pointer_uint64",
                                             "raw_pointer_float32",
                                             "raw_pointer_float64",
                                             "streamed_input_int16",
                                             "streamed_input_uint16",
                                             "streamed_input_int32",
                                             "streamed_input_uint32",
                                             "streamed_input_int64",
                                             "streamed_input_uint64",
                                             "streamed_input_float32",
                                             "streamed_input_float64",
                                             "constant_int16",
                                             "constant_uint16",
                                             "constant_int32",
                                             "constant_uint32",
                                             "constant_int64",
                                             "constant_uint64",
                                             "constant_float32",
                                             "constant_float64",
                                             "counting_int16",
                                             "counting_uint16",
                                             "counting_int32",
                                             "counting_uint32",
                                             "counting_int64",
                                             "counting_uint64",
                                             "counting_float32",
                                             "counting_float64",
                                             "map_mul2_int32_int32",
                                            ])
def test_device_sum_iterators(use_numpy_array, input_generator, num_items=3, start_sum_with=10):
    def add_op(a, b):
        return a + b

    def dtype_ntype(ix):
        intty = input_generator.split("_")[ix]
        return numpy.dtype(intty), getattr(numba.types, intty)

    if input_generator.startswith("raw_pointer_"):
        rng = random.Random(0)
        l_input = [rng.randrange(100) for _ in range(num_items)]
        dtype, ntype = dtype_ntype(-1)
        raw_pointer_devarr = numba.cuda.to_device(numpy.array(l_input, dtype=dtype))
        i_input = iterators.pointer(raw_pointer_devarr, ntype=ntype)
    elif input_generator.startswith("streamed_input_"):
        rng = random.Random(0)
        l_input = [rng.randrange(100) for _ in range(num_items)]
        dtype, ntype = dtype_ntype(-1)
        streamed_input_devarr = numba.cuda.to_device(numpy.array(l_input, dtype=dtype))
        i_input = iterators.cache(streamed_input_devarr, ntype=ntype, modifier='stream')
    elif input_generator.startswith("constant_"):
        l_input = [42 for distance in range(num_items)]
        dtype, ntype = dtype_ntype(-1)
        i_input = iterators.repeat(42, ntype=ntype)
    elif input_generator.startswith("counting_"):
        l_input = [start_sum_with + distance for distance in range(num_items)]
        dtype, ntype = dtype_ntype(-1)
        i_input = iterators.count(start_sum_with, ntype=ntype)
    elif input_generator.startswith("map_mul2_"):
        l_input = [2 * (start_sum_with + distance) for distance in range(num_items)]
        dtype_op, ntype_op = dtype_ntype(-1)
        dtype_it, ntype_it = dtype_ntype(-2)
        dtype = dtype_op
        i_input = iterators.cu_map(
            mul2,
            iterators.count(start_sum_with, ntype=ntype_it),
            op_return_ntype=ntype_op)
    else:
        raise RuntimeError("Unexpected input_generator")

    expected_result = start_sum_with
    for v in l_input:
        expected_result = add_op(expected_result, v)

    if use_numpy_array:
        h_input = numpy.array(l_input, dtype)
        d_input = numba.cuda.to_device(h_input)
    else:
        d_input = i_input

    d_output = numba.cuda.device_array(1, dtype) # to store device sum

    h_init = numpy.array([start_sum_with], dtype)

    reduce_into = cudax.reduce_into(d_in=d_input, d_out=d_output, op=add_op, init=h_init)

    temp_storage_size = reduce_into(None, num_items, d_in=d_input, d_out=d_output, init=h_init)
    d_temp_storage = numba.cuda.device_array(temp_storage_size, dtype=numpy.uint8)

    reduce_into(d_temp_storage, num_items, d_input, d_output, h_init)

    h_output = d_output.copy_to_host()
    assert h_output[0] == expected_result
