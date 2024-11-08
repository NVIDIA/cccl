# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes
import numpy
import pytest
import numba
from numba import cuda
import cuda.parallel.experimental as cudax
from numba.extending import register_jitable


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


class ConstantIterator:
    def __init__(self, val): # TODO Showcasing the case of int32, need dtype with at least primitive types, ideally any numba type
        thisty = numba.types.CPointer(numba.types.int32)
        self.val = ctypes.c_int32(val)
        self.ltoirs = [numba.cuda.compile(ConstantIterator.constant_int32_advance, sig=numba.types.void(thisty, numba.types.int32), output='ltoir'),
                       numba.cuda.compile(ConstantIterator.constant_int32_dereference, sig=numba.types.int32(thisty), output='ltoir')]
        self.prefix = 'constant_int32'

    def constant_int32_advance(this, _):
        print(f"\nLOOOK PYTHON constant_int32_advance")
        pass

    def constant_int32_dereference(this):
        print(f"\nLOOOK PYTHON constant_int32_dereference")
        return this[0]

    def host_address(self):
        # TODO should use numba instead for support of user-defined types
        return ctypes.byref(self.val)

    def state_c_void_p(self):
        return ctypes.cast(ctypes.pointer(self.val), ctypes.c_void_p)

    def size(self):
        return ctypes.sizeof(self.val) # TODO should be using numba for user-defined types support

    def alignment(self):
        return ctypes.alignment(self.val) # TODO should be using numba for user-defined types support


class CountingIterator:
    def __init__(self, count): # TODO Showcasing the case of int32, need dtype
        thisty = numba.types.CPointer(numba.types.int32)
        self.count = ctypes.c_int32(count)
        self.ltoirs = [numba.cuda.compile(CountingIterator.count_int32_advance, sig=numba.types.void(thisty, numba.types.int32), output='ltoir'),
                       numba.cuda.compile(CountingIterator.count_int32_dereference, sig=numba.types.int32(thisty), output='ltoir')]
        self.prefix = 'count_int32'

    def count_int32_advance(this, diff):
        print(f"\nLOOOK PYTHON count_int32_advance")
        this[0] += diff

    def count_int32_dereference(this):
        print(f"\nLOOOK PYTHON count_int32_dereference")
        return this[0]

    def host_address(self):
        return ctypes.byref(self.count)

    def state_c_void_p(self):
        return ctypes.cast(ctypes.pointer(self.count), ctypes.c_void_p)

    def size(self):
        return ctypes.sizeof(self.count) # TODO should be using numba for user-defined types support

    def alignment(self):
        return ctypes.alignment(self.count) # TODO should be using numba for user-defined types support


@pytest.mark.parametrize("use_numpy_array", [True, False])
@pytest.mark.parametrize("input_generator", ["constant", "counting"])
def test_device_sum_sentinel_iterator(use_numpy_array, input_generator, num_items=3, start_sum_with=10):
    def add_op(a, b):
        return a + b

    if input_generator == "constant":
        l_input = [42 for distance in range(num_items)]
        sentinel_iterator = ConstantIterator(42)
    elif input_generator == "counting":
        l_input = [start_sum_with + distance for distance in range(num_items)]
        sentinel_iterator = CountingIterator(start_sum_with)
    else:
        raise RuntimeError("Unexpected input_generator")

    expected_result = start_sum_with
    for v in l_input:
        expected_result = add_op(expected_result, v)

    dtype = numpy.int32 # TODO: Replace hard-wired dtype in production code.

    if use_numpy_array:
        h_input = numpy.array(l_input, dtype)
        d_input = cuda.to_device(h_input)
    else:
        d_input = sentinel_iterator

    d_output = cuda.device_array(1, dtype) # to store device sum

    h_init = numpy.array([start_sum_with], dtype)

    reduce_into = cudax.reduce_into(d_in=d_input, d_out=d_output, op=add_op, init=h_init)

    temp_storage_size = reduce_into(None, num_items, d_in=d_input, d_out=d_output, init=h_init)
    d_temp_storage = cuda.device_array(temp_storage_size, dtype=numpy.uint8)

    reduce_into(d_temp_storage, num_items, d_input, d_output, h_init)

    h_output = d_output.copy_to_host()
    assert h_output[0] == expected_result
