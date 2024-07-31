# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pynvjitlink import patch
import cuda.cooperative.experimental as cudax
from numba.core.extending import (lower_builtin, make_attribute_wrapper,
                                  models, register_model, type_callable,
                                  typeof_impl)
from numba.core import cgutils
import numpy as np
from helpers import random_int, NUMBA_TYPES_TO_NP
import pytest
from numba import cuda, types
import numba
numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


patch.patch_numba_linker(lto=True)


class Complex:
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    def construct(this):
        default_value = numba.int32(0)
        this[0] = Complex(default_value, default_value)

    def assign(this, that):
        this[0] = Complex(that[0].real, that[0].imag)


class ComplexType(types.Type):
    def __init__(self):
        super().__init__(name='Complex')


complex_type = ComplexType()


@typeof_impl.register(Complex)
def typeof_complex(val, c):
    return complex_type


@type_callable(Complex)
def type__complex(context):
    def typer(real, imag):
        if isinstance(real, types.Integer) and isinstance(imag, types.Integer):
            return complex_type
    return typer


@register_model(ComplexType)
class ComplexModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [('real', types.int32), ('imag', types.int32)]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(ComplexType, 'real', 'real')
make_attribute_wrapper(ComplexType, 'imag', 'imag')


@lower_builtin(Complex, types.Integer, types.Integer)
def impl_complex(context, builder, sig, args):
    typ = sig.return_type
    real, imag = args
    state = cgutils.create_struct_proxy(typ)(context, builder)
    state.real = real
    state.imag = imag
    return state._getvalue()


@pytest.mark.parametrize('threads_in_block', [32, 64, 128, 256, 512, 1024])
def test_block_reduction_of_user_defined_type(threads_in_block):
    def op(result_ptr, lhs_ptr, rhs_ptr):
        real_value = numba.int32(lhs_ptr[0].real + rhs_ptr[0].real)
        imag_value = numba.int32(lhs_ptr[0].imag + rhs_ptr[0].imag)
        result_ptr[0] = Complex(real_value, imag_value)

    block_reduce = cudax.block.reduce(dtype=complex_type,
                                      binary_op=op,
                                      threads_in_block=threads_in_block,
                                      methods={
                                          'construct': Complex.construct,
                                          'assign': Complex.assign,
                                      })
    temp_storage_bytes = block_reduce.temp_storage_bytes

    @cuda.jit(link=block_reduce.files)
    def kernel(input, output):
        temp_storage = cuda.shared.array(
            shape=temp_storage_bytes, dtype='uint8')
        block_output = block_reduce(temp_storage, Complex(input[cuda.threadIdx.x],
                                                          input[threads_in_block + cuda.threadIdx.x]))

        if cuda.threadIdx.x == 0:
            output[0] = block_output.real
            output[1] = block_output.imag

    h_input = random_int(2 * threads_in_block, 'int32')
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(2, dtype='int32')
    kernel[1, threads_in_block](d_input, d_output)
    cuda.synchronize()
    h_output = d_output.copy_to_host()
    h_expected = np.sum(h_input[:threads_in_block]), np.sum(
        h_input[threads_in_block:])

    assert h_output[0] == h_expected[0]
    assert h_output[1] == h_expected[1]

    sig = (numba.int32[::1], numba.int32[::1])
    sass = kernel.inspect_sass(sig)

    assert 'LDL' not in sass
    assert 'STL' not in sass


@pytest.mark.parametrize('T', [types.uint32, types.uint64])
@pytest.mark.parametrize('threads_in_block', [32, 64, 128, 256, 512, 1024])
def test_block_reduction_of_integral_type(T, threads_in_block):
    def op(a, b):
        return a if a < b else b

    block_reduce = cudax.block.reduce(dtype=T,
                                      binary_op=op,
                                      threads_in_block=threads_in_block)
    temp_storage_bytes = block_reduce.temp_storage_bytes

    @cuda.jit(link=block_reduce.files)
    def kernel(input, output):
        temp_storage = cuda.shared.array(
            shape=temp_storage_bytes, dtype='uint8')
        block_output = block_reduce(temp_storage, input[cuda.threadIdx.x])

        if cuda.threadIdx.x == 0:
            output[0] = block_output

    dtype = NUMBA_TYPES_TO_NP[T]
    h_input = random_int(threads_in_block, dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=dtype)
    kernel[1, threads_in_block](d_input, d_output)
    cuda.synchronize()
    h_output = d_output.copy_to_host()
    h_expected = np.min(h_input)

    assert h_output[0] == h_expected

    sig = (T[::1], T[::1])
    sass = kernel.inspect_sass(sig)

    assert 'LDL' not in sass
    assert 'STL' not in sass


@pytest.mark.parametrize('T', [types.uint32, types.uint64])
@pytest.mark.parametrize('threads_in_block', [32, 64, 128, 256, 512, 1024])
def test_block_sum(T, threads_in_block):
    block_reduce = cudax.block.sum(
        dtype=T, threads_in_block=threads_in_block)
    temp_storage_bytes = block_reduce.temp_storage_bytes

    @cuda.jit(link=block_reduce.files)
    def kernel(input, output):
        temp_storage = cuda.shared.array(shape=temp_storage_bytes, dtype='uint8')
        block_output = block_reduce(temp_storage, input[cuda.threadIdx.x])

        if cuda.threadIdx.x == 0:
            output[0] = block_output

    dtype = NUMBA_TYPES_TO_NP[T]
    h_input = random_int(threads_in_block, dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=dtype)
    kernel[1, threads_in_block](d_input, d_output)
    cuda.synchronize()
    h_output = d_output.copy_to_host()
    h_expected = np.sum(h_input)

    assert h_output[0] == h_expected

    sig = (T[::1], T[::1])
    sass = kernel.inspect_sass(sig)

    assert 'LDL' not in sass
    assert 'STL' not in sass


@pytest.mark.parametrize('T', [types.uint32, types.uint64])
@pytest.mark.parametrize('threads_in_block', [32, 64, 128, 256, 512, 1024])
def test_block_valid_sum(T, threads_in_block):
    block_reduce = cudax.block.sum(dtype=T, threads_in_block=threads_in_block)
    temp_storage_bytes = block_reduce.temp_storage_bytes

    @cuda.jit(link=block_reduce.files)
    def kernel(input, output):
        temp_storage = cuda.shared.array(shape=temp_storage_bytes, dtype='uint8')
        block_output = block_reduce(temp_storage, input[cuda.threadIdx.x], numba.int32(threads_in_block / 2))

        if cuda.threadIdx.x == 0:
            output[0] = block_output

    dtype = NUMBA_TYPES_TO_NP[T]
    h_input = random_int(threads_in_block, dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=dtype)
    kernel[1, threads_in_block](d_input, d_output)
    cuda.synchronize()
    h_output = d_output.copy_to_host()
    h_expected = np.sum(h_input[:threads_in_block // 2])

    assert h_output[0] == h_expected

    sig = (T[::1], T[::1])
    sass = kernel.inspect_sass(sig)

    assert 'LDL' not in sass
    assert 'STL' not in sass
