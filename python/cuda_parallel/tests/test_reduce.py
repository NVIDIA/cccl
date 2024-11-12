# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes
import numpy
import pytest
import random
import numba
from numba import cuda
import cuda.parallel.experimental as cudax
from numba.extending import register_jitable
from llvmlite import ir
from numba.core import cgutils
from numba.core.typing import signature
from numba.core.extending import intrinsic, overload
from numba.core.errors import NumbaTypeError


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


class RawPointer:
    def __init__(self, ptr, dtype):
        self.val = ctypes.c_void_p(ptr)
        self.ltoirs = [numba.cuda.compile(RawPointer.pointer_advance, sig=numba.types.void(numba.types.CPointer(numba.types.uint64), dtype), output='ltoir'),
                       numba.cuda.compile(RawPointer.pointer_dereference, sig=dtype(numba.types.CPointer(numba.types.CPointer(dtype))), output='ltoir')]
        self.prefix = 'pointer'

    def pointer_advance(this, distance):
        this[0] = this[0] + numba.types.uint64(4 * distance) # TODO Showcasing the case of int32, need dtype with at least primitive types, ideally any numba type

    def pointer_dereference(this):
        return this[0][0]

    def host_address(self):
        return ctypes.byref(self.val)

    def state_c_void_p(self):
        return ctypes.cast(ctypes.pointer(self.val), ctypes.c_void_p)

    def size(self):
        return 8 # TODO should be using numba for user-defined types support

    def alignment(self):
        return 8 # TODO should be using numba for user-defined types support


def pointer(container, dtype):
    return RawPointer(container.device_ctypes_pointer.value, dtype)


@intrinsic
def ldcs(typingctx, base):
    signature = numba.types.int32(numba.types.CPointer(numba.types.int32))

    def codegen(context, builder, sig, args):
        int32 = ir.IntType(32)
        int32_ptr = int32.as_pointer()
        ldcs_type = ir.FunctionType(int32, [int32_ptr])
        ldcs = ir.InlineAsm(ldcs_type, "ld.global.cs.b32 $0, [$1];", "=r, l")
        return builder.call(ldcs, args)

    return signature, codegen


class CacheModifiedPointer:
    def __init__(self, ptr): # TODO Showcasing the case of int32, need dtype with at least primitive types, ideally any numba type
        self.val = ctypes.c_void_p(ptr)
        self.ltoirs = [numba.cuda.compile(CacheModifiedPointer.cache_advance, sig=numba.types.void(numba.types.CPointer(numba.types.uint64), numba.types.int32), output='ltoir'),
                       numba.cuda.compile(CacheModifiedPointer.cache_dereference, sig=numba.types.int32(numba.types.CPointer(numba.types.CPointer(numba.types.int32))), output='ltoir')]
        self.prefix = 'cache'

    def cache_advance(this, distance):
        this[0] = this[0] + numba.types.uint64(4 * distance) # TODO Showcasing the case of int32, need dtype with at least primitive types, ideally any numba type

    def cache_dereference(this):
        return ldcs(this[0])

    def host_address(self):
        return ctypes.byref(self.val)

    def state_c_void_p(self):
        return ctypes.cast(ctypes.pointer(self.val), ctypes.c_void_p)

    def size(self):
        return 8 # TODO should be using numba for user-defined types support

    def alignment(self):
        return 8 # TODO should be using numba for user-defined types support


def cache(container, modifier='stream'):
    if modifier != 'stream':
        raise NotImplementedError("Only stream modifier is supported")
    return CacheModifiedPointer(container.device_ctypes_pointer.value)


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


def cu_map(op, it):
    def source_advance(it_state_ptr, diff):
        pass

    def make_advance_codegen(name):
        retty = numba.types.void
        statety = numba.types.CPointer(numba.types.int8)
        distty = numba.types.int32

        def codegen(context, builder, sig, args):
            state_ptr, dist = args
            fnty = ir.FunctionType(ir.VoidType(), (ir.PointerType(ir.IntType(8)), ir.IntType(32)))
            fn = cgutils.get_or_insert_function(builder.module, fnty, name)
            builder.call(fn, (state_ptr, dist))

        return signature(retty, statety, distty), codegen


    def advance_codegen(func_to_overload, name):
        @intrinsic
        def intrinsic_impl(typingctx, it_state_ptr, diff):
            return make_advance_codegen(name)

        @overload(func_to_overload, target='cuda')
        def impl(it_state_ptr, diff):
            def impl(it_state_ptr, diff):
                return intrinsic_impl(it_state_ptr, diff)
            return impl

    def source_dereference(it_state_ptr):
        pass

    def make_dereference_codegen(name):
        retty = numba.types.int32
        statety = numba.types.CPointer(numba.types.int8)

        def codegen(context, builder, sig, args):
            state_ptr, = args
            fnty = ir.FunctionType(ir.IntType(32), (ir.PointerType(ir.IntType(8)),))
            fn = cgutils.get_or_insert_function(builder.module, fnty, name)
            return builder.call(fn, (state_ptr,))

        return signature(retty, statety), codegen


    def dereference_codegen(func_to_overload, name):
        @intrinsic
        def intrinsic_impl(typingctx, it_state_ptr):
            return make_dereference_codegen(name)

        @overload(func_to_overload, target='cuda')
        def impl(it_state_ptr):
            def impl(it_state_ptr):
                return intrinsic_impl(it_state_ptr)
            return impl

    def make_op_codegen(name):
        retty = numba.types.int32
        valty = numba.types.int32

        def codegen(context, builder, sig, args):
            val, = args
            fnty = ir.FunctionType(ir.IntType(32), (ir.IntType(32),))
            fn = cgutils.get_or_insert_function(builder.module, fnty, name)
            return builder.call(fn, (val,))

        return signature(retty, valty), codegen


    def op_codegen(func_to_overload, name):
        @intrinsic
        def intrinsic_impl(typingctx, val):
            return make_op_codegen(name)

        @overload(func_to_overload, target='cuda')
        def impl(val):
            def impl(val):
                return intrinsic_impl(val)
            return impl

    advance_codegen(source_advance, f"{it.prefix}_advance")
    dereference_codegen(source_dereference, f"{it.prefix}_dereference")
    op_codegen(op, op.__name__)


    class TransformIterator:
        def __init__(self, it, op):
            self.it = it # TODO support row pointers
            self.op = op
            self.prefix = f'transform_{it.prefix}_{op.__name__}'
            print(f"\nLOOOK PYTHON TransformIterator {self.prefix=}", flush=True)
            self.ltoirs = it.ltoirs + [numba.cuda.compile(TransformIterator.transform_advance, sig=numba.types.void(numba.types.CPointer(numba.types.char), numba.types.int32), output='ltoir', abi_info={"abi_name": f"{self.prefix}_advance"}),
                                       numba.cuda.compile(TransformIterator.transform_dereference, sig=numba.types.int32(numba.types.CPointer(numba.types.char)), output='ltoir', abi_info={"abi_name": f"{self.prefix}_dereference"}),
                                       numba.cuda.compile(op, sig=numba.types.int32(numba.types.int32), output='ltoir')]

        def transform_advance(it_state_ptr, diff):
            source_advance(it_state_ptr, diff) # just a function call

        def transform_dereference(it_state_ptr):
            return op(source_dereference(it_state_ptr)) # just a function call

        def host_address(self):
            return self.it.host_address() # TODO support stateful operators

        def state_c_void_p(self):
            return self.it.state_c_void_p()

        def size(self):
            return self.it.size() # TODO fix for stateful op

        def alignment(self):
            return self.it.alignment() # TODO fix for stateful op


    return TransformIterator(it, op)


def mul2(val):
    return 2 * val


@pytest.mark.parametrize("use_numpy_array", [True, False])
@pytest.mark.parametrize("input_generator", ["constant", "counting", "map_mul2",
                                             "raw_pointer", "streamed_input"])
def test_device_sum_sentinel_iterator(use_numpy_array, input_generator, num_items=3, start_sum_with=10):
    def add_op(a, b):
        return a + b

    if input_generator == "constant":
        l_input = [42 for distance in range(num_items)]
        sentinel_iterator = ConstantIterator(42)
    elif input_generator == "counting":
        l_input = [start_sum_with + distance for distance in range(num_items)]
        sentinel_iterator = CountingIterator(start_sum_with)
    elif input_generator == "map_mul2":
        l_input = [2 * (start_sum_with + distance) for distance in range(num_items)]
        sentinel_iterator = cu_map(mul2, CountingIterator(start_sum_with))
    elif input_generator == "raw_pointer":
        rng = random.Random(0)
        l_input = [rng.randrange(100) for _ in range(num_items)]
        raw_pointer_devarr = numba.cuda.to_device(numpy.array(l_input, dtype=numpy.int32))
        sentinel_iterator = pointer(raw_pointer_devarr, numba.types.int32)
    elif input_generator == "streamed_input":
        rng = random.Random(0)
        l_input = [rng.randrange(100) for _ in range(num_items)]
        streamed_input_devarr = numba.cuda.to_device(numpy.array(l_input, dtype=numpy.int32))
        sentinel_iterator = cache(streamed_input_devarr, 'stream')
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
