from numba.core import cgutils
from llvmlite import ir
from numba import types
from numba.core.typing import signature
from numba.core.extending import intrinsic, overload
from numba.core.errors import NumbaTypeError

import ctypes
import numba
import numba.cuda
import numpy as np


class RawPointer:
    def __init__(self, ptr): # TODO Showcasing the case of int32, need dtype with at least primitive types, ideally any numba type
        self.val = ctypes.c_void_p(ptr)
        self.ltoirs = [numba.cuda.compile(RawPointer.pointer_advance, sig=numba.types.void(numba.types.CPointer(numba.types.uint64), numba.types.int32), output='ltoir'),
                       numba.cuda.compile(RawPointer.pointer_dereference, sig=numba.types.int32(numba.types.CPointer(numba.types.CPointer(numba.types.int32))), output='ltoir')]
        self.prefix = 'pointer'

    def pointer_advance(this, distance):
        this[0] = this[0] + numba.types.uint64(4 * distance) # TODO Showcasing the case of int32, need dtype with at least primitive types, ideally any numba type

    def pointer_dereference(this):
        return this[0][0]

    def host_address(self):
        return ctypes.byref(self.val)

    def size(self):
        return 8 # TODO should be using numba for user-defined types support

    def alignment(self):
        return 8 # TODO should be using numba for user-defined types support


def pointer(container):
    return RawPointer(container.device_ctypes_pointer.value)


@intrinsic
def ldcs(typingctx, base):
    signature = types.int32(types.CPointer(types.int32))

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
        pass

    def constant_int32_dereference(this):
        return this[0]

    def host_address(self):
        # TODO should use numba instead for support of user-defined types
        return ctypes.byref(self.val)

    def size(self):
        return ctypes.sizeof(self.val) # TODO should be using numba for user-defined types support

    def alignment(self):
        return ctypes.alignment(self.val) # TODO should be using numba for user-defined types support


def repeat(value):
    return ConstantIterator(value)


class CountingIterator:
    def __init__(self, count): # TODO Showcasing the case of int32, need dtype
        thisty = numba.types.CPointer(numba.types.int32)
        self.count = ctypes.c_int32(count)
        self.ltoirs = [numba.cuda.compile(CountingIterator.count_int32_advance, sig=numba.types.void(thisty, numba.types.int32), output='ltoir'),
                       numba.cuda.compile(CountingIterator.count_int32_dereference, sig=numba.types.int32(thisty), output='ltoir')]
        self.prefix = 'count_int32'

    def count_int32_advance(this, diff):
        this[0] += diff

    def count_int32_dereference(this):
        return this[0]

    def host_address(self):
        return ctypes.byref(self.count)

    def size(self):
        return ctypes.sizeof(self.count) # TODO should be using numba for user-defined types support

    def alignment(self):
        return ctypes.alignment(self.count) # TODO should be using numba for user-defined types support


def count(offset):
    return CountingIterator(offset)


def map(it, op):
    def source_advance(it_state_ptr, diff):
        pass

    def make_advance_codegen(name):
        retty = types.void
        statety = types.CPointer(types.int8)
        distty = types.int32

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
        retty = types.int32
        statety = types.CPointer(types.int8)

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
        retty = types.int32
        valty = types.int32

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
            self.ltoirs = it.ltoirs + [numba.cuda.compile(TransformIterator.transform_advance, sig=numba.types.void(numba.types.CPointer(numba.types.char), numba.types.int32), output='ltoir', abi_info={"abi_name": f"{self.prefix}_advance"}),
                                       numba.cuda.compile(TransformIterator.transform_dereference, sig=numba.types.int32(numba.types.CPointer(numba.types.char)), output='ltoir', abi_info={"abi_name": f"{self.prefix}_dereference"}),
                                       numba.cuda.compile(op, sig=numba.types.int32(numba.types.int32), output='ltoir')]

        def transform_advance(it_state_ptr, diff):
            source_advance(it_state_ptr, diff) # just a function call

        def transform_dereference(it_state_ptr):
            return op(source_dereference(it_state_ptr)) # just a function call

        def host_address(self):
            return self.it.host_address() # TODO support stateful operators

        def size(self):
            return self.it.size() # TODO fix for stateful op

        def alignment(self):
            return self.it.alignment() # TODO fix for stateful op


    return TransformIterator(it, op)


def parallel_algorithm(d_in, num_items, d_out):
    ltoirs = [ltoir[0] for ltoir in d_in.ltoirs]

    LTOIRArrayType = ctypes.c_char_p * len(ltoirs)
    LTOIRSizesArrayType = ctypes.c_int * len(ltoirs)
    ltoir_pointers = [ctypes.c_char_p(ltoir) for ltoir in ltoirs]
    ltoir_sizes = [len(ltoir) for ltoir in ltoirs]
    input_ltoirs = LTOIRArrayType(*ltoir_pointers)
    input_ltoir_sizes = LTOIRSizesArrayType(*ltoir_sizes)

    bindings = ctypes.CDLL('build/libkernel.so')
    bindings.host_code.argtypes = [ctypes.c_int,  # size
                                   ctypes.c_int,  # alignment
                                   ctypes.c_void_p, # input_pointer
                                   ctypes.c_char_p, # prefix
                                   ctypes.c_int,    # num_items
                                   ctypes.c_void_p, # output_pointer
                                   ctypes.POINTER(ctypes.c_char_p),
                                   ctypes.POINTER(ctypes.c_int),
                                   ctypes.c_int]

    output_pointer = output_array.device_ctypes_pointer.value
    bindings.host_code(d_in.size(), d_in.alignment(), d_in.host_address(), d_in.prefix.encode('utf-8'), num_items, output_pointer, input_ltoirs, input_ltoir_sizes, len(ltoirs))


# User Code

num_items = 4
output_array = numba.cuda.device_array(num_items, dtype=np.int32)

## Repeat
r = repeat(42)
parallel_algorithm(r, num_items, output_array)
print("expect: 42 42 42 42;  get: ", " ".join([str(x) for x in output_array.copy_to_host()]))

## Count
c = count(42)
parallel_algorithm(c, num_items, output_array)
print("expect: 42 43 44 45;  get: ", " ".join([str(x) for x in output_array.copy_to_host()]))

## Transform
def mult(x):
    return x * 2

mult_42_by_2 = map(r, mult)
parallel_algorithm(mult_42_by_2, num_items, output_array)
print("expect: 84 84 84 84;  get: ", " ".join([str(x) for x in output_array.copy_to_host()]))

def add(x):
    return x + 10

mult_42_by_2_plus10 = map(mult_42_by_2, add)
parallel_algorithm(mult_42_by_2_plus10, num_items, output_array)
print("expect: 94 94 94 94;  get: ", " ".join([str(x) for x in output_array.copy_to_host()]))

mult_count_by_2 = map(c, mult)
parallel_algorithm(mult_count_by_2, num_items, output_array)
print("expect: 84 86 88 90;  get: ", " ".join([str(x) for x in output_array.copy_to_host()]))

mult_count_by_2_and_add_10 = map(mult_count_by_2, add)
parallel_algorithm(mult_count_by_2_and_add_10, num_items, output_array)
print("expect: 94 96 98 100; get:", " ".join([str(x) for x in output_array.copy_to_host()]))

input_array = numba.cuda.to_device(np.array([4, 3, 2, 1], dtype=np.int32))
ptr = pointer(input_array) # TODO this transformation should be hidden on the transform implementation side
parallel_algorithm(ptr, num_items, output_array)
print("expect:  4  3  2 1  ; get:", " ".join([str(x) for x in output_array.copy_to_host()]))

input_array = numba.cuda.to_device(np.array([4, 3, 2, 1], dtype=np.int32))
ptr = pointer(input_array) # TODO this transformation should be hidden on the transform implementation side
tptr = map(ptr, mult)
parallel_algorithm(tptr, num_items, output_array)
print("expect:  8  6  4 2  ; get:", " ".join([str(x) for x in output_array.copy_to_host()]))

streamed_input = cache(input_array, 'stream')
parallel_algorithm(streamed_input, num_items, output_array)
print("expect:  4  3  2 1  ; get:", " ".join([str(x) for x in output_array.copy_to_host()]))
