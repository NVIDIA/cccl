import ctypes
import operator
import collections
from functools import cached_property, lru_cache

import numpy as np
from numba.core import cgutils
from llvmlite import ir
from numba.core.typing import signature
from numba.core.extending import intrinsic, overload
import numba
import numba.cuda
import numba.types


_DEVICE_POINTER_SIZE = 8
_DEVICE_POINTER_BITWIDTH = _DEVICE_POINTER_SIZE * 8
_DISTANCE_NUMBA_TYPE = numba.types.uint64
_DISTANCE_IR_TYPE = ir.IntType(64)
_CHAR_PTR_NUMBA_TYPE = numba.types.CPointer(numba.types.int8)
_CHAR_PTR_IR_TYPE = ir.PointerType(ir.IntType(8))


def numba_type_from_any(value_type):
    return getattr(numba.types, str(value_type))


def _sizeof_numba_type(ntype):
    mapping = {
        numba.types.int8: 1,
        numba.types.int16: 2,
        numba.types.int32: 4,
        numba.types.int64: 8,
        numba.types.uint8: 1,
        numba.types.uint16: 2,
        numba.types.uint32: 4,
        numba.types.uint64: 8,
        numba.types.float32: 4,
        numba.types.float64: 8,
    }
    return mapping[ntype]


@lru_cache
def _ctypes_type_given_numba_type(ntype):
    mapping = {
        numba.types.int8: ctypes.c_int8,
        numba.types.int16: ctypes.c_int16,
        numba.types.int32: ctypes.c_int32,
        numba.types.int64: ctypes.c_int64,
        numba.types.uint8: ctypes.c_uint8,
        numba.types.uint16: ctypes.c_uint16,
        numba.types.uint32: ctypes.c_uint32,
        numba.types.uint64: ctypes.c_uint64,
        numba.types.float32: ctypes.c_float,
        numba.types.float64: ctypes.c_double,
    }
    return mapping[ntype]


@lru_cache
def cached_compile(*args, **kwargs):
    result = numba.cuda.compile(*args, **kwargs)
    return result


class IteratorBase:
    def __init__(self, cvalue, value_type, numba_type):
        self.cvalue = cvalue
        self.value_type = value_type
        self.numba_type = numba_type

    @cached_property
    def ltoirs(self):
        advance_ltoir, _ = cached_compile(
            self.__class__.advance,
            (
                self.numba_type,
                numba.from_dtype(np.dtype("uint64"))
            ),
            output="ltoir"
        )

        deref_ltoir, _ = cached_compile(
            self.__class__.dereference,
            (
                self.numba_type,
            ),
            output="ltoir"
        )

        return advance_ltoir, deref_ltoir

    @property
    def state(self):
        return ctypes.cast(ctypes.pointer(self.cvalue), ctypes.c_void_p)


def sizeof_pointee(context, ptr):
    size = context.get_abi_sizeof(ptr.type.pointee)
    return ir.Constant(ir.IntType(_DEVICE_POINTER_BITWIDTH), size)


@intrinsic
def pointer_add_intrinsic(context, ptr, offset):
    def codegen(context, builder, sig, args):
        ptr, index = args
        base = builder.ptrtoint(ptr, ir.IntType(_DEVICE_POINTER_BITWIDTH))
        offset = builder.mul(index, sizeof_pointee(context, ptr))
        result = builder.add(base, offset)
        return builder.inttoptr(result, ptr.type)

    return ptr(ptr, offset), codegen


@overload(operator.add)
def pointer_add(ptr, offset):
    if not isinstance(ptr, numba.types.CPointer) or not isinstance(
        offset, numba.types.Integer
    ):
        return

    def impl(ptr, offset):
        return pointer_add_intrinsic(ptr, offset)

    return impl


class RawPointer(IteratorBase):
    def __init__(self, ptr, ntype):
        value_type = ntype
        cvalue = ctypes.c_void_p(ptr)
        numba_type = numba.types.CPointer(numba.types.CPointer(value_type))
        super().__init__(cvalue=cvalue, value_type=value_type, numba_type=numba_type)

    @staticmethod
    def advance(it, distance):
        it[0] = it[0] + distance

    @staticmethod
    def dereference(it):
        return it[0][0]


def pointer(container, ntype):
    return RawPointer(container.__cuda_array_interface__["data"][0], ntype)


def _ir_type_given_numba_type(ntype):
    bw = ntype.bitwidth
    irt = None
    if isinstance(ntype, numba.core.types.scalars.Integer):
        irt = ir.IntType(bw)
    elif isinstance(ntype, numba.core.types.scalars.Float):
        if bw == 32:
            irt = ir.FloatType()
        elif bw == 64:
            irt = ir.DoubleType()
    return irt


@intrinsic
def load_cs(typingctx, base):
    # Corresponding to `LOAD_CS` here:
    # https://nvidia.github.io/cccl/cub/api/classcub_1_1CacheModifiedInputIterator.html
    def codegen(context, builder, sig, args):
        rt = _ir_type_given_numba_type(sig.return_type)
        if rt is None:
            raise RuntimeError(f"Unsupported: {type(sig.return_type)=}")
        ftype = ir.FunctionType(rt, [rt.as_pointer()])
        bw = sig.return_type.bitwidth
        asm_txt = f"ld.global.cs.b{bw} $0, [$1];"
        if bw < 64:
            constraint = "=r, l"
        else:
            constraint = "=l, l"
        asm_ir = ir.InlineAsm(ftype, asm_txt, constraint)
        return builder.call(asm_ir, args)

    return base.dtype(base), codegen


class CacheModifiedPointer(IteratorBase):
    def __init__(self, ptr, ntype):
        cvalue = ctypes.c_void_p(ptr)
        value_type = ntype
        numba_type = numba.types.CPointer(numba.types.CPointer(value_type))
        super().__init__(cvalue=cvalue, value_type=value_type, numba_type=numba_type)

    @staticmethod
    def advance(it, distance):
        it[0] = it[0] + distance

    @staticmethod
    def dereference(it):
        return load_cs(it[0])


class ConstantIterator(IteratorBase):
    def __init__(self, value):
        value_type = numba.from_dtype(value.dtype)
        cvalue = _ctypes_type_given_numba_type(value_type)(value)
        numba_type = numba.types.CPointer(value_type)
        super().__init__(cvalue=cvalue, value_type=value_type, numba_type=numba_type)

    @staticmethod
    def advance(it, n):
        pass

    @staticmethod
    def dereference(it):
        return it[0]


class CountingIterator(IteratorBase):
    def __init__(self, value):
        value_type = numba.from_dtype(value.dtype)
        cvalue = _ctypes_type_given_numba_type(value_type)(value)
        numba_type = numba.types.CPointer(value_type)
        super().__init__(cvalue=cvalue, value_type=value_type, numba_type=numba_type)

    @staticmethod
    def advance(it, n):
        it[0] += n

    @staticmethod
    def dereference(it):
        return it[0]


def make_transform_iterator(it, op):
    if hasattr(it, "__cuda_array_interface__"):
        it = pointer(it, numba.from_dtype(it.dtype))

    it_advance = numba.cuda.jit(type(it).advance, device=True)
    it_dereference = numba.cuda.jit(type(it).dereference, device=True)
    op = numba.cuda.jit(op, device=True)

    class TransformIterator(IteratorBase):
        def __init__(self, it, op):
            self._it = it
            cvalue = it.cvalue
            numba_type = it.numba_type
            _, op_retty = cached_compile(
                op,
                (
                    self._it.value_type,
                ),
                output="ltoir"
            )
            value_type = op_retty
            super().__init__(cvalue=cvalue, value_type=value_type, numba_type=numba_type)

        @staticmethod
        def advance(it, n):
            return it_advance(it, n)

        @staticmethod
        def dereference(it):
            return op(it_dereference(it))

    return TransformIterator(it, op)
