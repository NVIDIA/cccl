import ctypes
import operator
import collections

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


CompileResult = collections.namedtuple("CompileResult", ["ltoir", "result_type"])


def _ncc(abi_name, pyfunc, sig):
    return CompileResult(
        *numba.cuda.compile(
            pyfunc=pyfunc, sig=sig, abi_info={"abi_name": abi_name}, output="ltoir"
        )
    )


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


class RawPointer:
    def __init__(self, ptr, ntype):
        self.val = ctypes.c_void_p(ptr)
        data_as_ntype_pp = numba.types.CPointer(numba.types.CPointer(ntype))
        self.ntype = ntype
        self.prefix = "pointer_" + ntype.name
        self.ltoirs = [
            _ncc(
                f"{self.prefix}_advance",
                RawPointer.pointer_advance,
                numba.types.void(data_as_ntype_pp, _DISTANCE_NUMBA_TYPE),
            ).ltoir,
            _ncc(
                f"{self.prefix}_dereference",
                RawPointer.pointer_dereference,
                (data_as_ntype_pp,),
            ).ltoir,
        ]

    # Exclusively for numba.cuda.compile (this is not an actual method).
    def pointer_advance(this, distance):
        this[0] = this[0] + distance

    # Exclusively for numba.cuda.compile (this is not an actual method).
    def pointer_dereference(this):
        return this[0][0]

    @property
    def state_c_void_p(self):
        return ctypes.cast(ctypes.pointer(self.val), ctypes.c_void_p)

    @property
    def size(self):
        return _DEVICE_POINTER_SIZE

    @property
    def alignment(self):
        return _DEVICE_POINTER_SIZE


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


class CacheModifiedPointer:
    def __init__(self, ptr, ntype):
        self.val = ctypes.c_void_p(ptr)
        self.ntype = ntype
        data_as_ntype_pp = numba.types.CPointer(numba.types.CPointer(ntype))
        self.prefix = "cache" + ntype.name
        self.ltoirs = [
            _ncc(
                f"{self.prefix}_advance",
                CacheModifiedPointer.cache_advance,
                numba.types.void(data_as_ntype_pp, _DISTANCE_NUMBA_TYPE),
            ).ltoir,
            _ncc(
                f"{self.prefix}_dereference",
                CacheModifiedPointer.cache_dereference,
                (data_as_ntype_pp,),
            ).ltoir,
        ]

    # Exclusively for numba.cuda.compile (this is not an actual method).
    def cache_advance(this, distance):
        this[0] = this[0] + distance

    # Exclusively for numba.cuda.compile (this is not an actual method).
    def cache_dereference(this):
        return load_cs(this[0])

    @property
    def state_c_void_p(self):
        return ctypes.cast(ctypes.pointer(self.val), ctypes.c_void_p)

    @property
    def size(self):
        return _DEVICE_POINTER_SIZE

    @property
    def alignment(self):
        return _DEVICE_POINTER_SIZE


class ConstantIterator:
    def __init__(self, val):
        ntype = numba.from_dtype(val.dtype)
        thisty = numba.types.CPointer(ntype)
        self.val = _ctypes_type_given_numba_type(ntype)(val)
        self.ntype = ntype
        self.prefix = "constant_" + ntype.name
        self.ltoirs = [
            _ncc(
                f"{self.prefix}_advance",
                ConstantIterator.constant_advance,
                numba.types.void(thisty, _DISTANCE_NUMBA_TYPE),
            ).ltoir,
            _ncc(
                f"{self.prefix}_dereference",
                ConstantIterator.constant_dereference,
                (thisty,),
            ).ltoir,
        ]

    # Exclusively for numba.cuda.compile (this is not an actual method).
    def constant_advance(this, _):
        pass

    # Exclusively for numba.cuda.compile (this is not an actual method).
    def constant_dereference(this):
        return this[0]

    @property
    def state_c_void_p(self):
        return ctypes.cast(ctypes.pointer(self.val), ctypes.c_void_p)

    @property
    def size(self):
        return self.ntype.bitwidth // 8

    @property
    def alignment(self):
        return self.size


class CountingIterator:
    def __init__(self, count):
        ntype = numba.from_dtype(count.dtype)
        thisty = numba.types.CPointer(ntype)
        self.count = _ctypes_type_given_numba_type(ntype)(count)
        self.ntype = ntype
        self.prefix = "count_" + ntype.name
        self.ltoirs = [
            _ncc(
                f"{self.prefix}_advance",
                CountingIterator.count_advance,
                numba.types.void(thisty, _DISTANCE_NUMBA_TYPE),
            ).ltoir,
            _ncc(
                f"{self.prefix}_dereference",
                CountingIterator.count_dereference,
                (thisty,),
            ).ltoir,
        ]

    # Exclusively for numba.cuda.compile (this is not an actual method).
    def count_advance(this, diff):
        this[0] += diff

    # Exclusively for numba.cuda.compile (this is not an actual method).
    def count_dereference(this):
        return this[0]

    @property
    def state_c_void_p(self):
        return ctypes.cast(ctypes.pointer(self.count), ctypes.c_void_p)

    @property
    def size(self):
        return self.ntype.bitwidth // 8

    @property
    def alignment(self):
        return self.size


class TransformIteratorImpl:
    def __init__(
        self,
        it,
        op_name,
        op_return_ntype,
        ltoirs,
    ):
        self.it = it
        self.ntype = op_return_ntype
        self.prefix = f"transform_{it.prefix}_{op_name}"
        self.ltoirs = ltoirs

    @property
    def state_c_void_p(self):
        return self.it.state_c_void_p

    @property
    def size(self):
        return self.it.size  # TODO fix for stateful op

    @property
    def alignment(self):
        return self.it.alignment  # TODO fix for stateful op


def TransformIterator(op, it):
    # TODO(rwgk): Resolve issue #3064

    if hasattr(it, "dtype"):
        assert not hasattr(it, "ntype")
        it = pointer(it, getattr(numba.types, str(it.dtype)))
    it_ntype_ir = _ir_type_given_numba_type(it.ntype)
    if it_ntype_ir is None:
        raise RuntimeError(f"Unsupported: {type(it.ntype)=}")

    op_abi_name = f"{op.__name__}_{it.ntype.name}"
    op_ltoir, op_return_ntype = _ncc(op_abi_name, op, (it.ntype,))
    op_return_ntype_ir = _ir_type_given_numba_type(op_return_ntype)

    def source_advance(it_state_ptr, diff):
        pass

    def make_advance_codegen(name):
        def codegen(context, builder, sig, args):
            state_ptr, dist = args
            fnty = ir.FunctionType(
                ir.VoidType(), (_CHAR_PTR_IR_TYPE, _DISTANCE_IR_TYPE)
            )
            fn = cgutils.get_or_insert_function(builder.module, fnty, name)
            builder.call(fn, (state_ptr, dist))

        return signature(
            numba.types.void, _CHAR_PTR_NUMBA_TYPE, _DISTANCE_NUMBA_TYPE
        ), codegen

    def advance_codegen(func_to_overload, name):
        @intrinsic
        def intrinsic_impl(typingctx, it_state_ptr, diff):
            return make_advance_codegen(name)

        @overload(func_to_overload, target="cuda")
        def impl(it_state_ptr, diff):
            def impl(it_state_ptr, diff):
                return intrinsic_impl(it_state_ptr, diff)

            return impl

    def source_dereference(it_state_ptr):
        pass

    def make_dereference_codegen(name):
        def codegen(context, builder, sig, args):
            (state_ptr,) = args
            fnty = ir.FunctionType(it_ntype_ir, (_CHAR_PTR_IR_TYPE,))
            fn = cgutils.get_or_insert_function(builder.module, fnty, name)
            return builder.call(fn, (state_ptr,))

        return signature(it.ntype, _CHAR_PTR_NUMBA_TYPE), codegen

    def dereference_codegen(func_to_overload, name):
        @intrinsic
        def intrinsic_impl(typingctx, it_state_ptr):
            return make_dereference_codegen(name)

        @overload(func_to_overload, target="cuda")
        def impl(it_state_ptr):
            def impl(it_state_ptr):
                return intrinsic_impl(it_state_ptr)

            return impl

    def make_op_codegen(name):
        def codegen(context, builder, sig, args):
            (val,) = args
            fnty = ir.FunctionType(op_return_ntype_ir, (it_ntype_ir,))
            fn = cgutils.get_or_insert_function(builder.module, fnty, name)
            return builder.call(fn, (val,))

        return signature(op_return_ntype, it.ntype), codegen

    def op_codegen(func_to_overload, name):
        @intrinsic
        def intrinsic_impl(typingctx, val):
            return make_op_codegen(name)

        @overload(func_to_overload, target="cuda")
        def impl(val):
            def impl(val):
                return intrinsic_impl(val)

            return impl

    advance_codegen(source_advance, f"{it.prefix}_advance")
    dereference_codegen(source_dereference, f"{it.prefix}_dereference")

    def op_caller(value):
        return op(value)

    op_codegen(op_caller, op_abi_name)

    def transform_advance(it_state_ptr, diff):
        source_advance(it_state_ptr, diff)  # just a function call

    def transform_dereference(it_state_ptr):
        # ATTENTION: op_caller here (see issue #3064)
        return op_caller(source_dereference(it_state_ptr))

    prefix = f"transform_{it.prefix}_{op.__name__}"
    ltoirs = it.ltoirs + [
        _ncc(
            f"{prefix}_advance",
            transform_advance,
            numba.types.void(
                numba.types.CPointer(numba.types.char), _DISTANCE_NUMBA_TYPE
            ),
        ).ltoir,
        _ncc(
            f"{prefix}_dereference",
            transform_dereference,
            (numba.types.CPointer(numba.types.char),),
        ).ltoir,
        # ATTENTION: NOT op_caller here! (see issue #3064)
        op_ltoir,
    ]

    return TransformIteratorImpl(it, op.__name__, op_return_ntype, ltoirs)
