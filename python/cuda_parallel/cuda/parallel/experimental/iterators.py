import ctypes
import operator

from numba.core import cgutils
from llvmlite import ir
from numba import types
from numba.core.typing import signature
from numba.core.extending import intrinsic, overload
import numba
import numba.cuda


def _sizeof_numba_type(ntype):
    mapping = {
        types.int8: 1,
        types.int16: 2,
        types.int32: 4,
        types.int64: 8,
        types.uint8: 1,
        types.uint16: 2,
        types.uint32: 4,
        types.uint64: 8,
        types.float16: 2,
        types.float32: 4,
        types.float64: 8,
    }
    return mapping[ntype]


def _ctypes_type_given_numba_type(ntype):
    mapping = {
        types.int8: ctypes.c_int8,
        types.int16: ctypes.c_int16,
        types.int32: ctypes.c_int32,
        types.int64: ctypes.c_int64,
        types.uint8: ctypes.c_uint8,
        types.uint16: ctypes.c_uint16,
        types.uint32: ctypes.c_uint32,
        types.uint64: ctypes.c_uint64,
        # types.float16: ctypes.c_half,  # c_half does not exist.
        types.float32: ctypes.c_float,
        types.float64: ctypes.c_double,
    }
    return mapping[ntype]


def _ncc(funcname, pyfunc, sig, prefix):
    return numba.cuda.compile(
        pyfunc=pyfunc,
        sig=sig,
        abi_info={"abi_name": f"{prefix}_{funcname}"},
        output="ltoir",
    )


def sizeof_pointee(context, ptr):
    size = context.get_abi_sizeof(ptr.type.pointee)
    return ir.Constant(ir.IntType(64), size)


@intrinsic
def pointer_add_intrinsic(context, ptr, offset):
    def codegen(context, builder, sig, args):
        ptr, index = args
        base = builder.ptrtoint(ptr, ir.IntType(64))
        offset = builder.mul(index, sizeof_pointee(context, ptr))
        result = builder.add(base, offset)
        return builder.inttoptr(result, ptr.type)

    return ptr(ptr, offset), codegen


@overload(operator.add)
def pointer_add(ptr, offset):
    if not isinstance(ptr, types.CPointer) or not isinstance(offset, types.Integer):
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
                "advance",
                RawPointer.pointer_advance,
                numba.types.void(data_as_ntype_pp, numba.types.uint64),
                self.prefix,
            ),
            _ncc(
                "dereference",
                RawPointer.pointer_dereference,
                ntype(data_as_ntype_pp),
                self.prefix,
            ),
        ]

    def pointer_advance(this, distance):
        this[0] = this[0] + distance

    def pointer_dereference(this):
        return this[0][0]

    def host_address(self):
        return ctypes.byref(self.val)

    def state_c_void_p(self):
        return ctypes.cast(ctypes.pointer(self.val), ctypes.c_void_p)

    def size(self):
        return 8  # TODO should be using numba for user-defined types support

    def alignment(self):
        return 8  # TODO should be using numba for user-defined types support


def pointer(container, ntype):
    return RawPointer(container.device_ctypes_pointer.value, ntype)


def _ir_type_given_numba_type(ntype):
    bw = ntype.bitwidth
    irt = None
    if isinstance(ntype, numba.core.types.scalars.Integer):
        irt = ir.IntType(bw)
    elif isinstance(ntype, numba.core.types.scalars.Float):
        if bw == 16:
            irt = ir.IntType(16)  # ir.HalfType() does not work here.
        elif bw == 32:
            irt = ir.FloatType()
        elif bw == 64:
            irt = ir.DoubleType()
    return irt


@intrinsic
def ldcs(typingctx, base):
    def codegen(context, builder, sig, args):
        rt = _ir_type_given_numba_type(sig.return_type)
        if rt is None:
            raise RuntimeError(f"Unsupported: {type(sig.return_type)=}")
        ftype = ir.FunctionType(rt, [rt.as_pointer()])
        bw = sig.return_type.bitwidth
        asm = f"ld.global.cs.b{bw} $0, [$1];"
        if bw < 64:
            constraint = "=r, l"
        else:
            constraint = "=l, l"
        ldcs = ir.InlineAsm(ftype, asm, constraint)
        return builder.call(ldcs, args)

    return base.dtype(base), codegen


class CacheModifiedPointer:
    def __init__(self, ptr, ntype):
        self.val = ctypes.c_void_p(ptr)
        self.ntype = ntype
        data_as_ntype_pp = numba.types.CPointer(numba.types.CPointer(ntype))
        self.prefix = "cache" + ntype.name
        self.ltoirs = [
            _ncc(
                "advance",
                CacheModifiedPointer.cache_advance,
                numba.types.void(data_as_ntype_pp, numba.types.uint64),
                self.prefix,
            ),
            _ncc(
                "dereference",
                CacheModifiedPointer.cache_dereference,
                ntype(data_as_ntype_pp),
                self.prefix,
            ),
        ]

    def cache_advance(this, distance):
        this[0] = this[0] + distance

    def cache_dereference(this):
        return ldcs(this[0])

    def host_address(self):
        return ctypes.byref(self.val)

    def state_c_void_p(self):
        return ctypes.cast(ctypes.pointer(self.val), ctypes.c_void_p)

    def size(self):
        return 8  # TODO should be using numba for user-defined types support

    def alignment(self):
        return 8  # TODO should be using numba for user-defined types support


def cache(container, ntype, modifier):
    if modifier != "stream":
        raise NotImplementedError("Only stream modifier is supported")
    return CacheModifiedPointer(container.device_ctypes_pointer.value, ntype)


class ConstantIterator:
    def __init__(self, val, ntype):
        thisty = numba.types.CPointer(ntype)
        self.val = _ctypes_type_given_numba_type(ntype)(val)
        self.ntype = ntype
        self.prefix = "constant_" + ntype.name
        self.ltoirs = [
            _ncc(
                "advance",
                ConstantIterator.constant_advance,
                numba.types.void(thisty, types.uint64),
                self.prefix,
            ),
            _ncc(
                "dereference",
                ConstantIterator.constant_dereference,
                ntype(thisty),
                self.prefix,
            ),
        ]

    def constant_advance(this, _):
        pass

    def constant_dereference(this):
        return this[0]

    def host_address(self):
        # TODO should use numba instead for support of user-defined types
        return ctypes.byref(self.val)

    def state_c_void_p(self):
        return ctypes.cast(ctypes.pointer(self.val), ctypes.c_void_p)

    def size(self):
        return ctypes.sizeof(
            self.val
        )  # TODO should be using numba for user-defined types support

    def alignment(self):
        return ctypes.alignment(
            self.val
        )  # TODO should be using numba for user-defined types support


def repeat(value, ntype):
    return ConstantIterator(value, ntype)


class CountingIterator:
    def __init__(self, count, ntype):
        thisty = numba.types.CPointer(ntype)
        self.count = _ctypes_type_given_numba_type(ntype)(count)
        self.ntype = ntype
        self.prefix = "count_" + ntype.name
        self.ltoirs = [
            _ncc(
                "advance",
                CountingIterator.count_advance,
                numba.types.void(thisty, types.uint64),
                self.prefix,
            ),
            _ncc(
                "dereference",
                CountingIterator.count_dereference,
                ntype(thisty),
                self.prefix,
            ),
        ]

    def count_advance(this, diff):
        this[0] += diff

    def count_dereference(this):
        return this[0]

    def host_address(self):
        return ctypes.byref(self.count)

    def state_c_void_p(self):
        return ctypes.cast(ctypes.pointer(self.count), ctypes.c_void_p)

    def size(self):
        return ctypes.sizeof(
            self.count
        )  # TODO should be using numba for user-defined types support

    def alignment(self):
        return ctypes.alignment(
            self.count
        )  # TODO should be using numba for user-defined types support


def count(offset, ntype):
    return CountingIterator(offset, ntype)


def cu_map(op, it, op_return_ntype):
    op_return_ntype_ir = _ir_type_given_numba_type(op_return_ntype)
    if op_return_ntype_ir is None:
        raise RuntimeError(f"Unsupported: {type(op_return_ntype)=}")
    it_ntype_ir = _ir_type_given_numba_type(it.ntype)
    if it_ntype_ir is None:
        raise RuntimeError(f"Unsupported: {type(it.ntype)=}")

    def source_advance(it_state_ptr, diff):
        pass

    def make_advance_codegen(name):
        retty = types.void
        statety = types.CPointer(types.int8)
        distty = types.uint64

        def codegen(context, builder, sig, args):
            state_ptr, dist = args
            fnty = ir.FunctionType(
                ir.VoidType(), (ir.PointerType(ir.IntType(8)), ir.IntType(64))
            )
            fn = cgutils.get_or_insert_function(builder.module, fnty, name)
            builder.call(fn, (state_ptr, dist))

        return signature(retty, statety, distty), codegen

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
        statety = types.CPointer(types.int8)

        def codegen(context, builder, sig, args):
            (state_ptr,) = args
            fnty = ir.FunctionType(op_return_ntype_ir, (ir.PointerType(ir.IntType(8)),))
            fn = cgutils.get_or_insert_function(builder.module, fnty, name)
            return builder.call(fn, (state_ptr,))

        return signature(op_return_ntype, statety), codegen

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
    op_codegen(op, op.__name__)

    class TransformIterator:
        def __init__(self, it, op):
            self.it = it  # TODO support row pointers
            self.op = op
            self.ntype = op_return_ntype
            self.prefix = f"transform_{it.prefix}_{op.__name__}"
            self.ltoirs = it.ltoirs + [
                _ncc(
                    "advance",
                    TransformIterator.transform_advance,
                    numba.types.void(
                        numba.types.CPointer(numba.types.char), numba.types.uint64
                    ),
                    self.prefix,
                ),
                _ncc(
                    "dereference",
                    TransformIterator.transform_dereference,
                    op_return_ntype(numba.types.CPointer(numba.types.char)),
                    self.prefix,
                ),
                numba.cuda.compile(op, sig=op_return_ntype(it.ntype), output="ltoir"),
            ]

        def transform_advance(it_state_ptr, diff):
            source_advance(it_state_ptr, diff)  # just a function call

        def transform_dereference(it_state_ptr):
            return op(source_dereference(it_state_ptr))  # just a function call

        def host_address(self):
            return self.it.host_address()  # TODO support stateful operators

        def state_c_void_p(self):
            return self.it.state_c_void_p()

        def size(self):
            return self.it.size()  # TODO fix for stateful op

        def alignment(self):
            return self.it.alignment()  # TODO fix for stateful op

    return TransformIterator(it, op)
