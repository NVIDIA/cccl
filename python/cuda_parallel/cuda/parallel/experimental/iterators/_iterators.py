import ctypes
import operator
from functools import lru_cache
from typing import Dict, Callable

from llvmlite import ir
from numba.core.extending import intrinsic, overload
from numba.core.typing.ctypes_utils import to_ctypes
from numba import cuda, types
import numba
import numpy as np


_DEVICE_POINTER_SIZE = 8
_DEVICE_POINTER_BITWIDTH = _DEVICE_POINTER_SIZE * 8


@lru_cache(maxsize=256)  # TODO: what's a reasonable value?
def cached_compile(func, sig, abi_name=None, **kwargs):
    return cuda.compile(func, sig, abi_info={"abi_name": abi_name}, **kwargs)


class IteratorBase:
    """
    An Iterator is a wrapper around a pointer, and must define the following:

    - a `state` property that returns a `ctypes.c_void_p` object, representing
      a pointer to some data.
    - an `advance` (static) method that receives the state pointer and performs
      an action that advances the pointer by the offset `distance`
      (returns nothing).
    - a `dereference` (static) method that dereferences the state pointer
      and returns a value.

    Iterators are not meant to be used directly. They are constructed and passed
    to algorithms (e.g., `reduce`), which internally invoke their methods.

    The `advance` and `dereference` must be compilable to device code by numba.
    """

    def __init__(self, numba_type: types.Type, value_type: types.Type, abi_name: str):
        """
        Parameters
        ----------
        numba_type
          A numba type that specifies how to interpret the state pointer.
        value_type
          The numba type of the value returned by the dereference operation.
        abi_name
          A unique identifier that will determine the abi_names for the
          advance and dereference operations.
        """
        self.numba_type = numba_type
        self.value_type = value_type
        self.abi_name = abi_name

    # TODO: should we cache this? Current docs environment doesn't allow
    # using Python > 3.7. We could use a hand-rolled cached_property if
    # needed.
    @property
    def ltoirs(self) -> Dict[str, bytes]:
        advance_abi_name = self.abi_name + "_advance"
        deref_abi_name = self.abi_name + "_dereference"
        advance_ltoir, _ = cached_compile(
            self.__class__.advance,
            (
                self.numba_type,
                types.uint64,  # distance type
            ),
            output="ltoir",
            abi_name=advance_abi_name,
        )

        deref_ltoir, _ = cached_compile(
            self.__class__.dereference,
            (self.numba_type,),
            output="ltoir",
            abi_name=deref_abi_name,
        )
        return {advance_abi_name: advance_ltoir, deref_abi_name: deref_ltoir}

    @property
    def state(self) -> ctypes.c_void_p:
        raise NotImplementedError("Subclasses must override advance staticmethod")

    @staticmethod
    def advance(state, distance):
        raise NotImplementedError("Subclasses must override advance staticmethod")

    @staticmethod
    def dereference(state):
        raise NotImplementedError("Subclasses must override dereference staticmethod")


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
    def __init__(self, ptr: int, ntype: types.Type):
        value_type = ntype
        self._cvalue = ctypes.c_void_p(ptr)
        numba_type = types.CPointer(types.CPointer(value_type))
        abi_name = f"{self.__class__.__name__}_{str(value_type)}"
        super().__init__(
            numba_type=numba_type,
            value_type=value_type,
            abi_name=abi_name,
        )

    @staticmethod
    def advance(state, distance):
        state[0] = state[0] + distance

    @staticmethod
    def dereference(state):
        return state[0][0]

    @property
    def state(self) -> ctypes.c_void_p:
        return ctypes.cast(ctypes.pointer(self._cvalue), ctypes.c_void_p)


def pointer(container, ntype: types.Type) -> RawPointer:
    return RawPointer(container.__cuda_array_interface__["data"][0], ntype)


@intrinsic
def load_cs(typingctx, base):
    # Corresponding to `LOAD_CS` here:
    # https://nvidia.github.io/cccl/cub/api/classcub_1_1CacheModifiedInputIterator.html
    def codegen(context, builder, sig, args):
        rt = context.get_value_type(sig.return_type)
        if rt is None:
            raise RuntimeError(f"Unsupported return type: {type(sig.return_type)}")
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
    def __init__(self, ptr: int, ntype: types.Type):
        self._cvalue = ctypes.c_void_p(ptr)
        value_type = ntype
        numba_type = types.CPointer(types.CPointer(value_type))
        abi_name = f"{self.__class__.__name__}_{str(value_type)}"
        super().__init__(
            numba_type=numba_type,
            value_type=value_type,
            abi_name=abi_name,
        )

    @staticmethod
    def advance(state, distance):
        state[0] = state[0] + distance

    @staticmethod
    def dereference(state):
        return load_cs(state[0])

    @property
    def state(self) -> ctypes.c_void_p:
        return ctypes.cast(ctypes.pointer(self._cvalue), ctypes.c_void_p)


class ConstantIterator(IteratorBase):
    def __init__(self, value: np.number):
        value_type = numba.from_dtype(value.dtype)
        self._cvalue = to_ctypes(value_type)(value)
        numba_type = types.CPointer(value_type)
        abi_name = f"{self.__class__.__name__}_{str(value_type)}"
        super().__init__(
            numba_type=numba_type,
            value_type=value_type,
            abi_name=abi_name,
        )

    @staticmethod
    def advance(state, distance):
        pass

    @staticmethod
    def dereference(state):
        return state[0]

    @property
    def state(self) -> ctypes.c_void_p:
        return ctypes.cast(ctypes.pointer(self._cvalue), ctypes.c_void_p)


class CountingIterator(IteratorBase):
    def __init__(self, value: np.number):
        value_type = numba.from_dtype(value.dtype)
        self._cvalue = to_ctypes(value_type)(value)
        numba_type = types.CPointer(value_type)
        abi_name = f"{self.__class__.__name__}_{str(value_type)}"
        super().__init__(
            numba_type=numba_type,
            value_type=value_type,
            abi_name=abi_name,
        )

    @staticmethod
    def advance(state, distance):
        state[0] += distance

    @staticmethod
    def dereference(state):
        return state[0]

    @property
    def state(self) -> ctypes.c_void_p:
        return ctypes.cast(ctypes.pointer(self._cvalue), ctypes.c_void_p)


def make_transform_iterator(it, op: Callable):
    if hasattr(it, "__cuda_array_interface__"):
        it = pointer(it, numba.from_dtype(it.dtype))

    it_advance = cuda.jit(type(it).advance, device=True)
    it_dereference = cuda.jit(type(it).dereference, device=True)
    op = cuda.jit(op, device=True)

    class TransformIterator(IteratorBase):
        def __init__(self, it: IteratorBase, op):
            self._it = it
            numba_type = it.numba_type
            # TODO: the abi name below isn't unique enough when we have e.g.,
            # two identically named `op` functions with different
            # signatures, bytecodes, and/or closure variables.
            op_abi_name = f"{self.__class__.__name__}_{op.py_func.__name__}"

            # TODO: it would be nice to not need to compile `op` to get
            # its return type, but there's nothing in the numba API
            # to do that (yet),
            _, op_retty = cached_compile(
                op,
                (self._it.value_type,),
                abi_name=op_abi_name,
                output="ltoir",
            )
            value_type = op_retty
            abi_name = f"{self.__class__.__name__}_{it.abi_name}_{op_abi_name}"
            super().__init__(
                numba_type=numba_type,
                value_type=value_type,
                abi_name=abi_name,
            )

        @staticmethod
        def advance(state, distance):
            return it_advance(state, distance)

        @staticmethod
        def dereference(state):
            return op(it_dereference(state))

        @property
        def state(self) -> ctypes.c_void_p:
            return it.state

    return TransformIterator(it, op)
