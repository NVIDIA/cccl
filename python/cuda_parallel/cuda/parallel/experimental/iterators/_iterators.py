import ctypes
import operator
import uuid
from enum import Enum
from functools import lru_cache
from typing import Callable, Dict, Tuple

import numba
import numpy as np
from llvmlite import ir
from numba import cuda, types
from numba.core.extending import intrinsic, overload
from numba.core.typing.ctypes_utils import to_ctypes
from numba.cuda.dispatcher import CUDADispatcher

from .._caching import CachableFunction
from .._utils.protocols import (
    compute_c_contiguous_strides_in_bytes,
    get_data_pointer,
    get_dtype,
    get_shape,
)
from ..typing import DeviceArrayLike

_DEVICE_POINTER_SIZE = 8
_DEVICE_POINTER_BITWIDTH = _DEVICE_POINTER_SIZE * 8


class IteratorIO(Enum):
    INPUT = 0
    OUTPUT = 1


@lru_cache(maxsize=256)  # TODO: what's a reasonable value?
def cached_compile(func, sig, abi_name=None, **kwargs):
    return cuda.compile(func, sig, abi_info={"abi_name": abi_name}, **kwargs)


class IteratorKind:
    def __init__(self, value_type):
        self.value_type = value_type

    def __repr__(self):
        return f"{self.__class__.__name__}[{str(self.value_type)}]"

    def __eq__(self, other):
        return type(self) is type(other) and self.value_type == other.value_type

    def __hash__(self):
        return hash((type(self), self.value_type))


@lru_cache(maxsize=None)
def _get_abi_suffix(kind: IteratorKind):
    # given an IteratorKind, return a UUID. The value is cached so
    # that the same UUID is always returned for a given IteratorKind.
    return uuid.uuid4().hex


class IteratorBase:
    """
    An Iterator is a wrapper around a pointer, and must define the following:

    - an `advance` (static) method that receives the pointer and performs
      an action that advances the pointer by the offset `distance`
      (returns nothing).
    - a `dereference` (static) method that dereferences the pointer
      and returns a value.

    Iterators are not meant to be used directly. They are constructed and passed
    to algorithms (e.g., `reduce`), which internally invoke their methods.

    The `advance` and `dereference` must be compilable to device code by numba.
    """

    iterator_kind_type: type  # must be a subclass of IteratorKind

    def __init__(
        self,
        cvalue: ctypes.c_void_p,
        numba_type: types.Type,
        value_type: types.Type,
        iterator_io: IteratorIO,
        prefix: str = "",
    ):
        """
        Parameters
        ----------
        cvalue
          A ctypes type representing the object pointed to by the iterator.
        numba_type
          A numba type representing the type of the input to the advance
          and dereference functions.
        value_type
          The numba type of the value returned by the dereference operation.
        prefix
          An optional prefix added to the iterator's methods to prevent name collisions.
        """
        self.cvalue = cvalue
        self.numba_type = numba_type
        self.value_type = value_type
        self.iterator_io = iterator_io
        self.prefix = prefix

    @property
    def kind(self):
        return self.__class__.iterator_kind_type(self.value_type)

    # TODO: should we cache this? Current docs environment doesn't allow
    # using Python > 3.7. We could use a hand-rolled cached_property if
    # needed.
    @property
    def ltoirs(self) -> Dict[str, bytes]:
        advance_abi_name = f"{self.prefix}advance_" + _get_abi_suffix(self.kind)
        deref_abi_name = f"{self.prefix}dereference_" + _get_abi_suffix(self.kind)
        advance_ltoir, _ = cached_compile(
            self.advance,
            self._get_advance_signature(),
            output="ltoir",
            abi_name=advance_abi_name,
        )

        deref_ltoir, _ = cached_compile(
            self.dereference,
            self._get_dereference_signature(),
            output="ltoir",
            abi_name=deref_abi_name,
        )
        return {advance_abi_name: advance_ltoir, deref_abi_name: deref_ltoir}

    @property
    def state(self) -> ctypes.c_void_p:
        return ctypes.cast(ctypes.pointer(self.cvalue), ctypes.c_void_p)

    @staticmethod
    def advance(state, distance):
        raise NotImplementedError("Subclasses must override advance staticmethod")

    @staticmethod
    def dereference(state, *args):
        raise NotImplementedError("Subclasses must override dereference staticmethod")

    def __add__(self, offset: int):
        return make_advanced_iterator(self, offset=offset)

    def _get_advance_signature(self) -> Tuple:
        return (
            self.numba_type,
            types.uint64,  # distance type
        )

    def _get_dereference_signature(self) -> Tuple:
        if self.iterator_io is IteratorIO.INPUT:
            return (self.numba_type,)
        else:
            # numba_type is a double pointer, so we get the datatype it points to
            dtype = self.numba_type.dtype.dtype
            return (self.numba_type, dtype)


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


class RawPointerKind(IteratorKind):
    pass


class RawPointer(IteratorBase):
    iterator_kind_type = RawPointerKind

    def __init__(self, ptr: int, value_type: types.Type, iterator_io: IteratorIO):
        cvalue = ctypes.c_void_p(ptr)
        numba_type = types.CPointer(types.CPointer(value_type))
        super().__init__(
            cvalue=cvalue,
            numba_type=numba_type,
            value_type=value_type,
            iterator_io=iterator_io,
        )

    @property
    def advance(self):
        return (
            RawPointer.input_advance
            if self.iterator_io is IteratorIO.INPUT
            else RawPointer.output_advance
        )

    @property
    def dereference(self):
        return (
            RawPointer.input_dereference
            if self.iterator_io is IteratorIO.INPUT
            else RawPointer.output_dereference
        )

    @staticmethod
    def input_advance(state, distance):
        state[0] = state[0] + distance

    @staticmethod
    def input_dereference(state):
        return state[0][0]

    @staticmethod
    def output_advance(state, distance):
        state[0] = state[0] + distance

    @staticmethod
    def output_dereference(state, x):
        state[0][0] = x


def pointer(container, value_type: types.Type) -> RawPointer:
    return RawPointer(
        container.__cuda_array_interface__["data"][0], value_type, IteratorIO.INPUT
    )


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


class CacheModifiedPointerKind(IteratorKind):
    pass


class CacheModifiedPointer(IteratorBase):
    iterator_kind_type = CacheModifiedPointerKind

    def __init__(self, ptr: int, ntype: types.Type, prefix: str):
        cvalue = ctypes.c_void_p(ptr)
        value_type = ntype
        numba_type = types.CPointer(types.CPointer(value_type))
        super().__init__(
            cvalue=cvalue,
            numba_type=numba_type,
            value_type=value_type,
            prefix=prefix,
            iterator_io=IteratorIO.INPUT,
        )

    @staticmethod
    def advance(state, distance):
        state[0] = state[0] + distance

    @staticmethod
    def dereference(state):
        return load_cs(state[0])


class ConstantIteratorKind(IteratorKind):
    pass


class ConstantIterator(IteratorBase):
    iterator_kind_type = ConstantIteratorKind

    def __init__(self, value: np.number):
        value_type = numba.from_dtype(value.dtype)
        cvalue = to_ctypes(value_type)(value)
        numba_type = types.CPointer(value_type)
        super().__init__(
            cvalue=cvalue,
            numba_type=numba_type,
            value_type=value_type,
            iterator_io=IteratorIO.INPUT,
        )

    @staticmethod
    def advance(state, distance):
        pass

    @staticmethod
    def dereference(state):
        return state[0]


class CountingIteratorKind(IteratorKind):
    pass


class CountingIterator(IteratorBase):
    iterator_kind_type = CountingIteratorKind

    def __init__(self, value: np.number):
        value_type = numba.from_dtype(value.dtype)
        cvalue = to_ctypes(value_type)(value)
        numba_type = types.CPointer(value_type)
        super().__init__(
            cvalue=cvalue,
            numba_type=numba_type,
            value_type=value_type,
            iterator_io=IteratorIO.INPUT,
        )

    @staticmethod
    def advance(state, distance):
        state[0] += distance

    @staticmethod
    def dereference(state):
        return state[0]


class ReverseInputIteratorKind(IteratorKind):
    pass


class ReverseOutputIteratorKind(IteratorKind):
    pass


def make_reverse_iterator(it: DeviceArrayLike | IteratorBase, iterator_io: IteratorIO):
    if not hasattr(it, "__cuda_array_interface__") and not isinstance(it, IteratorBase):
        raise NotImplementedError(
            f"Reverse iterator is not implemented for type {type(it)}"
        )

    if hasattr(it, "__cuda_array_interface__"):
        last_element_ptr = _get_last_element_ptr(it)
        it = RawPointer(last_element_ptr, numba.from_dtype(get_dtype(it)), iterator_io)

    it_advance = cuda.jit(it.advance, device=True)
    it_dereference = cuda.jit(it.dereference, device=True)

    class ReverseIterator(IteratorBase):
        iterator_kind_type = (
            ReverseInputIteratorKind
            if iterator_io is IteratorIO.INPUT
            else ReverseOutputIteratorKind
        )

        def __init__(self, it):
            self._it = it
            super().__init__(
                it.cvalue, it.numba_type, it.value_type, iterator_io=iterator_io
            )

        @property
        def kind(self):
            return self.__class__.iterator_kind_type(self._it.kind)

        @staticmethod
        def advance(state, distance):
            return it_advance(state, -distance)

        @property
        def dereference(self):
            return (
                ReverseIterator.input_dereference
                if self.iterator_io is IteratorIO.INPUT
                else ReverseIterator.output_dereference
            )

        @staticmethod
        def input_dereference(state):
            return it_dereference(state)

        @staticmethod
        def output_dereference(state, x):
            return it_dereference(state, x)

    return ReverseIterator(it)


class TransformIteratorKind(IteratorKind):
    pass


def make_transform_iterator(it, op: Callable):
    if hasattr(it, "__cuda_array_interface__"):
        it = pointer(it, numba.from_dtype(it.dtype))

    it_advance = cuda.jit(it.advance, device=True)
    it_dereference = cuda.jit(it.dereference, device=True)
    op = cuda.jit(op, device=True)

    class TransformIterator(IteratorBase):
        iterator_kind_type = TransformIteratorKind

        def __init__(self, it: IteratorBase, op: CUDADispatcher):
            self._it = it
            self._op = CachableFunction(op.py_func)
            numba_type = it.numba_type
            # TODO: it would be nice to not need to compile `op` to get
            # its return type, but there's nothing in the numba API
            # to do that (yet),
            _, op_retty = cached_compile(
                op,
                (self._it.value_type,),
                abi_name=f"{op.__name__}_{_get_abi_suffix(self.kind)}",
                output="ltoir",
            )
            value_type = op_retty
            super().__init__(
                cvalue=it.cvalue,
                numba_type=numba_type,
                value_type=value_type,
                iterator_io=it.iterator_io,
            )

        @property
        def kind(self):
            return self.__class__.iterator_kind_type((self._it.kind, self._op))

        @staticmethod
        def advance(state, distance):
            return it_advance(state, distance)

        @staticmethod
        def dereference(state):
            return op(it_dereference(state))

    return TransformIterator(it, op)


def make_advanced_iterator(it: IteratorBase, /, *, offset: int = 1):
    it_advance = cuda.jit(it.advance, device=True)
    it_dereference = cuda.jit(it.dereference, device=True)

    class AdvancedIteratorKind(IteratorKind):
        pass

    class AdvancedIterator(IteratorBase):
        iterator_kind_type = AdvancedIteratorKind

        def __init__(self, it: IteratorBase, advance_steps: int):
            self._it = it
            cvalue_advanced = to_ctypes(it.value_type)(
                it.cvalue + it.value_type(advance_steps)
            )
            super().__init__(
                cvalue=cvalue_advanced,
                numba_type=it.numba_type,
                value_type=it.value_type,
                iterator_io=it.iterator_io,
            )

        @property
        def kind(self):
            return self.__class__.iterator_kind_type(self._it.kind)

        @staticmethod
        def advance(state, distance):
            return it_advance(state, distance)

        @staticmethod
        def dereference(state):
            return it_dereference(state)

    return AdvancedIterator(it, offset)


def _get_last_element_ptr(device_array) -> int:
    shape = get_shape(device_array)
    dtype = get_dtype(device_array)

    strides_in_bytes = device_array.__cuda_array_interface__["strides"]
    if strides_in_bytes is None:
        strides_in_bytes = compute_c_contiguous_strides_in_bytes(shape, dtype.itemsize)

    offset_to_last_element = sum(
        (dim_size - 1) * stride for dim_size, stride in zip(shape, strides_in_bytes)
    )

    ptr = get_data_pointer(device_array)
    return ptr + offset_to_last_element
