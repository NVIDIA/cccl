import copy
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

from .._bindings import IteratorState
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


class IteratorIOKind(Enum):
    INPUT = 0
    OUTPUT = 1


@lru_cache(maxsize=256)  # TODO: what's a reasonable value?
def cached_compile(func, sig, abi_name=None, **kwargs):
    return cuda.compile(func, sig, abi_info={"abi_name": abi_name}, **kwargs)


class IteratorKind:
    def __init__(self, value_type, state_type):
        self.value_type = value_type
        self.state_type = state_type

    def __repr__(self):
        return (
            f"{self.__class__.__name__}[{str(self.value_type), str(self.state_type)}]"
        )

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self.value_type == other.value_type
            and self.state_type == other.state_type
        )

    def __hash__(self):
        return hash((type(self), self.value_type, self.state_type))


def _get_abi_suffix(kind: IteratorKind):
    return uuid.uuid4().hex


class IteratorBase:
    """
    An Iterator is a wrapper around a pointer, and must define the following:

    - an `advance` property that returns a (static) method which receives the
      pointer and performs an action that advances the pointer by the offset
      `distance` (returns nothing).
    - a `dereference` property that returns a (static) method which accepts the
      pointer and returns a value. For output iterators, `dereference` is used
      to write to the pointer, so it also the value to be written as an
      argument.

    Iterators are not meant to be used directly. They are constructed and passed
    to algorithms (e.g., `reduce`), which internally invoke their methods.

    The `advance` and `dereference` must be compilable to device code by numba.
    """

    iterator_kind_type: type  # must be a subclass of IteratorKind

    def __init__(
        self,
        cvalue,
        state_type: types.Type,
        value_type: types.Type,
        iterator_io: IteratorIOKind,
    ):
        """
        Parameters
        ----------
        cvalue
          A ctypes type representing the object pointed to by the iterator.
        state_type
          A numba type representing the type of the input to the advance
          and dereference functions. This should be a pointer type.
        value_type
          The numba type of the value returned by the dereference operation.
        iterator_io
          An enumerator specifying whether the iterator will be used as an input
          or output. This is used to select what methods that the `advance` and
          `dereference` properties will return.
        """
        self.cvalue = cvalue
        self.state_type = state_type
        self.state_ptr_type = types.CPointer(state_type)
        self.value_type = value_type

        self.iterator_io = iterator_io
        self.kind_ = self.__class__.iterator_kind_type(self.value_type, self.state_type)
        self.state_ = IteratorState(self.cvalue)
        self._ltoirs: Dict[str, bytes] | None = None

    @property
    def kind(self):
        return self.kind_

    @property
    def host_advance(self):
        return None

    @property
    def ltoirs(self) -> Dict[str, bytes]:
        if self._ltoirs is None:
            abi_suffix = _get_abi_suffix(self.kind)
            advance_abi_name = f"advance_{abi_suffix}"
            deref_abi_name = f"dereference_{abi_suffix}"
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
            self._ltoirs = {
                advance_abi_name: advance_ltoir,
                deref_abi_name: deref_ltoir,
            }
        assert self._ltoirs is not None
        return self._ltoirs

    @ltoirs.setter
    def ltoirs(self, value):
        self._ltoirs = value

    @property
    def state(self) -> IteratorState:
        return self.state_

    @property
    def advance(state):
        raise NotImplementedError("Subclasses must override advance property")

    @property
    def dereference(state):
        raise NotImplementedError("Subclasses must override dereference property")

    def __add__(self, offset: int):
        # add the offset to the iterator's state, and return a new iterator
        # with the new state.
        res = type(self).__new__(type(self))
        res.state_ptr_type = self.state_ptr_type
        res.state_type = self.state_type
        res.value_type = self.value_type
        res.iterator_io = self.iterator_io
        res.kind_ = self.kind_
        res._ltoirs = self._ltoirs
        res.cvalue = type(self.cvalue)(self.cvalue.value + offset)
        res.state_ = IteratorState(res.cvalue)

        return res

    def _get_advance_signature(self) -> Tuple:
        return (
            self.state_ptr_type,
            types.uint64,  # distance type
        )

    def _get_dereference_signature(self) -> Tuple:
        if self.iterator_io is IteratorIOKind.INPUT:
            # state, result
            return (self.state_ptr_type, types.CPointer(self.value_type))
        else:
            # state, value
            return (self.state_ptr_type, self.value_type)

    def copy(self):
        out = object.__new__(self.__class__)
        IteratorBase.__init__(
            out,
            self.cvalue,
            self.state_type,
            self.value_type,
            self.iterator_io,
        )
        out.ltoirs = copy.copy(self._ltoirs)
        return out


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

    def __init__(
        self, ptr: int, value_type: types.Type, iterator_io: IteratorIOKind, obj: object
    ):
        cvalue = ctypes.c_void_p(ptr)
        state_type = types.CPointer(value_type)
        self.obj = obj  # the container holding the data
        super().__init__(
            cvalue=cvalue,
            state_type=state_type,
            value_type=value_type,
            iterator_io=iterator_io,
        )

    @property
    def host_advance(self):
        """Raw pointer"""
        return self.input_advance

    @property
    def advance(self):
        return RawPointer.input_advance

    @property
    def dereference(self):
        return (
            RawPointer.input_dereference
            if self.iterator_io is IteratorIOKind.INPUT
            else RawPointer.output_dereference
        )

    @staticmethod
    def input_advance(state, distance):
        state[0] = state[0] + distance

    @staticmethod
    def input_dereference(state, result):
        result[0] = state[0][0]

    @staticmethod
    def output_dereference(state, x):
        state[0][0] = x


def pointer(container, value_type: types.Type) -> RawPointer:
    return RawPointer(
        container.__cuda_array_interface__["data"][0],
        value_type,
        IteratorIOKind.INPUT,
        container,
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

    def __init__(self, ptr: int, ntype: types.Type):
        cvalue = ctypes.c_void_p(ptr)
        value_type = ntype
        state_type = types.CPointer(value_type)
        super().__init__(
            cvalue=cvalue,
            state_type=state_type,
            value_type=value_type,
            iterator_io=IteratorIOKind.INPUT,
        )

    @property
    def host_advance(self):
        return self.input_advance

    @property
    def advance(self):
        return self.input_advance

    @property
    def dereference(self):
        return self.input_dereference

    @staticmethod
    def input_advance(state, distance):
        state[0] = state[0] + distance

    @staticmethod
    def input_dereference(state, result):
        result[0] = load_cs(state[0])


class ConstantIteratorKind(IteratorKind):
    pass


class ConstantIterator(IteratorBase):
    iterator_kind_type = ConstantIteratorKind

    def __init__(self, value: np.number):
        value_type = numba.from_dtype(value.dtype)
        cvalue = to_ctypes(value_type)(value)
        state_type = value_type
        super().__init__(
            cvalue=cvalue,
            state_type=state_type,
            value_type=value_type,
            iterator_io=IteratorIOKind.INPUT,
        )

    @property
    def host_advance(self):
        return self.input_advance

    @property
    def advance(self):
        return self.input_advance

    @property
    def dereference(self):
        return self.input_dereference

    @staticmethod
    def input_advance(state, distance):
        pass

    @staticmethod
    def input_dereference(state, result):
        result[0] = state[0]


class CountingIteratorKind(IteratorKind):
    pass


class CountingIterator(IteratorBase):
    iterator_kind_type = CountingIteratorKind

    def __init__(self, value: np.number):
        value_type = numba.from_dtype(value.dtype)
        cvalue = to_ctypes(value_type)(value)
        state_type = value_type
        super().__init__(
            cvalue=cvalue,
            state_type=state_type,
            value_type=value_type,
            iterator_io=IteratorIOKind.INPUT,
        )

    @property
    def host_advance(self):
        return self.input_advance

    @property
    def advance(self):
        return self.input_advance

    @property
    def dereference(self):
        return self.input_dereference

    @staticmethod
    def input_advance(state, distance):
        state[0] = state[0] + distance

    @staticmethod
    def input_dereference(state, result):
        result[0] = state[0]


class ReverseInputIteratorKind(IteratorKind):
    pass


class ReverseOutputIteratorKind(IteratorKind):
    pass


def make_reverse_iterator(
    it: DeviceArrayLike | IteratorBase, iterator_io: IteratorIOKind
):
    if not hasattr(it, "__cuda_array_interface__") and not isinstance(it, IteratorBase):
        raise NotImplementedError(
            f"Reverse iterator is not implemented for type {type(it)}"
        )

    if hasattr(it, "__cuda_array_interface__"):
        last_element_ptr = _get_last_element_ptr(it)
        it = RawPointer(
            last_element_ptr, numba.from_dtype(get_dtype(it)), iterator_io, it
        )

    it_advance = cuda.jit(it.advance, device=True)
    it_dereference = cuda.jit(it.dereference, device=True)

    class ReverseIterator(IteratorBase):
        iterator_kind_type = (
            ReverseInputIteratorKind
            if iterator_io is IteratorIOKind.INPUT
            else ReverseOutputIteratorKind
        )

        def __init__(self, it):
            self._it = it
            super().__init__(
                cvalue=it.cvalue,
                state_type=it.state_type,
                value_type=it.value_type,
                iterator_io=iterator_io,
            )
            self.kind_ = self.__class__.iterator_kind_type(
                (it.kind, it.value_type), it.state_type
            )

        @property
        def host_advance(self):
            return self.input_output_advance

        @property
        def advance(self):
            return self.input_output_advance

        @property
        def dereference(self):
            return (
                ReverseIterator.input_dereference
                if self.iterator_io is IteratorIOKind.INPUT
                else ReverseIterator.output_dereference
            )

        @staticmethod
        def input_output_advance(state, distance):
            return it_advance(state, -distance)

        @staticmethod
        def input_dereference(state, result):
            it_dereference(state, result)

        @staticmethod
        def output_dereference(state, x):
            it_dereference(state, x)

    return ReverseIterator(it)


class TransformIteratorKind(IteratorKind):
    pass


def make_transform_iterator(it, op: Callable):
    if hasattr(it, "__cuda_array_interface__"):
        it = pointer(it, numba.from_dtype(it.dtype))

    it_host_advance = it.host_advance
    it_advance = cuda.jit(it.advance, device=True)
    it_dereference = cuda.jit(it.dereference, device=True)
    op = cuda.jit(op, device=True)
    underlying_value_type = it.value_type

    # Create a specialized intrinsic for allocating temp storage of the underlying type
    @intrinsic
    def alloca_temp_for_underlying_type(context):
        def codegen(context, builder, sig, args):
            temp_value_type = context.get_value_type(underlying_value_type)
            temp_ptr = builder.alloca(temp_value_type)
            return temp_ptr

        return types.CPointer(underlying_value_type)(), codegen

    class TransformIterator(IteratorBase):
        iterator_kind_type = TransformIteratorKind

        def __init__(self, it: IteratorBase, op: CUDADispatcher):
            self._it = it
            self._op = CachableFunction(op.py_func)
            state_type = it.state_type
            # TODO: it would be nice to not need to compile `op` to get
            # its return type, but there's nothing in the numba API
            # to do that (yet),
            _, op_retty = cached_compile(
                op,
                (self._it.value_type,),
                abi_name=f"{op.__name__}_{_get_abi_suffix(self._it.kind)}",
                output="ltoir",
            )
            value_type = op_retty
            super().__init__(
                cvalue=it.cvalue,
                state_type=state_type,
                value_type=value_type,
                iterator_io=it.iterator_io,
            )
            self.kind_ = self.__class__.iterator_kind_type(
                (value_type, self._it.kind, self._op), self.state_type
            )

        @property
        def host_advance(self):
            return it_host_advance

        @property
        def advance(self):
            return self.input_advance

        @property
        def dereference(self):
            return self.input_dereference

        @staticmethod
        def input_advance(state, distance):
            return it_advance(state, distance)

        @staticmethod
        def input_dereference(state, result):
            # Allocate temporary storage for the underlying type
            temp_ptr = alloca_temp_for_underlying_type()
            # Call underlying iterator's dereference with temp storage
            it_dereference(state, temp_ptr)
            # Apply transformation and store in result
            result[0] = op(temp_ptr[0])

    return TransformIterator(it, op)


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
