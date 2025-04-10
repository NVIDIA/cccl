import ctypes
import operator
import uuid
from functools import lru_cache
from typing import Any, Callable, Dict

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
        cvalue,
        numba_type: types.Type,
        state_type: types.Type,
        value_type: types.Type,
    ):
        """
        Parameters
        ----------
        cvalue
          A ctypes type representing the object pointed to by the iterator.
        numba_type
          A numba type representing the type of the input to the advance
          and dereference functions.
        state_type
          A numba type of the iterator state.
        value_type
          The numba type of the value returned by the dereference operation.
        """
        self.cvalue = cvalue
        self.numba_type = numba_type
        self.state_type = state_type
        self.value_type = value_type
        self.kind_ = self.__class__.iterator_kind_type(self.value_type, self.state_type)
        self.state_ = IteratorState(self.cvalue)
        self._ltoirs: Dict[str, bytes] | None = None

    @property
    def kind(self):
        return self.kind_

    @property
    def ltoirs(self) -> Dict[str, bytes]:
        if self._ltoirs is None:
            abi_suffix = _get_abi_suffix(self.kind)
            advance_abi_name = f"advance_{abi_suffix}"
            deref_abi_name = f"dereference_{abi_suffix}"
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

    @staticmethod
    def advance(state, distance):
        raise NotImplementedError("Subclasses must override advance staticmethod")

    @staticmethod
    def dereference(state):
        raise NotImplementedError("Subclasses must override dereference staticmethod")

    def __add__(self, offset: int):
        return make_advanced_iterator(self, offset=offset)

    def copy(self):
        out = object.__new__(self.__class__)
        IteratorBase.__init__(
            out, self.cvalue, self.numba_type, self.state_type, self.value_type
        )
        out.ltoirs = self.ltoirs
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

    def __init__(self, ptr: int, value_type: types.Type):
        cvalue = ctypes.c_void_p(ptr)
        state_type = types.CPointer(value_type)
        numba_type = types.CPointer(state_type)
        super().__init__(
            cvalue=cvalue,
            numba_type=numba_type,
            state_type=state_type,
            value_type=value_type,
        )

    @staticmethod
    def advance(state, distance):
        state[0] = state[0] + distance

    @staticmethod
    def dereference(state):
        return state[0][0]


def pointer(container, value_type: types.Type) -> RawPointer:
    return RawPointer(container.__cuda_array_interface__["data"][0], value_type)


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
        numba_type = types.CPointer(state_type)
        super().__init__(
            cvalue=cvalue,
            numba_type=numba_type,
            state_type=state_type,
            value_type=value_type,
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
        state_type = value_type
        numba_type = types.CPointer(state_type)
        super().__init__(
            cvalue=cvalue,
            numba_type=numba_type,
            state_type=state_type,
            value_type=value_type,
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
        state_type = value_type
        numba_type = types.CPointer(state_type)
        super().__init__(
            cvalue=cvalue,
            numba_type=numba_type,
            state_type=state_type,
            value_type=value_type,
        )

    @staticmethod
    def advance(state, distance):
        state[0] += distance

    @staticmethod
    def dereference(state):
        return state[0]


class ReverseIteratorKind(IteratorKind):
    pass


def make_reverse_iterator(it: DeviceArrayLike | IteratorBase):
    if not hasattr(it, "__cuda_array_interface__") and not isinstance(it, IteratorBase):
        raise NotImplementedError(
            f"Reverse iterator is not implemented for type {type(it)}"
        )

    if hasattr(it, "__cuda_array_interface__"):
        last_element_ptr = _get_last_element_ptr(it)
        it = RawPointer(last_element_ptr, numba.from_dtype(get_dtype(it)))

    it_advance = cuda.jit(type(it).advance, device=True)
    it_dereference = cuda.jit(type(it).dereference, device=True)

    class ReverseIterator(IteratorBase):
        iterator_kind_type = ReverseIteratorKind

        def __init__(self, it):
            self._it = it
            super().__init__(
                cvalue=it.cvalue,
                numba_type=it.numba_type,
                state_type=it.state_type,
                value_type=it.value_type,
            )
            self.kind_ = self.__class__.iterator_kind_type(
                (it.kind, it.value_type), it.state_type
            )

        @staticmethod
        def advance(state, distance):
            return it_advance(state, -distance)

        @staticmethod
        def dereference(state):
            return it_dereference(state)

    return ReverseIterator(it)


class TransformIteratorKind(IteratorKind):
    pass


def make_transform_iterator(it, op: Callable):
    if hasattr(it, "__cuda_array_interface__"):
        it = pointer(it, numba.from_dtype(it.dtype))

    it_advance = cuda.jit(type(it).advance, device=True)
    it_dereference = cuda.jit(type(it).dereference, device=True)
    op = cuda.jit(op, device=True)

    class TransformIterator(IteratorBase):
        iterator_kind_type = TransformIteratorKind

        def __init__(self, it: IteratorBase, op: CUDADispatcher):
            self._it = it
            self._op = CachableFunction(op.py_func)
            state_type = it.state_type
            numba_type = it.numba_type
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
                numba_type=numba_type,
                state_type=state_type,
                value_type=value_type,
            )
            self.kind_ = self.__class__.iterator_kind_type(
                (value_type, self._it.kind, self._op), state_type
            )

        @staticmethod
        def advance(state, distance):
            return it_advance(state, distance)

        @staticmethod
        def dereference(state):
            return op(it_dereference(state))

    return TransformIterator(it, op)


def make_advanced_iterator(it: IteratorBase, /, *, offset: int = 1):
    it_advance = cuda.jit(type(it).advance, device=True)
    it_dereference = cuda.jit(type(it).dereference, device=True)

    class AdvancedIteratorKind(IteratorKind):
        pass

    class AdvancedIterator(IteratorBase):
        iterator_kind_type = AdvancedIteratorKind

        def __init__(self, it: IteratorBase, advance_steps: int):
            self._it = it
            cvalue_advanced = type(it.cvalue)(it.cvalue.value + advance_steps)
            super().__init__(
                cvalue=cvalue_advanced,
                numba_type=it.numba_type,
                state_type=it.state_type,
                value_type=it.value_type,
            )
            self.kind_ = self.__class__.iterator_kind_type(
                (it.value_type, self._it.kind), it.state_type
            )

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


def _replace_duplicate_values(*ds, replacement_value):
    # given a sequence of dictionaries, return a sequence of dictionaries
    # such that for any found duplicate keys, the value is set to `replacement_value`.
    if len(ds) <= 1:
        return ds
    seen = set(ds[0].keys())
    for d in ds[1:]:
        for key in d:
            if key in seen:
                d[key] = replacement_value
        seen.update(d.keys())
    return ds


def scrub_duplicate_ltoirs(*maybe_iterators: Any) -> tuple[Any, ...]:
    """
    Scrub duplicate `ltoirs` from iterators in the provided sequence.

    If the sequence contains iterators with duplicate advance/dereference
    ltoirs, those are set to the empty byte string b"". This pre-processing
    step ensures that NVRTC doesn't see the same symbol defined more than
    once.
    """
    # extract just the iterators:
    iterators = [it.copy() for it in maybe_iterators if isinstance(it, IteratorBase)]

    # replace duplicate ltoirs with empty byte strings:
    ltoirs = _replace_duplicate_values(
        *(it.ltoirs for it in iterators), replacement_value=b""
    )
    for it, ltoir in zip(iterators, ltoirs):
        it.ltoirs = ltoir

    return maybe_iterators
