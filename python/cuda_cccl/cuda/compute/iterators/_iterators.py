import ctypes
import operator
import uuid
from functools import lru_cache
from typing import Callable, Tuple

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
from ..numba_utils import get_inferred_return_type, signature_from_annotations
from ..typing import DeviceArrayLike

_DEVICE_POINTER_SIZE = 8
_DEVICE_POINTER_BITWIDTH = _DEVICE_POINTER_SIZE * 8


@lru_cache(maxsize=256)  # TODO: what's a reasonable value?
def cached_compile(func, sig, abi_name=None, **kwargs):
    return cuda.compile(func, sig, abi_info={"abi_name": abi_name}, **kwargs)


class IteratorKind:
    # The `.kind` of an iterator encapsulates additional metadata about the iterator,
    # analogous to the `.dtype` of a NumPy array.
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
    An Iterator is a wrapper around a state pointer, and must define the following:

    - an `advance` property returning a (static) method which receives the
      state pointer and performs an action that advances the state pointer by the offset
      `distance` (returns nothing).
    - `input_dereference` and `output_dereference` properties that return
      (static) methods for reading from and writing to the state pointer respectively.

    Iterators are not meant to be used directly. They are constructed and passed
    to algorithms which internally invoke their methods.

    The `advance`, `input_dereference`, and `output_dereference` must be compilable
    to device code by numba.
    """

    iterator_kind_type: type  # must be a subclass of IteratorKind

    def __init__(
        self,
        cvalue,
        state_type: types.Type,
        value_type: types.Type,
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
        """
        self.cvalue = cvalue
        self.state_type = state_type
        self.state_ptr_type = types.CPointer(state_type)
        self.value_type = value_type

        self._kind = self.__class__.iterator_kind_type(self.value_type, self.state_type)
        self._state = IteratorState(self.cvalue)

    @property
    def kind(self):
        return self._kind

    @property
    def state(self) -> IteratorState:
        return self._state

    @property
    def advance(state):
        raise NotImplementedError("Subclasses must override advance property")

    @property
    def input_dereference(state):
        return None

    @property
    def output_dereference(state):
        return None

    @property
    def host_advance(self):
        return None

    @property
    def is_input_iterator(self) -> bool:
        return self.input_dereference is not None

    @property
    def is_output_iterator(self) -> bool:
        return self.output_dereference is not None

    def get_advance_ltoir(self) -> Tuple:
        from .._odr_helpers import create_advance_void_ptr_wrapper

        abi_name = f"advance_{_get_abi_suffix(self.kind)}"
        wrapped_advance, wrapper_sig = create_advance_void_ptr_wrapper(
            self.advance, self.state_ptr_type
        )
        ltoir, _ = cached_compile(
            wrapped_advance,
            wrapper_sig,
            output="ltoir",
            abi_name=abi_name,
        )
        return (abi_name, ltoir)

    def get_input_dereference_ltoir(self) -> Tuple:
        from .._odr_helpers import create_input_dereference_void_ptr_wrapper

        abi_name = f"input_dereference_{_get_abi_suffix(self.kind)}"
        wrapped_deref, wrapper_sig = create_input_dereference_void_ptr_wrapper(
            self.input_dereference, self.state_ptr_type, self.value_type
        )
        ltoir, _ = cached_compile(
            wrapped_deref,
            wrapper_sig,
            output="ltoir",
            abi_name=abi_name,
        )
        return (abi_name, ltoir)

    def get_output_dereference_ltoir(self) -> Tuple:
        from .._odr_helpers import create_output_dereference_void_ptr_wrapper

        abi_name = f"output_dereference_{_get_abi_suffix(self.kind)}"
        wrapped_deref, wrapper_sig = create_output_dereference_void_ptr_wrapper(
            self.output_dereference, self.state_ptr_type, self.value_type
        )
        ltoir, _ = cached_compile(
            wrapped_deref,
            wrapper_sig,
            output="ltoir",
            abi_name=abi_name,
        )
        return (abi_name, ltoir)

    def __add__(self, offset: int):
        # add the offset to the iterator's state, and return a new iterator
        # with the new state.
        res = type(self).__new__(type(self))
        res.state_ptr_type = self.state_ptr_type
        res.state_type = self.state_type
        res.value_type = self.value_type
        res._kind = self._kind
        res.cvalue = type(self.cvalue)(self.cvalue.value + offset)
        res._state = IteratorState(res.cvalue)

        return res


def sizeof_pointee(context, ptr):
    size = context.get_abi_sizeof(ptr.type.pointee)
    return ir.Constant(ir.IntType(_DEVICE_POINTER_BITWIDTH), size)


@intrinsic
def pointer_add_intrinsic(context, ptr, offset):
    def codegen(context, builder, sig, args):
        ptr, index = args
        base = builder.ptrtoint(ptr, ir.IntType(_DEVICE_POINTER_BITWIDTH))
        sizeof = sizeof_pointee(context, ptr)
        # Cast index to match sizeof type if needed
        if index.type != sizeof.type:
            index = (
                builder.sext(index, sizeof.type)
                if index.type.width < sizeof.type.width
                else builder.trunc(index, sizeof.type)
            )
        offset = builder.mul(index, sizeof)
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

    def __init__(self, ptr: int, value_type: types.Type, obj: object):
        cvalue = ctypes.c_void_p(ptr)
        state_type = types.CPointer(value_type)
        self.obj = obj  # the container holding the data
        super().__init__(
            cvalue=cvalue,
            state_type=state_type,
            value_type=value_type,
        )

    @property
    def host_advance(self):
        return self._advance

    @property
    def advance(self):
        return self._advance

    @staticmethod
    def _advance(state, distance):
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
        )

    @property
    def host_advance(self):
        return self._advance

    @property
    def advance(self):
        return self._advance

    @property
    def input_dereference(self):
        return self._input_dereference

    @property
    def output_dereference(self):
        raise AttributeError(
            "CacheModifiedPointer cannot be used as an output iterator"
        )

    @staticmethod
    def _advance(state, distance):
        state[0] = state[0] + distance

    @staticmethod
    def _input_dereference(state, result):
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
        )

    @property
    def host_advance(self):
        return self._advance

    @property
    def advance(self):
        return self._advance

    @property
    def input_dereference(self):
        return self._input_dereference

    @property
    def output_dereference(self):
        raise AttributeError("ConstantIterator cannot be used as an output iterator")

    @staticmethod
    def _advance(state, distance):
        pass

    @staticmethod
    def _input_dereference(state, result):
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
        )

    @property
    def host_advance(self):
        return self._advance

    @property
    def advance(self):
        return self._advance

    @property
    def input_dereference(self):
        return self._input_dereference

    @property
    def output_dereference(self):
        raise AttributeError("CountingIterator cannot be used as an output iterator")

    @staticmethod
    def _advance(state, distance):
        state[0] = state[0] + distance

    @staticmethod
    def _input_dereference(state, result):
        result[0] = state[0]


class DiscardIteratorKind(IteratorKind):
    pass


class DiscardIterator(IteratorBase):
    iterator_kind_type = DiscardIteratorKind

    def __init__(self, reference_iterator=None):
        from .._utils.temp_storage_buffer import TempStorageBuffer

        if reference_iterator is None:
            reference_iterator = TempStorageBuffer(1)

        if hasattr(reference_iterator, "__cuda_array_interface__"):
            iter = RawPointer(
                reference_iterator.__cuda_array_interface__["data"][0],
                numba.from_dtype(get_dtype(reference_iterator)),
                reference_iterator,
            )
        else:
            iter = reference_iterator

        super().__init__(
            cvalue=iter.cvalue,
            state_type=iter.state_type,
            value_type=iter.value_type,
        )

    @property
    def host_advance(self):
        return self._advance

    @property
    def advance(self):
        return self._advance

    @property
    def input_dereference(self):
        return self._dereference

    @property
    def output_dereference(self):
        return self._dereference

    @staticmethod
    def _advance(state, distance):
        pass

    @staticmethod
    def _dereference(state, result):
        pass


class ReverseIteratorKind(IteratorKind):
    pass


def make_reverse_iterator(it: DeviceArrayLike | IteratorBase):
    if not hasattr(it, "__cuda_array_interface__") and not isinstance(it, IteratorBase):
        raise NotImplementedError(
            f"Reverse iterator is not implemented for type {type(it)}"
        )

    if hasattr(it, "__cuda_array_interface__"):
        last_element_ptr = _get_last_element_ptr(it)
        it = RawPointer(last_element_ptr, numba.from_dtype(get_dtype(it)), it)

    it_advance = cuda.jit(it.advance, device=True)
    it_input_dereference = (
        cuda.jit(it.input_dereference, device=True)
        if hasattr(it, "input_dereference")
        else None
    )
    it_output_dereference = (
        cuda.jit(it.output_dereference, device=True)
        if hasattr(it, "output_dereference")
        else None
    )

    class ReverseIterator(IteratorBase):
        iterator_kind_type = ReverseIteratorKind

        def __init__(self, it):
            self._it = it
            super().__init__(
                cvalue=it.cvalue,
                state_type=it.state_type,
                value_type=it.value_type,
            )
            self._kind = self.__class__.iterator_kind_type(
                (it.kind, it.value_type), it.state_type
            )

        @property
        def host_advance(self):
            return self._advance

        @property
        def advance(self):
            return self._advance

        @property
        def input_dereference(self):
            if it_input_dereference is None:
                raise AttributeError("This iterator is not an input iterator")
            return ReverseIterator._input_dereference

        @property
        def output_dereference(self):
            if it_output_dereference is None:
                raise AttributeError("This iterator is not an output iterator")
            return ReverseIterator._output_dereference

        @staticmethod
        def _advance(state, distance):
            return it_advance(state, -distance)

        @staticmethod
        def _input_dereference(state, result):
            it_input_dereference(state, result)

        @staticmethod
        def _output_dereference(state, x):
            it_output_dereference(state, x)

    return ReverseIterator(it)


class TransformIteratorKind(IteratorKind):
    pass


def make_transform_iterator(it, op: Callable, io_kind: str):
    if hasattr(it, "__cuda_array_interface__"):
        it = pointer(it, numba.from_dtype(it.dtype))

    it_host_advance = it.host_advance
    it_advance = cuda.jit(it.advance, device=True)

    it_input_dereference = (
        cuda.jit(it.input_dereference, device=True)
        if hasattr(it, "input_dereference")
        else None
    )
    it_output_dereference = (
        cuda.jit(it.output_dereference, device=True)
        if hasattr(it, "output_dereference")
        else None
    )

    if io_kind == "input":
        underlying_it_type = it.value_type

    op = cuda.jit(op, device=True)

    # Create a specialized intrinsic for allocating temp storage of the underlying type
    @intrinsic
    def alloca_temp_for_underlying_type(context):
        def codegen(context, builder, sig, args):
            temp_value_type = context.get_value_type(underlying_it_type)
            temp_ptr = builder.alloca(temp_value_type)
            return temp_ptr

        return types.CPointer(underlying_it_type)(), codegen

    class TransformIterator(IteratorBase):
        iterator_kind_type = TransformIteratorKind

        def __init__(self, it: IteratorBase, op: CUDADispatcher, io_kind: str):
            self._it = it
            self._op = CachableFunction(op.py_func)
            state_type = it.state_type

            if io_kind == "input":
                # For input iterators, use annotations if available, otherwise
                # rely on numba to infer the return type.
                try:
                    value_type = signature_from_annotations(op.py_func).args[0]
                except ValueError:
                    value_type = get_inferred_return_type(
                        op.py_func, (underlying_it_type,)
                    )
            else:
                # For output iterators, always require annotations.
                # The inferred type may not match the iterator being
                # written to.
                value_type = signature_from_annotations(op.py_func).args[0]

            super().__init__(
                cvalue=it.cvalue,
                state_type=state_type,
                value_type=value_type,
            )
            self._kind = self.__class__.iterator_kind_type(
                (self._it.value_type, self._it.kind, self._op), self.state_type
            )

        @property
        def host_advance(self):
            return it_host_advance

        @property
        def advance(self):
            return self._advance

        @property
        def input_dereference(self):
            if it_input_dereference is None:
                raise AttributeError("This iterator is not an input iterator")
            return TransformIterator._input_dereference

        @property
        def output_dereference(self):
            if it_output_dereference is None:
                raise AttributeError("This iterator is not an output iterator")
            return TransformIterator._output_dereference

        @staticmethod
        def _advance(state, distance):
            return it_advance(state, distance)

        @staticmethod
        def _input_dereference(state, result):
            # Allocate temporary storage for the deref input type
            temp_ptr = alloca_temp_for_underlying_type()
            # Call underlying iterator's dereference with temp storage
            it_input_dereference(state, temp_ptr)
            # Apply transformation and store in result
            result[0] = op(temp_ptr[0])

        @staticmethod
        def _output_dereference(state, x):
            it_output_dereference(state, op(x))

    return TransformIterator(it, op, io_kind)


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
