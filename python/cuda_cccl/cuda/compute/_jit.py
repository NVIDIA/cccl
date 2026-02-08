# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ast
import functools
import inspect
import operator
import struct
import textwrap
import uuid
from types import new_class
from typing import Callable, Hashable, List, Tuple, get_type_hints

import numba
import numba.cuda
import numba.np.numpy_support
import numba.types
import numpy as np
from numba import types
from numba.core import cgutils
from numba.core.datamodel import models
from numba.core.extending import (
    as_numba_type,
    make_attribute_wrapper,
    overload,
    register_model,
    typeof_impl,
)
from numba.core.typeconv import Conversion
from numba.core.typing import signature as nb_signature
from numba.core.typing.templates import ConcreteTemplate
from numba.cuda.cudadecl import registry as cuda_registry
from numba.extending import lower_builtin, lower_cast

from . import types as cccl_types
from ._bindings import Op, OpKind
from ._caching import CachableFunction, cache_with_registered_key_functions
from ._odr_helpers import create_stateful_op_void_ptr_wrapper
from ._utils import sanitize_identifier
from ._utils.protocols import (
    get_data_pointer,
    get_dtype,
    is_contiguous,
    is_device_array,
)
from .op import OpAdapter
from .typing import DeviceArrayLike

# -----------------------------------------------------------------------------
# Struct registration and casting
# -----------------------------------------------------------------------------


# Base class for all struct types, used for struct-to-struct cast matching.
class _StructBase(numba.types.Type):
    """Base class for all CCCL GPU struct types."""

    _field_spec: dict  # Mapping of field names to Numba types


# The struct registration logic is isolated here to avoid polluting other
# modules with Numba-specific type plumbing.
@functools.lru_cache(maxsize=256)
def _make_struct_type(struct_class_or_name, field_names, field_types):
    """
    Core factory function that uses the Numba extension machinery to
    create a struct type with pass-by-value semantics.

    This function is called lazily from _register_struct_with_numba.

    Args:
        struct_class_or_name: Either an existing struct class (from gpu_struct) or a string name
        field_names: Tuple of field names
        field_types: Tuple of TypeDescriptors
    """
    from .struct import _is_struct_type, _Struct

    # If we're given an existing class, use it; otherwise create a new one
    if isinstance(struct_class_or_name, type):
        struct_class = struct_class_or_name
    else:
        struct_class = new_class(struct_class_or_name, bases=(_Struct,))

    # Convert TypeDescriptors to Numba types
    numba_field_types = [type_descriptor_to_numba(ft) for ft in field_types]

    raw_field_spec = dict(zip(field_names, numba_field_types))
    assert all(
        _is_struct_type(tp) or isinstance(tp, types.Type)
        for tp in raw_field_spec.values()
    )

    field_spec = {
        name: as_numba_type(typ) if _is_struct_type(typ) else typ
        for name, typ in raw_field_spec.items()
    }

    # Only set _field_spec if it doesn't already exist (for new classes created here)
    # For classes from gpu_struct, they already have _field_spec with TypeDescriptors
    if not hasattr(struct_class, "_field_spec"):
        struct_class._field_spec = raw_field_spec

    class StructType(_StructBase):
        def __init__(self):
            super().__init__(name=struct_class.__name__)
            # Store field_spec for struct-to-struct conversion checks
            self._field_spec = field_spec

        def can_convert_from(self, typingctx, other):
            if isinstance(other, types.UniTuple):
                tuple_size = other.count
                if tuple_size == len(field_types):
                    return Conversion.safe

            elif isinstance(other, types.Tuple):
                tuple_size = len(other.types)
                if tuple_size == len(field_types):
                    all_compatible = all(
                        typingctx.can_convert(src_type, tgt_type) is not None
                        for src_type, tgt_type in zip(other.types, field_spec.values())
                    )
                    if all_compatible:
                        return Conversion.safe

            # Allow conversion from another StructType with identical field layout
            elif hasattr(other, "_field_spec"):
                other_field_spec = other._field_spec
                # Check if field count matches
                if len(other_field_spec) == len(field_spec):
                    # Check if all field types are compatible (by position, not name)
                    other_field_types = list(other_field_spec.values())
                    self_field_types = list(field_spec.values())
                    all_compatible = all(
                        typingctx.can_convert(src_type, tgt_type) is not None
                        for src_type, tgt_type in zip(
                            other_field_types, self_field_types
                        )
                    )
                    if all_compatible:
                        return Conversion.safe

            return None

    numba_type = StructType()
    numba_type.python_type = struct_class

    as_numba_type.register(struct_class, numba_type)

    @typeof_impl.register(struct_class)
    def typeof_struct(val, c):
        return numba_type  # Must return the SAME instance, not a new StructType()

    @register_model(StructType)
    class StructModel(models.StructModel):
        def __init__(self, dmm, fe_type):
            members = [(name, typ) for name, typ in field_spec.items()]
            super().__init__(dmm, fe_type, members)

    for field_name in field_spec:
        make_attribute_wrapper(StructType, field_name, field_name)

    field_names_list = list(field_spec.keys())

    @overload(operator.getitem)
    def struct_getitem(struct_val, idx):
        if not isinstance(struct_val, StructType):
            return

        if isinstance(idx, (types.IntegerLiteral)):
            idx_val = getattr(idx, "literal_value", getattr(idx, "value", None))

            if idx_val is None or not (0 <= idx_val < len(field_names_list)):

                def error_impl(struct_val, idx):
                    raise IndexError(
                        f"Index out of range for struct with {len(field_names_list)} fields"
                    )

                return error_impl

            field_name = field_names_list[idx_val]
            exec(
                f"def impl(struct_val, idx): return struct_val.{field_name}",
                namespace := {},
            )
            return namespace["impl"]

        conditions = "\n".join(
            f"    {'if' if i == 0 else 'elif'} idx == {i}: return struct_val.{name}"
            for i, name in enumerate(field_names_list)
        )
        exec(
            f"def impl(struct_val, idx):\n{conditions}\n    else: raise IndexError('Index out of range')",
            namespace := {},
        )
        return namespace["impl"]

    @cuda_registry.register
    class StructConstructor(ConcreteTemplate):
        key = struct_class
        cases = [nb_signature(numba_type, *list(field_spec.values()))]

    cuda_registry.register_global(struct_class, numba.types.Function(StructConstructor))

    def struct_constructor(context, builder, sig, args):
        ty = sig.return_type
        retval = cgutils.create_struct_proxy(ty)(context, builder)
        for field_name, val in zip(field_spec.keys(), args):
            setattr(retval, field_name, val)
        return retval._getvalue()

    lower_builtin(struct_class, *list(field_spec.values()))(struct_constructor)

    @lower_cast(types.BaseTuple, StructType)
    def tuple_to_struct_cast(context, builder, fromty, toty, val):
        if isinstance(fromty, types.UniTuple):
            tuple_size = fromty.count
            element_types = [fromty.dtype] * tuple_size
        elif isinstance(fromty, types.Tuple):
            tuple_size = len(fromty.types)
            element_types = list(fromty.types)
        else:
            tuple_size = len(field_spec)
            element_types = list(field_spec.values())

        if tuple_size != len(field_spec):
            raise ValueError(
                f"Cannot cast tuple of size {tuple_size} to {struct_class.__name__} "
                f"with {len(field_types)} fields"
            )

        retval = cgutils.create_struct_proxy(toty)(context, builder)

        for i, (field_name, target_type) in enumerate(field_spec.items()):
            element = builder.extract_value(val, i)

            source_type = element_types[i]
            if source_type != target_type:
                element = context.cast(builder, element, source_type, target_type)

            setattr(retval, field_name, element)

        return retval._getvalue()

    @lower_cast(types.Tuple, StructType)
    @lower_cast(types.UniTuple, StructType)
    def cast_tuple_to_struct(context, builder, fromty, toty, val):
        if isinstance(fromty, types.UniTuple):
            if fromty.count != len(field_spec):
                return None
            tuple_types = [fromty.dtype] * fromty.count
        else:
            if len(fromty.types) != len(field_spec):
                return None
            tuple_types = list(fromty.types)

        struct_val = cgutils.create_struct_proxy(toty)(context, builder)
        for i, (field_name, field_type) in enumerate(field_spec.items()):
            elem = builder.extract_value(val, i)
            elem = context.cast(builder, elem, tuple_types[i], field_type)
            setattr(struct_val, field_name, elem)

        return struct_val._getvalue()

    @lower_cast(_StructBase, StructType)
    def cast_struct_to_struct(context, builder, fromty, toty, val):
        """Cast from one CCCL struct type to another with identical layout."""
        # Get field specs from both types
        from_field_spec = fromty._field_spec
        to_field_spec = toty._field_spec

        if len(from_field_spec) != len(to_field_spec):
            return None

        from_field_types = list(from_field_spec.values())
        from_field_names = list(from_field_spec.keys())
        to_field_types = list(to_field_spec.values())
        to_field_names = list(to_field_spec.keys())

        # Create struct proxy for source value
        from_struct = cgutils.create_struct_proxy(fromty)(context, builder, value=val)

        # Create struct proxy for target value
        to_struct = cgutils.create_struct_proxy(toty)(context, builder)

        # Copy and cast each field by position
        for i, (to_name, to_type) in enumerate(zip(to_field_names, to_field_types)):
            from_name = from_field_names[i]
            from_type = from_field_types[i]

            # Get the field value from source struct
            elem = getattr(from_struct, from_name)

            # Cast if types differ
            if from_type != to_type:
                elem = context.cast(builder, elem, from_type, to_type)

            # Set the field in target struct
            setattr(to_struct, to_name, elem)

        return to_struct._getvalue()

    return struct_class


@functools.lru_cache(maxsize=256)
def _register_struct_with_numba(struct_class):
    field_spec = struct_class._type_descriptor.fields

    registered_class = _make_struct_type(
        struct_class,
        tuple(field_spec.keys()),
        tuple(field_spec.values()),
    )

    return as_numba_type(registered_class)


# -----------------------------------------------------------------------------
# TypeDescriptor <-> Numba conversions
# -----------------------------------------------------------------------------


@functools.cache
def type_descriptor_to_numba(td):
    """
    Convert a TypeDescriptor to a Numba type.

    Handles:
    - PointerTypeDescriptor: creates CPointer to the pointee's numba type
    - StructTypeDescriptor: registers a struct class for the layout
    - POD TypeDescriptor: uses numba.from_dtype
    - Numba types: pass through
    """
    from . import types as cccl_types

    # Pass through if already a Numba type
    if isinstance(td, numba.types.Type):
        return td

    # Handle PointerTypeDescriptor (must check before TypeDescriptor since it's a subclass)
    if isinstance(td, cccl_types.PointerTypeDescriptor):
        return types.CPointer(type_descriptor_to_numba(td.pointee))

    # Handle TypeDescriptor (includes StructTypeDescriptor)
    if isinstance(td, cccl_types.TypeDescriptor):
        return _convert_type_descriptor_to_numba(td)

    raise TypeError(f"Expected TypeDescriptor or numba type, got {type(td)}")


def _convert_type_descriptor_to_numba(td):
    """Internal helper to convert TypeDescriptor to Numba type."""
    from . import types as cccl_types

    # For struct types
    if isinstance(td, cccl_types.StructTypeDescriptor):
        from .struct import (
            _get_struct_record_dtype,
            _get_struct_type_descriptor,
            _Struct,
        )

        layout_key = tuple(td.layout_key())
        struct_class = new_class(td.name, bases=(_Struct,))
        struct_class._field_spec = dict(layout_key)
        struct_class._type_descriptor = _get_struct_type_descriptor(struct_class)
        struct_class.dtype = _get_struct_record_dtype(struct_class)
        try:
            return as_numba_type(struct_class)
        except numba.core.errors.NumbaError:
            return _register_struct_with_numba(struct_class)

    # For POD types
    return numba.from_dtype(td.dtype)


def _is_gpu_struct_class(obj):
    """Check if an object is a gpu_struct class (not instance)."""
    return (
        isinstance(obj, type)
        and hasattr(obj, "_type_descriptor")
        and hasattr(obj, "_field_spec")
    )


def _iter_function_objects(py_func):
    if hasattr(py_func, "__globals__"):
        yield from py_func.__globals__.values()

    if py_func.__closure__ is not None:
        for cell in py_func.__closure__:
            try:
                yield cell.cell_contents
            except ValueError:
                pass


def _ensure_function_structs_registered(py_func):
    """
    Scan a function's globals and closure for gpu_struct classes and ensure
    they're registered with Numba before compilation.
    """

    def _register_if_needed(struct_class):
        try:
            return as_numba_type(struct_class)
        except numba.core.errors.NumbaError:
            return _register_struct_with_numba(struct_class)

    for value in _iter_function_objects(py_func):
        if _is_gpu_struct_class(value):
            _register_if_needed(value)


def _numba_type_to_type_descriptor(numba_type):
    """Convert a Numba type to a TypeDescriptor (internal helper)."""
    from . import types as cccl_types
    from .struct import _is_struct_type

    # Already a TypeDescriptor
    if isinstance(numba_type, cccl_types.TypeDescriptor):
        return numba_type

    # Custom StructType with an associated gpu_struct Python class
    if hasattr(numba_type, "python_type") and _is_struct_type(numba_type.python_type):
        return numba_type.python_type._type_descriptor

    # POD type - convert via numpy dtype
    dtype = numba.np.numpy_support.as_dtype(numba_type)
    return cccl_types.from_numpy_dtype(dtype)


def _annotation_to_type_descriptor(annotation):
    """
    Convert a type annotation to a TypeDescriptor.

    Handles:
    - TypeDescriptor: returns as-is
    - gpu_struct classes: returns their _type_descriptor
    - numpy dtypes/types: converts via from_numpy_dtype
    """
    from . import types as cccl_types
    from .struct import _is_struct_type

    if isinstance(annotation, cccl_types.TypeDescriptor):
        return annotation

    if _is_struct_type(annotation):
        return annotation._type_descriptor  # type: ignore[union-attr]

    # numpy dtype or type
    return cccl_types.from_numpy_dtype(np.dtype(annotation))


# -----------------------------------------------------------------------------
# Signature inference
# -----------------------------------------------------------------------------


def _infer_signature(py_func, input_types=None):
    """
    Infer the signature of a function as TypeDescriptors.

    If annotations are provided, uses those directly.
    Otherwise, compiles the function to infer types.

    Args:
        py_func: The Python function to analyze
        input_types: Optional tuple of TypeDescriptors for the inputs.
                     Used for inference when annotations are missing.

    Returns:
        Tuple of (input_type_descriptors, output_type_descriptor)
        where input_type_descriptors is a tuple of TypeDescriptor
    """
    try:
        annotations = get_type_hints(py_func)
    except Exception:
        annotations = py_func.__annotations__
    spec = inspect.getfullargspec(py_func)
    arg_names = list(spec.args)

    input_tds = []
    has_all_input_annotations = True

    # Try to get input types from annotations
    for name in arg_names:
        if name in annotations:
            input_tds.append(_annotation_to_type_descriptor(annotations[name]))
        else:
            has_all_input_annotations = False
            break

    # Try to get output type from return annotation
    output_td = None
    if "return" in annotations:
        output_td = _annotation_to_type_descriptor(annotations["return"])

    # If we have all annotations, we're done
    if has_all_input_annotations and output_td is not None:
        return tuple(input_tds), output_td

    # Need to infer by compiling
    if input_types is None:
        if not has_all_input_annotations:
            raise ValueError(
                "Function must have type annotations for all arguments, "
                "or input_types must be provided"
            )
        input_tds_for_compile = input_tds
    else:
        # Use provided input types (TypeDescriptors)
        input_tds_for_compile = input_types

    # Convert TypeDescriptors to Numba types for compilation
    input_numba_types = tuple(
        type_descriptor_to_numba(td) for td in input_tds_for_compile
    )

    # Ensure any gpu_struct classes referenced in the function are registered
    _ensure_function_structs_registered(py_func)

    # Compile to infer return type
    from ._utils import sanitize_identifier

    sanitized_name = sanitize_identifier(py_func.__name__)
    unique_suffix = hex(id(py_func))[2:]
    abi_name = f"{sanitized_name}_{unique_suffix}"
    _, return_type = numba.cuda.compile(
        py_func, input_numba_types, abi_info={"abi_name": abi_name}
    )
    output_td = _numba_type_to_type_descriptor(return_type)

    # Use provided input types or convert from numba types
    if input_types is not None:
        input_tds = list(input_types)
    elif not has_all_input_annotations:
        input_tds = [_numba_type_to_type_descriptor(t) for t in input_numba_types]

    return tuple(input_tds), output_td


# -----------------------------------------------------------------------------
# Iterator compilation
# -----------------------------------------------------------------------------


@functools.lru_cache(maxsize=256)
def _cached_compile(func, sig, abi_name=None, **kwargs):
    """Cached wrapper around numba.cuda.compile."""
    return numba.cuda.compile(func, sig, abi_info={"abi_name": abi_name}, **kwargs)


def _get_abi_suffix():
    """Generate a unique ABI suffix."""
    return uuid.uuid4().hex


def _resolve_iterator_value_types(it):
    # transform iterators sometimes need help figuring their input or
    # output types (depending on whether it's an input or output
    # iterator). This requires inspecting type annotations, or
    # using numba's type inference. This function takes an iterator
    # (possibly a compound iterator like ZipIterator) and traverses
    # its children recursively, finding any TransformIterators
    # and setting their value types. At the end, it calls
    # it._rebuild_value_type_from_children() which propagates
    # the updated value types back up to the "parent" iterators.
    from .iterators._iterators import TransformIteratorKind

    children = getattr(it, "children", ())
    for child in children:
        _resolve_iterator_value_types(child)

    kind = getattr(it, "kind", None)
    if isinstance(kind, TransformIteratorKind):
        op_func = kind.op._func
        _ensure_function_structs_registered(op_func)
        if kind.io_kind == "input":
            _, output_td = _infer_signature(op_func, (it.value_type,))
            it.value_type = output_td
        else:
            input_tds, _ = _infer_signature(op_func)
            it.value_type = input_tds[0]

    rebuild_value_type = getattr(it, "_rebuild_value_type_from_children", None)
    if rebuild_value_type is not None:
        rebuild_value_type()


def compile_iterator(it, io_kind: str):
    """
    Compile an iterator into a CCCL Iterator binding object.

    Args:
        it: The iterator to compile (an IteratorBase instance)
        io_kind: Either "input" or "output"

    Returns:
        An Iterator binding object ready for use with CCCL algorithms
    """
    from ._bindings import Iterator, IteratorKind, Op, OpKind
    from ._odr_helpers import (
        create_advance_void_ptr_wrapper,
        create_input_dereference_void_ptr_wrapper,
        create_output_dereference_void_ptr_wrapper,
    )

    _resolve_iterator_value_types(it)

    # Convert TypeDescriptors to Numba types for compilation
    numba_state_type = type_descriptor_to_numba(it.state_type)
    state_ptr_type = types.CPointer(numba_state_type)
    numba_value_type = type_descriptor_to_numba(it.value_type)

    # Validate state size using TypeDescriptor info
    # (PointerTypeDescriptor.info() returns 8 bytes, TypeDescriptor.info() returns dtype size)
    state_info = it.state_type.info()
    iterator_state = memoryview(it.state)
    if iterator_state.nbytes != state_info.size:
        raise ValueError(
            f"Iterator state size, {iterator_state.nbytes} bytes, for iterator type {type(it)} "
            f"does not match expected size, {state_info.size} bytes"
        )
    alignment = state_info.alignment

    # Compile advance operation
    advance_abi_name = f"advance_{_get_abi_suffix()}"
    wrapped_advance, wrapper_sig = create_advance_void_ptr_wrapper(
        it.advance, state_ptr_type
    )
    advance_ltoir, _ = _cached_compile(
        wrapped_advance, wrapper_sig, abi_name=advance_abi_name, output="ltoir"
    )
    advance_op = Op(
        operator_type=OpKind.STATELESS,
        name=advance_abi_name,
        ltoir=advance_ltoir,
    )

    # Compile dereference operation based on io_kind
    if io_kind == "input":
        deref_abi_name = f"input_dereference_{_get_abi_suffix()}"
        wrapped_deref, wrapper_sig = create_input_dereference_void_ptr_wrapper(
            it.input_dereference, state_ptr_type, numba_value_type
        )
    elif io_kind == "output":
        deref_abi_name = f"output_dereference_{_get_abi_suffix()}"
        wrapped_deref, wrapper_sig = create_output_dereference_void_ptr_wrapper(
            it.output_dereference, state_ptr_type, numba_value_type
        )
    else:
        raise ValueError(f"Invalid io_kind: {io_kind}. Must be 'input' or 'output'")

    deref_ltoir, _ = _cached_compile(
        wrapped_deref, wrapper_sig, abi_name=deref_abi_name, output="ltoir"
    )
    deref_op = Op(
        operator_type=OpKind.STATELESS,
        name=deref_abi_name,
        ltoir=deref_ltoir,
    )

    # Get TypeInfo directly from TypeDescriptor
    value_type_info = it.value_type.info()

    return Iterator(
        alignment,
        IteratorKind.ITERATOR,
        advance_op,
        deref_op,
        value_type_info,
        state=it.state,
    )


# -----------------------------------------------------------------------------
# Stateless ops
# -----------------------------------------------------------------------------


def _compile_op(op, input_types, output_type=None):
    """Compile a user-provided stateless operator for use with CCCL algorithms."""
    from . import types as cccl_types
    from ._bindings import Op, OpKind
    from ._odr_helpers import create_op_void_ptr_wrapper

    # Ensure any gpu_struct classes referenced in the op are registered
    _ensure_function_structs_registered(op)

    numba_input_types = tuple(
        type_descriptor_to_numba(t) if isinstance(t, cccl_types.TypeDescriptor) else t
        for t in input_types
    )

    if isinstance(output_type, cccl_types.TypeDescriptor):
        numba_output_type = type_descriptor_to_numba(output_type)
    else:
        numba_output_type = output_type

    sig = numba_output_type(*numba_input_types)
    wrapped_op, wrapper_sig = create_op_void_ptr_wrapper(op, sig)
    ltoir, _ = numba.cuda.compile(wrapped_op, sig=wrapper_sig, output="ltoir")
    return Op(
        operator_type=OpKind.STATELESS,
        name=wrapped_op.__name__,
        ltoir=ltoir,
        state_alignment=1,
        state=None,
    )


class _StatelessOp(OpAdapter):
    """Adapter for stateless callables."""

    __slots__ = ["_func", "_cachable"]

    def __init__(self, func: Callable):
        self._func = func
        self._cachable = CachableFunction(func)

    def compile(self, input_types, output_type=None) -> Op:
        return _compile_op(self._func, input_types, output_type)

    @property
    def func(self) -> Callable:
        """Access the wrapped callable."""
        return self._func


# -----------------------------------------------------------------------------
# Stateful ops
# -----------------------------------------------------------------------------
#
# Stateful ops are callables that capture device arrays (state) as globals or
# closure variables.
#
# When numba-cuda encounters a device function that references device
# arrays in its globals/closures, it captures pointers to the
# referenced arrays as constants.  If an object being referenced
# changes (e.g., common in a loop) the device function will be
# recompiled because it sees a new pointer.
#
# We avoid this by relying on support for stateful ops in the CCCL C
# library itself. Here's how this works
#
# 1. Detect any global/closure device arrays referenced by a given callable.
#
# 2. Transform the callable's AST to add parameters for each detected device array.
#
# 3. Compilation: the compiled device function whose LTOIR we
#    eventually pass to the CCCL C library must take a single `void*`
#    argument for is state.  The corresponding declaration of stateful
#    operators in CCCL C looks like:
#
#       extern "C" __device__ void foo(void* state, )`.
#
#    Thus, we have Thus, at compilation time, we
#    define some intrinsics to unpack a `void*` into typed arrays and
#    invoke the transformed callable from step 2.  Much of this is
#    implemented in _odr_helpers.py


def _detect_device_array_globals(func: Callable) -> List[Tuple[str, object]]:
    """
        Detect device arrays referenced as globals in a function.
    }
        Args:
            func: The function to inspect

        Returns:
            List of (name, array) tuples for detected device arrays
    """
    state_arrays = []
    code = func.__code__

    for name in code.co_names:
        val = func.__globals__.get(name)
        if val is not None and is_device_array(val):
            state_arrays.append((name, val))

    return state_arrays


def _detect_device_array_closures(func: Callable) -> List[Tuple[str, object]]:
    """
    Detect device arrays captured in function closures.

    Args:
        func: The function to inspect

    Returns:
        List of (name, array) tuples for detected device arrays
    """
    state_arrays: List[Tuple[str, object]] = []
    code = func.__code__
    closure = func.__closure__

    if closure is None:
        return state_arrays

    # co_freevars contains the names of closure variables
    for name, cell in zip(code.co_freevars, closure):
        try:
            val = cell.cell_contents
            if is_device_array(val):
                state_arrays.append((name, val))
        except ValueError:
            # Cell is empty
            pass

    return state_arrays


def _detect_all_device_arrays(func: Callable) -> List[Tuple[str, object]]:
    """
    Detect all device arrays referenced by a function (globals + closures).

    Args:
        func: The function to inspect

    Returns:
        List of (name, array) tuples for detected device arrays
    """
    globals_arrays = _detect_device_array_globals(func)
    closure_arrays = _detect_device_array_closures(func)
    return globals_arrays + closure_arrays


class _AddStateParameters(ast.NodeTransformer):
    """AST transformer that adds state parameters to a function definition."""

    def __init__(self, state_names: List[str]):
        self.state_names = state_names

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        # Prepend state parameters to the function arguments
        # Inner function signature: (state_arrays..., regular_args...)
        new_args = [ast.arg(arg=name, annotation=None) for name in self.state_names]
        node.args.args = new_args + node.args.args
        return node

    def visit_AsyncFunctionDef(
        self, node: ast.AsyncFunctionDef
    ) -> ast.AsyncFunctionDef:
        # Handle async functions the same way
        new_args = [ast.arg(arg=name, annotation=None) for name in self.state_names]
        node.args.args = new_args + node.args.args
        return node


def _transform_function_ast(func: Callable, state_names: List[str]) -> Callable:
    """
    Transform a function to add state arrays captured as globals or closures
    as explicit parameters.

    For example, if the function is:

        def func(x): return x + state[0]  # state is a global array

    Then the transformed function will be:

        def func(state, x): return x + state[0]

    Args:
        func: The original function
        state_names: Names of device arrays to add as parameters

    Returns:
        A new function with state arrays as explicit parameters,
        appearing after the regular parameters.

    Raises:
        ValueError: If the function source cannot be obtained
    """
    # Get source code
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError) as e:
        raise ValueError(
            f"Cannot get source code for function '{func.__name__}'. "
        ) from e

    # Dedent source (in case function is defined inside a class/function)
    source = textwrap.dedent(source)

    # Parse to AST
    tree = ast.parse(source)

    # Transform: add state parameters
    transformer = _AddStateParameters(state_names)
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)

    # Compile and execute to create new function
    # We need to provide the original function's globals for imports
    # AND inject closure variables so they're accessible in the new function
    globals_dict = func.__globals__.copy()

    # Inject closure variables (except state arrays which become parameters)
    if func.__closure__:
        for name, cell in zip(func.__code__.co_freevars, func.__closure__):
            if name not in state_names:  # Don't inject state arrays
                try:
                    globals_dict[name] = cell.cell_contents
                except ValueError:
                    pass  # Cell is empty

    local_ns: dict[str, Callable] = {}
    exec(
        compile(tree, filename=f"<auto_stateful:{func.__name__}>", mode="exec"),
        globals_dict,
        local_ns,
    )

    # Get the transformed function
    transformed_func = local_ns[func.__name__]

    return transformed_func


def _extract_state(func: Callable):
    # Detect device arrays
    state_info = _detect_all_device_arrays(func)

    # Extract names and arrays
    state_names = [name for name, _ in state_info]
    state_arrays: List[DeviceArrayLike] = [arr for _, arr in state_info]  # type: ignore[misc]

    return state_names, state_arrays


def _compile_stateful_op(op, input_types, state_arrays, output_type=None):
    """
    Compile a stateful operator for use with CCCL algorithms.

    Args:
        op: The operator function (already transformed to take state as explicit parameters)
        input_types: Tuple of TypeDescriptors for regular input arguments
        state_arrays: Tuple of device arrays containing state
        output_type: Optional TypeDescriptor for return value (inferred if None)

    Returns:
        Compiled Op object for C++ interop
    """

    # Ensure any gpu_struct classes referenced in the op are registered
    _ensure_function_structs_registered(op)

    # Validate all state arrays are contiguous
    for i, state_array in enumerate(state_arrays):
        if not is_contiguous(state_array):
            raise ValueError(f"state array {i} must be contiguous")

    # Convert input types to Numba types
    numba_input_types = tuple(type_descriptor_to_numba(t) for t in input_types)

    # Create Numba array types for state arrays
    state_array_types = [
        numba.types.Array(numba.from_dtype(get_dtype(s)), 1, "A") for s in state_arrays
    ]

    # Infer output type if needed
    if output_type is None:
        # Compile with Numba to infer return type
        # The transformed function expects (state_arrays..., regular_args...)
        all_numba_input_types = tuple(state_array_types) + numba_input_types
        sanitized_name = sanitize_identifier(op.__name__)
        unique_suffix = hex(id(op))[2:]
        abi_name = f"{sanitized_name}_{unique_suffix}"
        _, return_type = numba.cuda.compile(
            op, all_numba_input_types, abi_info={"abi_name": abi_name}
        )
        # Convert return type to TypeDescriptor
        output_type = cccl_types.from_numpy_dtype(
            numba.np.numpy_support.as_dtype(return_type)
        )

    # Convert output type to Numba type
    numba_output_type = type_descriptor_to_numba(output_type)

    # Build full signature: output_type(state_arrays..., regular_args...)
    sig = numba_output_type(*state_array_types, *numba_input_types)

    # Get state pointers - pointers to the device array data
    state_ptrs = [get_data_pointer(arr) for arr in state_arrays]

    # Get shape and itemsize from each state array
    state_info = []
    for state_array in state_arrays:
        state_info.append(
            {
                "shape": len(state_array),
                "itemsize": get_dtype(state_array).itemsize,
                "strides": get_dtype(state_array).itemsize,
            }
        )

    # All pointers have the same alignment, use pointer-sized int alignment
    state_alignment = np.dtype(np.intp).alignment

    # Create the stateful wrapper (constructs arrays from pointers)
    wrapped_op, wrapper_sig = create_stateful_op_void_ptr_wrapper(
        op, sig, state_array_types, state_info
    )

    # Compile the wrapper to LTOIR
    ltoir, _ = numba.cuda.compile(wrapped_op, sig=wrapper_sig, output="ltoir")

    # Pack all data pointers as bytes (sequentially)
    state_bytes = struct.pack(f"{len(state_ptrs)}P", *state_ptrs)

    # Return Op with STATEFUL kind and packed pointers
    return Op(
        operator_type=OpKind.STATEFUL,
        name=wrapped_op.__name__,
        ltoir=ltoir,
        state_alignment=state_alignment,
        state=state_bytes,
    )


class _JitOpState:
    def __init__(self, names: List[str], arrays: List[DeviceArrayLike]):
        self.names = names
        self.arrays = arrays

    def get_cache_key(self) -> Hashable:
        return (tuple(self.names), tuple(get_dtype(s) for s in self.arrays))

    def to_bytes(self):
        state_ptrs = [get_data_pointer(arr) for arr in self.arrays]
        return struct.pack(f"{len(state_ptrs)}P", *state_ptrs)


class _StatefulOp(OpAdapter):
    __slots__ = ["_func", "_cachable", "_state"]

    def __init__(self, func, state):
        self._func = func
        self._cachable = CachableFunction(self._func)
        self._state = state

    @property
    def state(self):
        return self._state

    def compile(self, input_types, output_type=None) -> Op:
        transformed_func = _transform_function_ast(self._func, self._state.names)
        return _compile_stateful_op(
            transformed_func, input_types, self._state.arrays, output_type
        )

    def get_cache_key(self):
        return (self._func.__name__, self._cachable, self._state.get_cache_key())

    @property
    def is_stateful(self) -> bool:
        return True

    def update_op_state(self, cccl_op) -> None:
        """
        Update state by detecting device arrays from the Python callable.

        Args:
            cccl_op: The compiled CCCL Op to update
            op: The original Python callable (needed to detect current arrays)
        """
        cccl_op.state = self._state.to_bytes()

    @property
    def func(self) -> Callable:
        """Access the wrapped callable."""
        return self._func


def to_jit_op_adapter(op: Callable) -> OpAdapter:
    """
    Convert the Python callable into the appropriate stateful or stateless
    OpAdapter, depending on whether or not it references any global state.
    """
    state_names, state_arrays = _extract_state(op)
    if state_names:
        return _StatefulOp(op, _JitOpState(state_names, state_arrays))
    else:
        return _StatelessOp(op)


cache_with_registered_key_functions.register(_StatelessOp, lambda op: op._cachable)

cache_with_registered_key_functions.register(_StatefulOp, lambda op: op.get_cache_key())


__all__ = [
    "to_jit_op_adapter",
    "compile_iterator",
    "type_descriptor_to_numba",
]
