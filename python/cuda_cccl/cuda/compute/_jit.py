# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
import inspect
import operator
from types import new_class
from typing import get_type_hints

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
def make_struct_type(struct_class_or_name, field_names, field_types):
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

    registered_class = make_struct_type(
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


def get_input_types_from_annotations(py_func):
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

    if not has_all_input_annotations:
        raise ValueError("Function must have type annotations for all arguments")
    return input_tds


def get_or_infer_return_type(py_func, input_types):
    try:
        annotations = get_type_hints(py_func)
    except Exception:
        annotations = py_func.__annotations__

    # Try to get output type from return annotation
    if "return" in annotations:
        return _annotation_to_type_descriptor(annotations["return"])

    # Ensure any gpu_struct classes referenced in the function are registered
    _ensure_function_structs_registered(py_func)

    # Compile to infer return type
    from ._utils import sanitize_identifier

    sanitized_name = sanitize_identifier(py_func.__name__)
    unique_suffix = hex(id(py_func))[2:]
    abi_name = f"{sanitized_name}_{unique_suffix}"
    input_numba_types = tuple(type_descriptor_to_numba(t) for t in input_types)
    _, return_type = numba.cuda.compile(
        py_func, input_numba_types, abi_info={"abi_name": abi_name}
    )
    return _numba_type_to_type_descriptor(return_type)


@functools.lru_cache(maxsize=256)
def _compile_op_impl(cachable_op, input_types_tuple: tuple, output_type):
    """Cached implementation of op compilation.

    Args:
        cachable_op: CachableFunction wrapper around the operator
        input_types_tuple: Tuple of input TypeDescriptors
        output_type: Output TypeDescriptor
    """
    from . import types as cccl_types
    from ._bindings import Op, OpKind
    from ._odr_helpers import create_op_void_ptr_wrapper

    # Extract the actual function from CachableFunction
    op = cachable_op._func

    # Ensure any gpu_struct classes referenced in the op are registered
    _ensure_function_structs_registered(op)

    numba_input_types = tuple(
        type_descriptor_to_numba(t) if isinstance(t, cccl_types.TypeDescriptor) else t
        for t in input_types_tuple
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


def compile_op(op, input_types, output_type=None):
    """Compile a user-provided binary operator for use with CCCL algorithms.

    This function is cached to ensure that identical operators with identical
    types produce the same compiled result (same symbol names and LTOIR),
    which allows proper deduplication during linking.
    """
    from ._caching import CachableFunction

    cachable_op = CachableFunction(op)
    return _compile_op_impl(cachable_op, tuple(input_types), output_type)


__all__ = [
    "get_input_types_from_annotations",
    "get_or_infer_return_type",
    "compile_op",
]
