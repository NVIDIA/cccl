# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Numba registration for gpu_struct types.

This module provides the Numba integration for gpu_struct types. The struct types
themselves are created in the parent struct.py module without Numba, and this
module handles the lazy registration with Numba when needed for JIT compilation.

The main entry point is `_register_struct_with_numba(struct_class)`, which
registers a struct class (created by gpu_struct) with Numba's type system.
"""

import functools
import operator

import numba
import numpy as np
from numba import types
from numba.core import cgutils
from numba.core.extending import (
    make_attribute_wrapper,
    models,
    overload,
    register_model,
    typeof_impl,
)
from numba.core.typeconv import Conversion
from numba.core.typing import signature as nb_signature
from numba.core.typing.templates import AttributeTemplate, ConcreteTemplate
from numba.cuda.cudadecl import registry as cuda_registry
from numba.extending import as_numba_type, lower_builtin, lower_cast


def _convert_field_type_to_numba(val):
    """Convert a field type to a Numba type."""
    from ..struct import _Struct

    # If it's a nested struct class, get its Numba type
    if isinstance(val, type) and issubclass(val, _Struct):
        # Ensure the nested struct is registered first
        if not val._numba_registered:
            _register_struct_with_numba(val)
            val._numba_registered = True
        return as_numba_type(val)

    # If it's already a Numba type, return it
    if isinstance(val, types.Type):
        return val

    # Otherwise, convert from numpy dtype
    return numba.np.numpy_support.from_dtype(np.dtype(val))


def _is_struct_type(typ) -> bool:
    """Check if a type is a gpu_struct class."""
    from ..struct import _Struct

    return isinstance(typ, type) and issubclass(typ, _Struct)


@functools.lru_cache(maxsize=256)
def _register_struct_with_numba(struct_class: type) -> None:
    """
    Register a gpu_struct class with Numba's type system.

    This function is called lazily when the struct is first used with a
    JIT-compiled operation. It creates the Numba type, model, and lowering
    implementations needed for the struct to work in Numba.

    Args:
        struct_class: A struct class created by gpu_struct()
    """

    name = struct_class.__name__
    raw_field_spec = struct_class._field_spec  # type: ignore[attr-defined]

    # Convert all field types to Numba types
    field_spec = {
        fname: _convert_field_type_to_numba(ftype)
        for fname, ftype in raw_field_spec.items()
    }

    field_types = tuple(field_spec.values())

    # The internal Numba type for this struct
    class StructType(numba.types.Type):
        def __init__(self):
            super().__init__(name=name)

        def can_convert_from(self, typingctx, other):
            """Allow implicit conversion from tuples to struct."""

            # Check if tuple has the right number of elements
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

            return None

    numba_type = StructType()

    # Enables as_numba_type(py_type) -> numba_type
    as_numba_type.register(struct_class, numba_type)

    # Enables typeof(py_val) -> numba_type
    @typeof_impl.register(struct_class)
    def typeof_struct(val, c):
        return StructType()

    # Models the type as a struct in the Numba type system
    @register_model(StructType)
    class StructModel(models.StructModel):
        def __init__(self, dmm, fe_type):
            members = [(name, typ) for name, typ in field_spec.items()]
            super().__init__(dmm, fe_type, members)

    # Implements attribute access for the struct type
    class StructAttrsTemplate(AttributeTemplate):
        pass

    for field_name, field_numba_type in field_spec.items():

        def resolver(self, struct, _ftype=field_numba_type):
            return _ftype

        setattr(StructAttrsTemplate, f"resolve_{field_name}", resolver)

    for field_name in field_spec:
        make_attribute_wrapper(StructType, field_name, field_name)

    field_names_list = list(field_spec.keys())

    # Implements indexing for the struct type
    @overload(operator.getitem)
    def struct_getitem(struct_val, idx):
        if not isinstance(struct_val, StructType):
            return

        # Compile-time constant index
        if isinstance(idx, types.IntegerLiteral):
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

        # Runtime index
        conditions = "\n".join(
            f"    {'if' if i == 0 else 'elif'} idx == {i}: return struct_val.{name}"
            for i, name in enumerate(field_names_list)
        )
        exec(
            f"def impl(struct_val, idx):\n{conditions}\n    else: raise IndexError('Index out of range')",
            namespace := {},
        )
        return namespace["impl"]

    # Implements construction for the struct type
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

    # Register implicit cast from tuple to struct
    @lower_cast(types.BaseTuple, StructType)
    def tuple_to_struct_cast(context, builder, fromty, toty, val):
        """Cast a tuple to the struct type by unpacking elements."""

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
        """Lower implicit cast from tuple to struct."""
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


# Legacy exports for backward compatibility
def gpu_struct(field_dict, name: str = "AnonymousStruct"):
    """
    Legacy function for creating gpu_struct types.

    This function is deprecated. Use cuda.compute.gpu_struct instead.
    """
    from ..struct import gpu_struct as _gpu_struct

    struct_class = _gpu_struct(field_dict, name)

    # Immediately register with Numba for backward compatibility
    if not struct_class._numba_registered:  # type: ignore[attr-defined]
        _register_struct_with_numba(struct_class)
        struct_class._numba_registered = True  # type: ignore[attr-defined]

    return struct_class


@functools.lru_cache(maxsize=256)
def make_struct_type(name, field_names, field_types):
    """
    Create a struct type from field names and Numba types.

    This is used internally by iterators (ZipIterator, PermutationIterator)
    that need to create struct types from Numba types. Unlike gpu_struct(),
    this function accepts Numba types directly (including pointer types).

    For user-facing struct creation, use cuda.compute.gpu_struct() instead.
    """
    from types import new_class

    from ..struct import _Struct

    # Create struct class
    struct_class = new_class(name, bases=(_Struct,))
    struct_class.__name__ = name
    struct_class.__qualname__ = name

    # Store field spec with Numba types
    raw_field_spec = dict(zip(field_names, field_types))
    struct_class._field_spec = raw_field_spec
    struct_class._numba_registered = False

    # Convert Numba types to field spec for the internal StructType
    field_spec = {
        fname: _convert_field_type_to_numba(ftype)
        for fname, ftype in raw_field_spec.items()
    }

    # Create Numba type directly (no numpy dtype needed for internal structs)
    class StructType(numba.types.Type):
        def __init__(self):
            super().__init__(name=name)

        def can_convert_from(self, typingctx, other):
            if isinstance(other, types.UniTuple):
                if other.count == len(field_types):
                    return Conversion.safe
            elif isinstance(other, types.Tuple):
                if len(other.types) == len(field_types):
                    all_compatible = all(
                        typingctx.can_convert(src_type, tgt_type) is not None
                        for src_type, tgt_type in zip(other.types, field_spec.values())
                    )
                    if all_compatible:
                        return Conversion.safe
            return None

    numba_type = StructType()

    # Register with Numba
    as_numba_type.register(struct_class, numba_type)

    @typeof_impl.register(struct_class)
    def typeof_struct(val, c):
        return StructType()

    @register_model(StructType)
    class StructModel(models.StructModel):
        def __init__(self, dmm, fe_type):
            members = [(n, t) for n, t in field_spec.items()]
            super().__init__(dmm, fe_type, members)

    for field_name, field_numba_type in field_spec.items():

        def resolver(self, struct, _ftype=field_numba_type):
            return _ftype

        setattr(struct_class, f"resolve_{field_name}", resolver)

    for field_name in field_spec:
        make_attribute_wrapper(StructType, field_name, field_name)

    field_names_list = list(field_spec.keys())

    @overload(operator.getitem)
    def struct_getitem(struct_val, idx):
        if not isinstance(struct_val, StructType):
            return

        if isinstance(idx, types.IntegerLiteral):
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
            f"    {'if' if i == 0 else 'elif'} idx == {i}: return struct_val.{n}"
            for i, n in enumerate(field_names_list)
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
                f"Cannot cast tuple of size {tuple_size} to {name} "
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

    struct_class._numba_registered = True
    return struct_class
