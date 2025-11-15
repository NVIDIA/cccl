# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
This module provides `gpu_struct`, a factory for producing "struct" types that have
pass-by-value semantics when used with Numba CUDA device functions.

Numba supports record types, but they have pass-by-reference semantics:

.. code-block:: python

    from numba.np.numpy_support import from_struct_dtype
    from numba import cuda
    import numpy as np

    tp = from_struct_dtype(np.dtype([('x', 'i4'), ('y', 'i8')]))

    def foo(a: tp, b: tp):
        return a

    ptx, _ = cuda.compile(foo, (tp, tp))
    print(ptx)

Inspecting the PTX, we see that the structs are passed by reference to the device function
`foo`.

.. code-block:: ptx

    .visible .func  (.param .b64 func_retval0) foo(
            .param .b64 foo_param_0,
            .param .b64 foo_param_1
    )

With gpu_struct, we can create a struct type with pass-by-value semantics:

.. code-block:: python

    from numba.core.extending import as_numba_type
    from cuda.compute import gpu_struct

    S = gpu_struct({'x': np.int32, 'y': np.int64})
    tp = as_numba_type(S)

    def foo(a: tp, b: tp):
        return a

    ptx, _ = cuda.compile(foo, (tp, tp))
    print(ptx)

Inspecting the PTX, we see that the structs are passed by value to the device function:

.. code-block:: ptx

    .visible .func  (.param .align 8 .b8 func_retval0[16]) foo(
        .param .b32 foo_param_0,
        .param .b64 foo_param_1,
        .param .b32 foo_param_2,
        .param .b64 foo_param_3
)
"""

import functools
import operator
from types import new_class
from typing import Any, Union

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


def gpu_struct(field_dict: Union[dict, np.dtype, type], name: str = "AnonymousStruct"):
    """
    A factory for creating struct types with pass-by-value semantics in Numba.

    Args:
        field_dict
            A dictionary, numpy dtype, or annotated class providing the
            mapping of field names to data types.

        name
            The name of the struct type that will be returned.

    Returns:
        A Python class that has been registered with Numba as a struct type.
        ``as_numba_type()`` can be used to get the underlying Numba type.
        Instances of this class can be passed as arguments to device functions.

    Examples:

    Construction from a dictionary.

    .. code-block:: python

        S = gpu_struct({'x': np.int32, 'y': np.int64})

    Construction from a numpy dtype.

    .. code-block:: python

        S = gpu_struct(np.dtype([('x', 'i4'), ('y', 'i8')]))

    Construction from an annotated class.

    .. code-block:: python

        @gpu_struct
        class MyStruct:
            x: np.int32
            y: np.int64

    Nesting gpu_structs.

    .. code-block:: python

        @gpu_struct
        class MyStruct:
            x: np.int32
            y: np.int64

        @gpu_struct
        class MyNestedStruct:
            a: MyStruct
            b: MyStruct

    Compiling a device function with gpu_struct arguments.

    .. code-block:: python

        def foo(a: MyStruct, b: MyStruct):
            return a

        nb_type = as_numba_type(MyStruct)
        ptx, _ = cuda.compile(foo, (nb_type, nb_type))
        print(ptx)
    """
    # Handle numpy dtype input
    if isinstance(field_dict, np.dtype):
        return _from_numpy_record_dtype(field_dict)

    # Handle annotated class (decorator usage)
    if isinstance(field_dict, type) and hasattr(field_dict, "__annotations__"):
        name = field_dict.__name__
        field_dict = field_dict.__annotations__

    # At this point, field_dict must be a dict
    assert isinstance(field_dict, dict)

    # Convert all fields to numba types
    field_spec = {key: _convert_field_type(val) for key, val in field_dict.items()}

    # Create base struct
    struct_class = make_struct_type(
        name,
        tuple(field_spec.keys()),
        tuple(field_spec.values()),
    )

    # Add Python-side features to make this usable as a regular Python type
    _patch_struct_class(struct_class)

    return struct_class


class _Struct:
    """Internal base class for all gpu_structs."""

    _fields: dict[str, Any]


@functools.lru_cache(maxsize=256)
def make_struct_type(name, field_names, field_types):
    """
    Core factory function that uses the Numba extension machinery to
    create a struct type with pass-by-value semantics.

    - Creates a Python type (and corresponding Numba type) representing a struct with the given
      field names and types.
    - Implements typing and lowering for operations like construction, indexing, and attribute
      access within device functions.
    - Returns the Python type. The corresponding Numba type can be obtained using
      ``as_numba_type()``.

    Note, the return Python type does not have any useful methods or attributes,
    other than the private class attribute ``_field_spec`` containing the  mapping
    of field names to field types. Any Python-side functionality can be implemented
    by patching in methods.

    Args:
        name
            The name of the struct type that will be returned.
        field_names
            A tuple of field names.
        field_types
            A tuple of field types.

    Returns:
        Python type that can be used to construct struct instances. The corresponding
        Numba type can be obtained using ``as_numba_type()``.
    """
    struct_class = new_class(name, bases=(_Struct,))
    raw_field_spec = dict(zip(field_names, field_types))
    assert all(
        _is_struct_type(tp) or isinstance(tp, types.Type)
        for tp in raw_field_spec.values()
    )

    # Convert struct classes to Numba types for internal use
    field_spec = {
        name: as_numba_type(typ) if _is_struct_type(typ) else typ
        for name, typ in raw_field_spec.items()
    }

    # Store the original field_spec with struct classes for user access
    struct_class._field_spec = raw_field_spec

    # The internal Numba type for this struct
    class StructType(numba.types.Type):
        def __init__(self):
            super().__init__(name=struct_class.__name__)

        def can_convert_from(self, typingctx, other):
            """Allow implicit conversion from tuples to struct."""

            # Check if tuple has the right number of elements
            if isinstance(other, types.UniTuple):
                tuple_size = other.count
                # Check if all elements can convert to corresponding fields
                if tuple_size == len(field_types):
                    return Conversion.safe

            elif isinstance(other, types.Tuple):
                tuple_size = len(other.types)
                if tuple_size == len(field_types):
                    # Check if each element can convert to its field
                    all_compatible = all(
                        typingctx.can_convert(src_type, tgt_type) is not None
                        for src_type, tgt_type in zip(other.types, field_spec.values())
                    )
                    if all_compatible:
                        return Conversion.safe

            return None

    # The Numba type system typically works with instances of Type
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

        def resolver(self, struct):
            return field_numba_type

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
        if isinstance(idx, (types.IntegerLiteral)):
            idx_val = getattr(idx, "literal_value", getattr(idx, "value", None))

            if idx_val is None or not (0 <= idx_val < len(field_names_list)):

                def error_impl(struct_val, idx):
                    raise IndexError(
                        f"Index out of range for struct with {len(field_names_list)} fields"
                    )

                return error_impl

            # exec required: Numba needs compile-time field name for type inference
            field_name = field_names_list[idx_val]
            exec(
                f"def impl(struct_val, idx): return struct_val.{field_name}",
                namespace := {},
            )
            return namespace["impl"]

        # Runtime index
        # exec required: Numba needs explicit field access per branch for type inference
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

        # Determine tuple element types and size
        if isinstance(fromty, types.UniTuple):
            tuple_size = fromty.count
            element_types = [fromty.dtype] * tuple_size
        elif isinstance(fromty, types.Tuple):
            tuple_size = len(fromty.types)
            element_types = list(fromty.types)
        else:
            # For other tuple types, try to extract the count
            tuple_size = len(field_spec)
            element_types = list(field_spec.values())

        if tuple_size != len(field_spec):
            raise ValueError(
                f"Cannot cast tuple of size {tuple_size} to {struct_class.__name__} "
                f"with {len(field_types)} fields"
            )

        # Create the struct
        retval = cgutils.create_struct_proxy(toty)(context, builder)

        # Extract each tuple element and assign to struct field
        for i, (field_name, target_type) in enumerate(field_spec.items()):
            element = builder.extract_value(val, i)

            # Cast element to target field type if necessary
            source_type = element_types[i]
            if source_type != target_type:
                element = context.cast(builder, element, source_type, target_type)

            setattr(retval, field_name, element)

        return retval._getvalue()

    @lower_cast(types.Tuple, StructType)
    @lower_cast(types.UniTuple, StructType)
    def cast_tuple_to_struct(context, builder, fromty, toty, val):
        """Lower implicit cast from tuple to struct."""
        # Check tuple size matches struct fields
        if isinstance(fromty, types.UniTuple):
            if fromty.count != len(field_spec):
                return None
            tuple_types = [fromty.dtype] * fromty.count
        else:
            if len(fromty.types) != len(field_spec):
                return None
            tuple_types = list(fromty.types)

        # Build struct from tuple elements, casting each to the correct field type
        struct_val = cgutils.create_struct_proxy(toty)(context, builder)
        for i, (field_name, field_type) in enumerate(field_spec.items()):
            elem = builder.extract_value(val, i)
            # Cast tuple element to field type
            elem = context.cast(builder, elem, tuple_types[i], field_type)
            setattr(struct_val, field_name, elem)

        return struct_val._getvalue()

    return struct_class


def _patch_struct_class(struct_class):
    """Add the required Python-side attributes to the struct class after creation."""

    def _fields_from_args(self, *args, **kwargs):
        # A help for __init__ that normalizes the user-provided *args, **kwargs
        # into a dictionary of fields

        field_spec = self.__class__._field_spec

        if args and isinstance(args[0], dict):
            fields = args[0]
        elif args:
            assert len(args) == len(field_spec), (
                f"Expected {len(field_spec)} arguments, got {len(args)}"
            )
            fields = dict(zip(field_spec.keys(), args))
        else:
            fields = kwargs

        assert fields.keys() == field_spec.keys()

        # Convert values to the correct field types
        fields = {
            name: _coerce_value(field_spec[name], fields[name]) for name in field_spec
        }
        return fields

    def __init__(self, *args, **kwargs):
        """Supporting construction from positional, keyword, and dict arguments."""

        self._fields = _fields_from_args(self, *args, **kwargs)

        for name, value in self._fields.items():
            setattr(self, name, value)

        # NumPy array representation:
        self._data = np.asarray(_as_numpy_record_value(self))
        self.__array_interface__ = self._data.__array_interface__

    struct_class.__init__ = __init__
    struct_class.dtype = _as_numpy_record_dtype(struct_class)


def _from_numpy_record_dtype(dtype: np.dtype) -> Union[_Struct, np.dtype]:
    """Convert a numpy record dtype to a gpu_struct type."""
    if dtype.type != np.void:
        return dtype
    if dtype.fields is None:
        return dtype
    return gpu_struct(
        {
            name: _from_numpy_record_dtype(field_info[0])
            for name, field_info in dtype.fields.items()
        }
    )


def _as_numpy_record_dtype(typ):
    """Convert a gpu_struct *type* to a numpy record dtype."""
    return (
        np.dtype(
            [(k, _as_numpy_record_dtype(v)) for k, v in typ._field_spec.items()],
            align=True,
        )
        if _is_struct_type(typ)
        else numba.np.numpy_support.as_dtype(typ)
    )


def _as_numpy_record_value(val) -> np.void:
    """Convert a gpu_struct *value* to a numpy record."""

    def _fields_to_tuples(fields_dict: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(
            _fields_to_tuples(v._fields) if isinstance(v, _Struct) else v
            for v in fields_dict.values()
        )

    return np.void(
        _fields_to_tuples(val._fields), dtype=_as_numpy_record_dtype(type(val))
    )


def _coerce_value(field_type, value: Any) -> Any:
    if isinstance(value, _Struct):
        return value
    if isinstance(value, tuple):
        return field_type(*value)
    if isinstance(value, dict):
        return field_type(**value)
    return field_type(value)


def _convert_field_type(val):
    """Convert a field value to a numba-compatible type."""
    if isinstance(val, dict):
        return gpu_struct(val)
    if _is_struct_type(val):
        return val
    return numba.np.numpy_support.from_dtype(val)


def _is_struct_type(typ: Any) -> bool:
    """Check if a type is a GPU struct class."""
    return isinstance(typ, type) and issubclass(typ, _Struct)
