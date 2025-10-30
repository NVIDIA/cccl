# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
import operator
import warnings
from typing import Dict, Type

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
from numba.core.typing import signature as nb_signature
from numba.core.typing.templates import AttributeTemplate, ConcreteTemplate
from numba.cuda.cudadecl import registry as cuda_registry
from numba.extending import as_numba_type, lower_builtin

from .typing import GpuStruct


def _setup_numba_struct(struct_class: Type, field_types: Dict[str, types.Type]):
    """Set up a class to work as a numba struct type."""

    # Define a numba type corresponding to the struct class
    class StructType(numba.types.Type):
        def __init__(self):
            super().__init__(name=struct_class.__name__)

    struct_numba_type = StructType()

    # Register the numba type
    as_numba_type.register(struct_class, struct_numba_type)

    @typeof_impl.register(struct_class)
    def typeof_struct(val, c):
        return StructType()

    # Data model corresponding to StructType
    @register_model(StructType)
    class StructModel(models.StructModel):
        def __init__(self, dmm, fe_type):
            # Use the original numba types for the struct model
            members = [
                (field_name, field_type)
                for field_name, field_type in field_types.items()
            ]
            super().__init__(dmm, fe_type, members)

    # Typing for accessing attributes (fields) of the struct
    class StructAttrsTemplate(AttributeTemplate):
        pass

    for field_name, field_numba_type in field_types.items():

        def resolver(self, struct):
            return field_numba_type

        setattr(StructAttrsTemplate, f"resolve_{field_name}", resolver)

    # Lowering for attribute access
    for field_name in field_types:
        make_attribute_wrapper(StructType, field_name, field_name)

    # Add indexing support for Numba compilation
    field_names_list = list(field_types.keys())

    @overload(operator.getitem)
    def struct_getitem(struct_val, idx):
        if not isinstance(struct_val, StructType):
            return

        # Check if idx is a literal (compile-time constant)
        if isinstance(idx, (types.IntegerLiteral, types.Literal)):
            if hasattr(idx, "literal_value"):
                idx_val = idx.literal_value
            else:
                # For types.Literal
                idx_val = idx.value if hasattr(idx, "value") else None

            if idx_val is not None and 0 <= idx_val < len(field_names_list):
                field_name = field_names_list[idx_val]

                # Return a function that accesses the specific field
                # Use exec to create a function with the field name baked in
                impl_code = f"""
def struct_getitem_literal_impl(struct_val, idx):
    return struct_val.{field_name}
"""
                local_vars = {}
                exec(impl_code, {}, local_vars)
                return local_vars["struct_getitem_literal_impl"]
            elif idx_val is not None:

                def struct_getitem_error_impl(struct_val, idx):
                    raise IndexError(
                        f"Index {idx_val} out of range for struct with {len(field_names_list)} fields"
                    )

                return struct_getitem_error_impl

        # Fallback for non-literal indices (runtime indexing)
        # Build the if-elif chain dynamically
        impl_lines = []
        for i, field_name in enumerate(field_names_list):
            if i == 0:
                impl_lines.append(f"    if idx == {i}:")
            else:
                impl_lines.append(f"    elif idx == {i}:")
            impl_lines.append(f"        return struct_val.{field_name}")

        # Add the error case
        impl_lines.append("    else:")
        impl_lines.append("        raise IndexError('Index out of range')")

        impl_code = f"""
def struct_getitem_impl(struct_val, idx):
{chr(10).join(impl_lines)}
"""
        # Execute the generated code to create the implementation
        local_vars = {}
        exec(impl_code, {}, local_vars)
        return local_vars["struct_getitem_impl"]

    # Typing for constructor
    @cuda_registry.register
    class StructConstructor(ConcreteTemplate):
        key = struct_class
        cases = [nb_signature(struct_numba_type, *list(field_types.values()))]

    cuda_registry.register_global(struct_class, numba.types.Function(StructConstructor))

    # Lowering for constructor
    def struct_constructor(context, builder, sig, args):
        ty = sig.return_type
        retval = cgutils.create_struct_proxy(ty)(context, builder)
        for field_name, val in zip(field_types.keys(), args):
            setattr(retval, field_name, val)
        return retval._getvalue()

    lower_builtin(struct_class, *list(field_types.values()))(struct_constructor)

    # Store the struct type on the class for later use
    struct_class._numba_type = struct_numba_type

    # Store field_types on the struct class for use in overload
    struct_class._field_types = field_types


@functools.lru_cache(maxsize=None)
def gpu_struct_from_numba_types(
    name: str, field_names: tuple, field_types: tuple
) -> Type:
    """
    Create a struct type from tuples of field names and numba types.

    Args:
        name: The name of the struct class
        field_names: Tuple of field names
        field_types: Tuple of corresponding numba types

    Returns:
        A dynamically created struct class with the specified fields
    """
    if len(field_names) != len(field_types):
        raise ValueError("field_names and field_types must have the same length")

    # Create a dict mapping field names to types
    field_dict = dict(zip(field_names, field_types))

    class StructClass:
        def __init__(self, *args):
            if len(args) != len(field_types):
                raise ValueError(
                    f"Expected {len(field_types)} arguments, got {len(args)}"
                )

            for _, (field_name, arg_value) in enumerate(zip(field_names, args)):
                setattr(self, field_name, arg_value)

    StructClass.__name__ = name

    # Set up the class for use with numba
    _setup_numba_struct(StructClass, field_dict)

    return StructClass


def gpu_struct(field_dict: dict, name: str = "AnonymousStruct") -> Type[GpuStruct]:
    """
    Create a GPU struct from a dictionary mapping field names to types.

    A GpuStruct represents a value composed of one or more other
    values, defined as a dictionary mapping field names to types.

    The type of each field must be either:
    - A subclass of `np.number`, like `np.int32` or `np.float64`
    - Another GPU struct (for nested structs)
    - A dictionary defining a nested struct inline

    Arrays of GPUStruct objects can be used as inputs to cuda.compute
    algorithms.

    Args:
        field_dict: Dictionary mapping field names to types. Values can be numpy dtypes,
                   existing GPU structs, or nested dictionaries.
        name: Optional name for the struct class (default: "AnonymousStruct")

    Returns:
        A GPU struct class that can be instantiated and used in CUDA computations.

    Example:
        Basic struct definition:

        .. code-block:: python

            MinMax = gpu_struct({"min_val": np.float32, "max_val": np.float32})

        Using the struct:

        .. code-block:: python

            result = MinMax(1.0, 10.0)
            print(result.min_val)  # Access field

        Creating nested structs:

        .. code-block:: python

            # Define inner struct first
            Inner = gpu_struct({"a": np.int32, "b": np.float32})

            # Use it in outer struct
            Outer = gpu_struct({"x": np.int64, "inner": Inner})

            # Or define inline
            Outer = gpu_struct({
                "x": np.int64,
                "inner": {"a": np.int32, "b": np.float32}
            })

    """
    # Check if called as a decorator with type annotations (legacy syntax)
    if not isinstance(field_dict, dict):
        # Check if it's a class with __annotations__
        if isinstance(field_dict, type) and hasattr(field_dict, "__annotations__"):
            warnings.warn(
                "The @gpu_struct decorator syntax is deprecated and will be removed in a future release. "
                "Please use the dictionary syntax instead: MyStruct = gpu_struct({...})",
                UserWarning,
                stacklevel=2,
            )
            # Extract annotations from the class
            annotations = field_dict.__annotations__
            class_name = field_dict.__name__
            return _gpu_struct_from_dict(annotations, name=class_name)
        else:
            raise TypeError(
                "gpu_struct() requires a dictionary argument mapping field names to types. "
                "The @gpu_struct decorator syntax is deprecated. "
                "Please use: MyStruct = gpu_struct({...})"
            )

    return _gpu_struct_from_dict(field_dict, name=name)


def _gpu_struct_from_dict(
    field_dict: dict, name: str = "AnonymousStruct"
) -> Type[GpuStruct]:
    """
    Helper function to create a GPU struct from a dictionary.

    Recursively processes nested dictionaries to create nested structs.

    Args:
        field_dict: Dictionary mapping field names to types (numpy dtypes,
                   GPU structs, or nested dictionaries)
        name: Name for the struct class

    Returns:
        A GPU struct class
    """
    processed_fields = {}
    dtype_fields = []

    for field_name, field_type in field_dict.items():
        # Handle nested dictionary - recursively create a struct
        if isinstance(field_type, dict):
            nested_struct = _gpu_struct_from_dict(
                field_type, name=f"{name}_{field_name}"
            )
            processed_fields[field_name] = nested_struct
            dtype_fields.append((field_name, nested_struct.dtype))
        # Handle existing GPU struct
        elif hasattr(field_type, "_numba_type"):
            processed_fields[field_name] = field_type
            dtype_fields.append((field_name, field_type.dtype))
        # Handle numpy dtype
        else:
            processed_fields[field_name] = field_type
            dtype_fields.append((field_name, field_type))

    # Convert to numba types
    field_names = tuple(processed_fields.keys())
    field_types = []
    for typ in processed_fields.values():
        if hasattr(typ, "_numba_type"):
            field_types.append(typ._numba_type)
        else:
            field_types.append(numba.from_dtype(typ))

    # Create the struct using gpu_struct_from_numba_types
    StructClass = gpu_struct_from_numba_types(name, field_names, tuple(field_types))

    # Set the dtype
    setattr(StructClass, "dtype", np.dtype(dtype_fields, align=True))

    # Store the original __init__ method
    original_init = StructClass.__init__

    # Store processed_fields for use in __init__ and __post_init__
    processed_fields_tuple = tuple(processed_fields.items())

    def __post_init__(self):
        def extract_value(val, typ):
            if hasattr(typ, "_numba_type"):
                if hasattr(val, "_data"):
                    return val._data[0]
                else:
                    # For nested structs without _data, extract field by field
                    nested_tuple = tuple(
                        extract_value(getattr(val, field_name), field_typ)
                        for field_name, field_typ in typ.__annotations__.items()
                        if hasattr(typ, "__annotations__")
                    )
                    if not nested_tuple and hasattr(val, "__len__"):
                        # If no annotations but has __len__, try to extract as sequence
                        nested_tuple = tuple(
                            extract_value(val[i], list(processed_fields_tuple)[i][1])
                            for i in range(len(val))
                        )
                    return nested_tuple
            else:
                return val

        values = tuple(
            extract_value(getattr(self, name), typ)
            for name, typ in processed_fields_tuple
        )
        self._data = np.array([values], dtype=self.dtype)

    def __array_interface__(self):
        return self._data.__array_interface__

    # Create a new __init__ that handles nested struct construction
    def new_init(self, *args, **kwargs):
        # Handle single dictionary initialization (all fields from dict)
        if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            data_dict = args[0]
            args = tuple(
                data_dict.get(field_name) for field_name, _ in processed_fields_tuple
            )

        # Convert tuple/list/dict arguments to nested struct instances
        processed_args = []
        for i, (arg, (field_name, field_type)) in enumerate(
            zip(args, processed_fields_tuple)
        ):
            # If this field is a nested struct and arg is a tuple/list/dict, construct the struct
            if hasattr(field_type, "_numba_type"):
                if isinstance(arg, dict):
                    # Recursively construct from dict
                    processed_args.append(field_type(arg))
                elif isinstance(arg, (tuple, list)):
                    # Construct from tuple/list
                    processed_args.append(field_type(*arg))
                else:
                    # Already a struct instance or primitive
                    processed_args.append(arg)
            else:
                processed_args.append(arg)

        original_init(self, *processed_args, **kwargs)
        __post_init__(self)

    setattr(StructClass, "__init__", new_init)
    setattr(StructClass, "__array_interface__", property(__array_interface__))

    # Add overload to handle tuple construction for nested structs in device functions
    # This allows syntax like: Outer(x, (a, b)) instead of Outer(x, Inner(a, b))
    @overload(StructClass)
    def struct_constructor_with_tuples(*args):
        # Check if any arguments are tuples and need conversion to structs
        if len(args) != len(processed_fields_tuple):
            return  # Wrong number of args, let the normal path handle the error

        # Check which fields need tuple-to-struct conversion
        conversions = []
        needs_conversion = False

        for i, (arg_type, (field_name, field_struct_class)) in enumerate(
            zip(args, processed_fields_tuple)
        ):
            # Check if this argument is a tuple and the field expects a struct
            if isinstance(arg_type, (types.Tuple, types.UniTuple)) and hasattr(
                field_struct_class, "_numba_type"
            ):
                conversions.append((i, field_struct_class))
                needs_conversion = True

        if not needs_conversion:
            return  # No tuples to convert, use the standard constructor

        # Generate implementation code that converts tuples to structs
        impl_lines = ["def impl(*args):"]

        # Build the converted arguments list
        for i, (field_name, field_struct_class) in enumerate(processed_fields_tuple):
            # Check if this index needs conversion
            needs_conv = any(conv_i == i for conv_i, _ in conversions)
            if needs_conv:
                impl_lines.append(f"    arg{i} = field_struct_class_{i}(*args[{i}])")
            else:
                impl_lines.append(f"    arg{i} = args[{i}]")

        # Build the constructor call with all converted arguments
        arg_list = ", ".join(f"arg{i}" for i in range(len(processed_fields_tuple)))
        impl_lines.append(f"    return struct_class({arg_list})")

        impl_code = "\n".join(impl_lines)

        # Create namespace with necessary references
        namespace = {"struct_class": StructClass}
        for i, (field_name, field_struct_class) in enumerate(processed_fields_tuple):
            if hasattr(field_struct_class, "_numba_type"):
                namespace[f"field_struct_class_{i}"] = field_struct_class

        # Execute the generated code to create the implementation function
        exec(impl_code, namespace)
        return namespace["impl"]

    return StructClass


def gpu_struct_from_numpy_dtype(name, np_dtype):
    """
    Create a GPU struct from a numpy record dtype.
    """
    field_dict = {
        field_name: dtype for field_name, (dtype, _) in np_dtype.fields.items()
    }
    return gpu_struct(field_dict, name=name)
