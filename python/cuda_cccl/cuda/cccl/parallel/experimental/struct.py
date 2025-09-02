# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
import operator
from dataclasses import make_dataclass
from typing import Any, Dict, Type

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

    # Generate indexing methods dynamically for Python execution
    # Generate __getitem__ method
    getitem_lines = []
    for i, field_name in enumerate(field_names):
        if i == 0:
            getitem_lines.append(f"    if idx == {i}:")
        else:
            getitem_lines.append(f"    elif idx == {i}:")
        getitem_lines.append(f"        return self.{field_name}")

    getitem_lines.append("    else:")
    getitem_lines.append(
        f"        raise IndexError(f'Index {{idx}} out of range for struct with {len(field_names)} fields')"
    )

    getitem_code = f"""
def __getitem__(self, idx):
    '''Get field by index.'''
{chr(10).join(getitem_lines)}
"""

    # Generate __len__ method
    len_code = f"""
def __len__(self):
    '''Return the number of fields.'''
    return {len(field_names)}
"""

    # Execute the generated code and add methods to the class
    namespace: Dict[str, Any] = {}
    exec(getitem_code, namespace)
    exec(len_code, namespace)

    StructClass.__getitem__ = namespace["__getitem__"]  # type: ignore
    StructClass.__len__ = namespace["__len__"]  # type: ignore

    # Set up the class for use with numba
    _setup_numba_struct(StructClass, field_dict)

    return StructClass


def gpu_struct(this: type) -> Type[GpuStruct]:
    """
    Decorate a class as a GPU struct.

    A GpuStruct represents a value composed of one or more other
    values, and is defined as a class with annotated fields (similar
    to a dataclass). The type of each field must be a subclass of
    `np.number`, like `np.int32` or `np.float64`.

    Arrays of GPUStruct objects can be used as inputs to cuda.cccl.parallel
    algorithms.

    Example:
        The code snippet below shows how to use `gpu_struct` to define
        a `MinMax` type (composed of `min_val`, `max_val` values), and perform
        a reduction on an input array of floating point values to compute its
        the smallest and the largest absolute values:

        .. literalinclude:: ../../python/cuda_cccl/tests/parallel/test_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin reduce-minmax
            :end-before: example-end reduce-minmax
    """
    anns = getattr(this, "__annotations__", {})

    # Convert numpy types to numba types and preserve field names
    field_names = tuple(anns.keys())
    field_types = tuple(numba.from_dtype(typ) for typ in anns.values())

    # Create the struct using gpu_struct_from_numba_types
    StructClass = gpu_struct_from_numba_types(this.__name__, field_names, field_types)

    # Set a .dtype attribute on the class that returns the
    # corresponding numpy structure dtype. This makes it convenient to
    # create CuPy/NumPy arrays of this type.
    setattr(StructClass, "dtype", np.dtype(list(anns.items()), align=True))

    # Store the original __init__ method
    original_init = StructClass.__init__

    # Define __post_init__ to create a numpy struct from the fields,
    # and keep a reference to it in the `._data` attribute. The data
    # underlying this array is what is ultimately passed to the C
    # library, and we need to keep a reference to it for the lifetime
    # of the object.
    def __post_init__(self):
        self._data = np.array(
            [tuple(getattr(self, name) for name in anns.keys())], dtype=self.dtype
        )

    def __array_interface__(self):
        return self._data.__array_interface__

    # Create a new __init__ that calls the original and then __post_init__
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        __post_init__(self)

    setattr(StructClass, "__init__", new_init)
    setattr(StructClass, "__array_interface__", property(__array_interface__))

    return StructClass


def gpu_struct_from_numpy_dtype(name, np_dtype):
    """
    Create a GPU struct from a numpy record dtype.
    """
    ret = make_dataclass(
        name, [(name, dtype) for name, (dtype, _) in np_dtype.fields.items()]
    )
    return gpu_struct(ret)
