# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass, make_dataclass
from dataclasses import fields as dataclass_fields
from typing import Type

import numba
import numpy as np
from numba.core import cgutils
from numba.core.extending import (
    make_attribute_wrapper,
    models,
    register_model,
    typeof_impl,
)
from numba.core.typing import signature as nb_signature
from numba.core.typing.templates import AttributeTemplate, ConcreteTemplate
from numba.cuda.cudadecl import registry as cuda_registry
from numba.extending import as_numba_type, lower_builtin

from .typing import GpuStruct


def gpu_struct(this: type) -> Type[GpuStruct]:
    """
    Defines the given class as being a GpuStruct.

    A GpuStruct represents a value composed of one or more other
    values, and is defined as a class with annotated fields (similar
    to a dataclass). The type of each field must be a subclass of
    `np.number`, like `np.int32` or `np.float64`.

    Arrays of GPUStruct objects can be used as inputs to cuda.parallel
    algorithms.

    Example:
        The code snippet below shows how to use `gpu_struct` to define
        a `MinMax` type (composed of `min_val`, `max_val` values), and perform
        a reduction on an input array of floating point values to compute its
        the smallest and the largest absolute values:

        .. literalinclude:: ../../python/cuda_parallel/tests/test_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin reduce-minmax
            :end-before: example-end reduce-minmax
    """
    # Implementation-wise, @gpu_struct creates and registers a
    # corresponding numba type to the given type, so that it can be
    # used within device functions (e.g., unary and binary operations)
    # The numba typing/lowering code is largely based on the example
    # in # https://github.com/gmarkall/numba-accelerated-udfs/blob/e78876c5d3ace9e1409d37029bd79b2a8b706c62/filigree/numba_extension.py

    anns = getattr(this, "__annotations__", {})

    # Set a .dtype attribute on the class that returns the
    # corresponding numpy structure dtype. This makes it convenient to
    # create CuPy/NumPy arrays of this type.
    setattr(this, "dtype", np.dtype(list(anns.items()), align=True))

    # Define __post_init__ to create a numpy struct from the fields,
    # and keep a reference to it in the `._data` attribute. The data
    # underlying this array is what is ultimately passed to the C
    # library, and we need to keep a reference to it for the lifetime
    # of the object.
    def __post_init__(self):
        self._data = np.array(
            [tuple(getattr(self, name) for name in anns)], dtype=self.dtype
        )

    def __array_interface__(self):
        return self._data.__array_interface__

    setattr(this, "__post_init__", __post_init__)
    setattr(this, "__array_interface__", property(__array_interface__))

    # Wrap `this` in a dataclass for convenience:
    this = dataclass(this)
    fields = dataclass_fields(this)

    # Define a numba type corresponding to `this`:
    class ThisType(numba.types.Type):
        def __init__(self):
            super().__init__(name=this.__name__)

    this_type = ThisType()

    as_numba_type.register(this, this_type)

    @typeof_impl.register(this)
    def typeof_this(val, c):
        return ThisType()

    # Data model corresponding to ThisType:
    @register_model(ThisType)
    class ThisModel(models.StructModel):
        def __init__(self, dmm, fe_type):
            members = [(field.name, numba.from_dtype(field.type)) for field in fields]
            super().__init__(dmm, fe_type, members)

    # Typing for accessing attributes (fields) of the dataclass:
    class ThisAttrsTemplate(AttributeTemplate):
        pass

    for field in fields:
        typ = field.type
        name = field.name

        def resolver(self, this):
            return numba.from_dtype(typ)

        setattr(ThisAttrsTemplate, f"resolve_{name}", resolver)

    # Lowering for attribute access:
    for field in fields:
        make_attribute_wrapper(ThisType, field.name, field.name)

    # Typing for constructor.
    @cuda_registry.register
    class ThisConstructor(ConcreteTemplate):
        key = this
        cases = [
            nb_signature(this_type, *[numba.from_dtype(field.type) for field in fields])
        ]

    cuda_registry.register_global(this, numba.types.Function(ThisConstructor))

    # Lowering for constructor:
    def this_constructor(context, builder, sig, args):
        ty = sig.return_type
        retval = cgutils.create_struct_proxy(ty)(context, builder)
        for field, val in zip(fields, args):
            setattr(retval, field.name, val)
        return retval._getvalue()

    lower_builtin(this, *[numba.from_dtype(field.type) for field in fields])(
        this_constructor
    )

    return this


def gpu_struct_from_numpy_dtype(name, np_dtype):
    ret = make_dataclass(
        name, [(name, dtype) for name, (dtype, _) in np_dtype.fields.items()]
    )
    return gpu_struct(ret)
