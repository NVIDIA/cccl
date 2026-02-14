# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba
import numpy as np
from numba import cuda, types
from numba.core import cgutils
from numba.core.extending import (
    lower_builtin,
    make_attribute_wrapper,
    models,
    register_model,
    type_callable,
    typeof_impl,
)
from numba.core.imputils import lower_constant

NUMBA_TYPES_TO_NP = {
    types.int8: np.int8,
    types.int16: np.int16,
    types.int32: np.int32,
    types.int64: np.int64,
    types.uint8: np.uint8,
    types.uint16: np.uint16,
    types.uint32: np.uint32,
    types.uint64: np.uint64,
    types.float32: np.float32,
    types.float64: np.float64,
}


def random_int(shape, dtype):
    return np.random.randint(0, 128, size=shape).astype(dtype)


@cuda.jit(device=True)
def row_major_tid():
    dim = cuda.blockDim
    idx = cuda.threadIdx
    return (
        (0 if dim.z == 1 else idx.z * dim.x * dim.y)
        + (0 if dim.y == 1 else idx.y * dim.x)
        + idx.x
    )


class Complex:
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    def construct(this):
        default_value = numba.int32(0)
        this[0] = Complex(default_value, default_value)

    def assign(this, that):
        this[0] = Complex(that[0].real, that[0].imag)


class ComplexType(types.Type):
    def __init__(self):
        super().__init__(name="Complex")


complex_type = ComplexType()


@typeof_impl.register(Complex)
def typeof_complex(val, c):
    return complex_type


@type_callable(Complex)
def type__complex(context):
    def typer(real, imag):
        if isinstance(real, types.Integer) and isinstance(imag, types.Integer):
            return complex_type

    return typer


@register_model(ComplexType)
class ComplexModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("real", types.int32), ("imag", types.int32)]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(ComplexType, "real", "real")
make_attribute_wrapper(ComplexType, "imag", "imag")


@lower_builtin(Complex, types.Integer, types.Integer)
def impl_complex(context, builder, sig, args):
    typ = sig.return_type
    real, imag = args
    state = cgutils.create_struct_proxy(typ)(context, builder)
    state.real = real
    state.imag = imag
    return state._getvalue()


@lower_constant(ComplexType)
def lower_constant_complex(context, builder, typ, value):
    state = cgutils.create_struct_proxy(typ)(context, builder)
    state.real = context.get_constant(types.int32, value.real)
    state.imag = context.get_constant(types.int32, value.imag)
    return state._getvalue()
