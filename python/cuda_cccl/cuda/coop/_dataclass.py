# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import dataclasses
from typing import Any

import numba
from numba.core.extending import (
    make_attribute_wrapper,
    models,
    register_model,
    typeof_impl,
)
from numba.core.typing.templates import AttributeTemplate
from numba.cuda.cudadecl import registry as cuda_registry


def gpu_dataclass(dc: Any, *, compute_temp_storage: bool = True) -> Any:
    fields = dataclasses.fields(dc)
    names = [f.name for f in fields]
    objs = [getattr(dc, name) for name in names]

    members = [(name, numba.typeof(obj)) for (name, obj) in zip(names, objs)]

    from ._types import BasePrimitive

    primitives = {
        name: obj for (name, obj) in zip(names, objs) if isinstance(obj, BasePrimitive)
    }
    if compute_temp_storage and primitives:
        temp_storage_bytes_sum = sum(
            obj.temp_storage_bytes for obj in primitives.values()
        )
        temp_storage_bytes_max = max(
            obj.temp_storage_bytes for obj in primitives.values()
        )
        temp_storage_alignment = max(
            obj.temp_storage_alignment for obj in primitives.values()
        )
        setattr(dc, "temp_storage_bytes_sum", temp_storage_bytes_sum)
        setattr(dc, "temp_storage_bytes_max", temp_storage_bytes_max)
        setattr(dc, "temp_storage_alignment", temp_storage_alignment)
    else:
        temp_storage_bytes_sum = 0
        temp_storage_bytes_max = 0
        temp_storage_alignment = 0

    class GpuDataClassType(numba.types.Type):
        def __init__(self):
            super().__init__(name="GpuDataClass")

    gpu_dataclass_type = GpuDataClassType()

    @typeof_impl.register(dc.__class__)
    def typeof_gpu_dataclass(val, c):
        return gpu_dataclass_type

    @register_model(GpuDataClassType)
    class GpuDataClassModel(models.StructModel):
        def __init__(self, dmm, fe_type):
            super().__init__(dmm, fe_type, members)

    class GpuDataClassAttrsTemplate(AttributeTemplate):
        key = gpu_dataclass_type

    for name, type_obj in members:

        def resolver(self, this, type_obj=type_obj):
            return type_obj

        setattr(GpuDataClassAttrsTemplate, f"resolve_{name}", resolver)

        make_attribute_wrapper(GpuDataClassType, name, name)

    cuda_registry.register_attr(GpuDataClassAttrsTemplate)

    def pre_launch_callback(kernel, launch_config):
        from ._rewrite import register_kernel_extension

        register_kernel_extension(kernel, dc)

    setattr(dc, "pre_launch_callback", pre_launch_callback)

    def prepare_args(ty, val, *args, **kwds):
        if val is not dc:
            return (ty, val)

        def coerce_field(field_val: Any):
            if isinstance(field_val, BasePrimitive):
                # Primitives are compile-time only; use a dummy pointer-sized
                # value to satisfy argument packing.
                return numba.types.uintp, 0
            return numba.typeof(field_val), field_val

        field_types = []
        field_vals = []
        for name in names:
            field_ty, field_val = coerce_field(getattr(val, name))
            field_types.append(field_ty)
            field_vals.append(field_val)

        # Flatten the struct into a tuple so that _Kernel._prepare_args can
        # marshal the argument according to the backend ABI.
        tuple_ty = numba.types.Tuple(tuple(field_types))
        return (tuple_ty, tuple(field_vals))

    setattr(dc, "prepare_args", prepare_args)
    setattr(dc, "__cuda_coop_gpu_dataclass__", True)

    return dc
