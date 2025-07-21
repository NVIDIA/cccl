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


def gpu_dataclass(dc: Any) -> Any:
    fields = dataclasses.fields(dc)
    names = [f.name for f in fields]
    objs = [getattr(dc, name) for name in names]

    members = [(name, numba.typeof(obj)) for (name, obj) in zip(names, objs)]

    from ._types import BasePrimitive

    primitives = {
        name: obj for (name, obj) in zip(names, objs) if isinstance(obj, BasePrimitive)
    }
    if primitives:
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

        def resolver(self, this):
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

        # The values we return here just need to pacify _Kernel's
        # _parse_args() routines--we never actually use the kernel
        # parameters by way of the arguments provided at kernel launch.
        addr = id(val)
        return (numba.types.uint64, addr)

    setattr(dc, "prepare_args", prepare_args)

    return dc
