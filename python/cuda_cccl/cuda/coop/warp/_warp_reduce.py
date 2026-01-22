# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba

from .._common import normalize_dtype_param
from .._types import (
    Algorithm,
    BasePrimitive,
    Dependency,
    DependentPythonOperator,
    DependentReference,
    Invocable,
    TemplateParameter,
    Value,
    numba_type_to_wrapper,
)


class reduce(BasePrimitive):
    is_one_shot = True

    def __init__(
        self,
        dtype,
        binary_op,
        threads_in_warp=32,
        valid_items=None,
        methods=None,
        unique_id=None,
        temp_storage=None,
        node=None,
    ):
        """Computes a warp-wide reduction for lane :sub:`0` using the specified binary reduction functor."""
        self.node = node
        self.temp_storage = temp_storage
        self.dtype = normalize_dtype_param(dtype)
        self.binary_op = binary_op
        self.threads_in_warp = threads_in_warp
        self.valid_items = valid_items
        self.methods = methods

        parameters = [
            [
                DependentReference(Dependency("T")),
                DependentPythonOperator(
                    Dependency("T"),
                    [Dependency("T"), Dependency("T")],
                    Dependency("Op"),
                    name="binary_op",
                ),
                DependentReference(Dependency("T"), True),
            ]
        ]
        if valid_items is not None:
            parameters[0].append(Value(numba.types.int32, name="valid_items"))

        template = Algorithm(
            "WarpReduce",
            "Reduce",
            "warp_reduce",
            ["cub/warp/warp_reduce.cuh"],
            [TemplateParameter("T"), TemplateParameter("VIRTUAL_WARP_THREADS")],
            parameters,
            self,
            type_definitions=[numba_type_to_wrapper(self.dtype, methods=methods)]
            if methods is not None
            else None,
            threads=threads_in_warp,
            unique_id=unique_id,
        )
        self.algorithm = template
        self.specialization = template.specialize(
            {
                "T": self.dtype,
                "VIRTUAL_WARP_THREADS": threads_in_warp,
                "Op": binary_op,
            }
        )

    @classmethod
    def create(
        cls, dtype, binary_op, threads_in_warp=32, valid_items=None, methods=None
    ):
        algo = cls(
            dtype=dtype,
            binary_op=binary_op,
            threads_in_warp=threads_in_warp,
            valid_items=valid_items,
            methods=methods,
        )
        specialization = algo.specialization
        return Invocable(
            ltoir_files=specialization.get_lto_ir(threads=threads_in_warp),
            temp_storage_bytes=specialization.temp_storage_bytes,
            temp_storage_alignment=specialization.temp_storage_alignment,
            algorithm=specialization,
        )


class sum(BasePrimitive):
    is_one_shot = True

    def __init__(
        self,
        dtype,
        threads_in_warp=32,
        valid_items=None,
        unique_id=None,
        temp_storage=None,
    ):
        """Computes a warp-wide reduction for lane :sub:`0` using addition (+)."""
        self.temp_storage = temp_storage
        self.dtype = normalize_dtype_param(dtype)
        self.binary_op = None
        self.threads_in_warp = threads_in_warp
        self.valid_items = valid_items

        parameters = [
            [
                DependentReference(Dependency("T")),
                DependentReference(Dependency("T"), True),
            ]
        ]
        if valid_items is not None:
            parameters[0].append(Value(numba.types.int32, name="valid_items"))

        template = Algorithm(
            "WarpReduce",
            "Sum",
            "warp_reduce",
            ["cub/warp/warp_reduce.cuh"],
            [TemplateParameter("T"), TemplateParameter("VIRTUAL_WARP_THREADS")],
            parameters,
            self,
            threads=threads_in_warp,
            unique_id=unique_id,
        )
        self.algorithm = template
        self.specialization = template.specialize(
            {"T": self.dtype, "VIRTUAL_WARP_THREADS": threads_in_warp}
        )

    @classmethod
    def create(cls, dtype, threads_in_warp=32, valid_items=None):
        algo = cls(
            dtype=dtype,
            threads_in_warp=threads_in_warp,
            valid_items=valid_items,
        )
        specialization = algo.specialization
        return Invocable(
            ltoir_files=specialization.get_lto_ir(threads=threads_in_warp),
            temp_storage_bytes=specialization.temp_storage_bytes,
            temp_storage_alignment=specialization.temp_storage_alignment,
            algorithm=specialization,
        )
