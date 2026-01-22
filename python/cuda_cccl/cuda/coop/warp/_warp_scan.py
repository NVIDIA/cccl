# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from .._common import normalize_dtype_param
from .._types import (
    Algorithm,
    BasePrimitive,
    Dependency,
    DependentReference,
    Invocable,
    TemplateParameter,
)


class exclusive_sum(BasePrimitive):
    is_one_shot = True

    def __init__(self, dtype, threads_in_warp=32, unique_id=None, temp_storage=None):
        """Computes an exclusive warp-wide prefix sum using addition (+)."""
        self.temp_storage = temp_storage
        dtype = normalize_dtype_param(dtype)

        template = Algorithm(
            "WarpScan",
            "ExclusiveSum",
            "warp_scan",
            ["cub/warp/warp_scan.cuh"],
            [TemplateParameter("T"), TemplateParameter("VIRTUAL_WARP_THREADS")],
            [
                [
                    DependentReference(Dependency("T")),
                    DependentReference(Dependency("T"), True),
                ]
            ],
            self,
            fake_return=True,
            threads=threads_in_warp,
            unique_id=unique_id,
        )
        self.algorithm = template
        self.specialization = template.specialize(
            {"T": dtype, "VIRTUAL_WARP_THREADS": threads_in_warp}
        )

    @classmethod
    def create(cls, dtype, threads_in_warp=32):
        algo = cls(dtype=dtype, threads_in_warp=threads_in_warp)
        specialization = algo.specialization
        return Invocable(
            ltoir_files=specialization.get_lto_ir(),
            temp_storage_bytes=specialization.temp_storage_bytes,
            temp_storage_alignment=specialization.temp_storage_alignment,
            algorithm=specialization,
        )
