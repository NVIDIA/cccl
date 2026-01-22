# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba

from .._common import normalize_dtype_param
from .._types import (
    Algorithm,
    BasePrimitive,
    Constant,
    Dependency,
    DependentArray,
    DependentPythonOperator,
    Invocable,
    TemplateParameter,
    numba_type_to_wrapper,
)


class merge_sort_keys(BasePrimitive):
    is_one_shot = True

    def __init__(
        self,
        dtype,
        items_per_thread,
        compare_op,
        threads_in_warp=32,
        methods=None,
        unique_id=None,
        temp_storage=None,
        node=None,
    ):
        """Performs a warp-wide merge sort over blocked keys."""
        self.node = node
        self.temp_storage = temp_storage
        self.dtype = normalize_dtype_param(dtype)
        self.items_per_thread = items_per_thread
        self.compare_op = compare_op
        self.threads_in_warp = threads_in_warp
        self.methods = methods

        template = Algorithm(
            "WarpMergeSort",
            "Sort",
            "warp_merge_sort",
            ["cub/warp/warp_merge_sort.cuh"],
            [
                TemplateParameter("KeyT"),
                TemplateParameter("ITEMS_PER_THREAD"),
                TemplateParameter("VIRTUAL_WARP_THREADS"),
            ],
            [
                [
                    DependentArray(Dependency("KeyT"), Dependency("ITEMS_PER_THREAD")),
                    DependentPythonOperator(
                        Constant(numba.int8),
                        [Dependency("KeyT"), Dependency("KeyT")],
                        Dependency("Op"),
                        name="compare_op",
                    ),
                ]
            ],
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
                "KeyT": self.dtype,
                "VIRTUAL_WARP_THREADS": threads_in_warp,
                "ITEMS_PER_THREAD": items_per_thread,
                "Op": compare_op,
            }
        )

    @classmethod
    def create(
        cls,
        dtype,
        items_per_thread,
        compare_op,
        threads_in_warp=32,
        methods=None,
    ):
        algo = cls(
            dtype=dtype,
            items_per_thread=items_per_thread,
            compare_op=compare_op,
            threads_in_warp=threads_in_warp,
            methods=methods,
        )
        specialization = algo.specialization
        return Invocable(
            ltoir_files=specialization.get_lto_ir(threads=threads_in_warp),
            temp_storage_bytes=specialization.temp_storage_bytes,
            temp_storage_alignment=specialization.temp_storage_alignment,
            algorithm=specialization,
        )
