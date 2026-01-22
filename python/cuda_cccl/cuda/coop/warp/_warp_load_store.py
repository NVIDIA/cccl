# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

import numba

from .._common import normalize_dtype_param
from .._enums import WarpLoadAlgorithm, WarpStoreAlgorithm
from .._types import (
    Algorithm,
    BasePrimitive,
    Dependency,
    DependentArray,
    DependentPointer,
    DependentReference,
    Invocable,
    TemplateParameter,
    Value,
    numba_type_to_wrapper,
)

CUB_WARP_LOAD_ALGOS = {
    "direct": "::cub::WARP_LOAD_DIRECT",
    "striped": "::cub::WARP_LOAD_STRIPED",
    "vectorize": "::cub::WARP_LOAD_VECTORIZE",
    "transpose": "::cub::WARP_LOAD_TRANSPOSE",
}

CUB_WARP_STORE_ALGOS = {
    "direct": "::cub::WARP_STORE_DIRECT",
    "striped": "::cub::WARP_STORE_STRIPED",
    "vectorize": "::cub::WARP_STORE_VECTORIZE",
    "transpose": "::cub::WARP_STORE_TRANSPOSE",
}


def _resolve_algorithm(algorithm, default_algorithm, cub_map):
    if algorithm is None:
        return str(default_algorithm)
    if isinstance(algorithm, str):
        if algorithm not in cub_map:
            raise ValueError(f"Invalid algorithm: {algorithm}")
        return cub_map[algorithm]
    if isinstance(algorithm, int):
        return str(default_algorithm.__class__(algorithm))
    if isinstance(algorithm, default_algorithm.__class__):
        return str(algorithm)
    raise ValueError(f"Invalid algorithm: {algorithm}")


class load(BasePrimitive):
    is_one_shot = True

    def __init__(
        self,
        dtype,
        items_per_thread: int,
        threads_in_warp: int = 32,
        algorithm: Optional[WarpLoadAlgorithm] = None,
        num_valid_items: Optional[int] = None,
        oob_default=None,
        methods: Optional[dict] = None,
        unique_id=None,
        temp_storage=None,
        node=None,
    ):
        """Create a warp-wide load operation."""
        if items_per_thread < 1:
            raise ValueError("items_per_thread must be >= 1")
        if oob_default is not None and num_valid_items is None:
            raise ValueError("oob_default requires num_valid_items to be set")

        self.node = node
        self.temp_storage = temp_storage

        dtype = normalize_dtype_param(dtype)
        algorithm_cub = _resolve_algorithm(
            algorithm, WarpLoadAlgorithm.DIRECT, CUB_WARP_LOAD_ALGOS
        )

        template_parameters = [
            TemplateParameter("T"),
            TemplateParameter("ITEMS_PER_THREAD"),
            TemplateParameter("ALGORITHM"),
            TemplateParameter("LOGICAL_WARP_THREADS"),
        ]

        parameters = [
            [
                DependentPointer(
                    value_dtype=Dependency("T"),
                    restrict=True,
                    is_array_pointer=True,
                    name="src",
                ),
                DependentArray(
                    value_dtype=Dependency("T"),
                    size=Dependency("ITEMS_PER_THREAD"),
                    name="dst",
                ),
            ]
        ]
        if num_valid_items is not None:
            parameters[0].append(Value(numba.types.int32, name="num_valid_items"))
        if oob_default is not None:
            parameters[0].append(
                DependentReference(Dependency("T"), name="oob_default")
            )

        type_definitions = None
        if methods is not None:
            type_definitions = [numba_type_to_wrapper(dtype, methods=methods)]

        template = Algorithm(
            "WarpLoad",
            "Load",
            "warp_load",
            ["cub/warp/warp_load.cuh"],
            template_parameters,
            parameters,
            self,
            type_definitions=type_definitions,
            threads=threads_in_warp,
            unique_id=unique_id,
        )

        self.algorithm = template
        self.specialization = template.specialize(
            {
                "T": dtype,
                "ITEMS_PER_THREAD": items_per_thread,
                "ALGORITHM": algorithm_cub,
                "LOGICAL_WARP_THREADS": threads_in_warp,
            }
        )

    @classmethod
    def create(
        cls,
        dtype,
        items_per_thread: int,
        threads_in_warp: int = 32,
        algorithm: Optional[WarpLoadAlgorithm] = None,
        num_valid_items: Optional[int] = None,
        oob_default=None,
        methods: Optional[dict] = None,
    ):
        algo = cls(
            dtype=dtype,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=algorithm,
            num_valid_items=num_valid_items,
            oob_default=oob_default,
            methods=methods,
        )
        specialization = algo.specialization
        return Invocable(
            ltoir_files=specialization.get_lto_ir(threads=threads_in_warp),
            temp_storage_bytes=specialization.temp_storage_bytes,
            temp_storage_alignment=specialization.temp_storage_alignment,
            algorithm=specialization,
        )


class store(BasePrimitive):
    is_one_shot = True

    def __init__(
        self,
        dtype,
        items_per_thread: int,
        threads_in_warp: int = 32,
        algorithm: Optional[WarpStoreAlgorithm] = None,
        num_valid_items: Optional[int] = None,
        methods: Optional[dict] = None,
        unique_id=None,
        temp_storage=None,
        node=None,
    ):
        """Create a warp-wide store operation."""
        if items_per_thread < 1:
            raise ValueError("items_per_thread must be >= 1")

        self.node = node
        self.temp_storage = temp_storage

        dtype = normalize_dtype_param(dtype)
        algorithm_cub = _resolve_algorithm(
            algorithm, WarpStoreAlgorithm.DIRECT, CUB_WARP_STORE_ALGOS
        )

        template_parameters = [
            TemplateParameter("T"),
            TemplateParameter("ITEMS_PER_THREAD"),
            TemplateParameter("ALGORITHM"),
            TemplateParameter("LOGICAL_WARP_THREADS"),
        ]

        parameters = [
            [
                DependentPointer(
                    value_dtype=Dependency("T"),
                    restrict=True,
                    is_array_pointer=True,
                    name="dst",
                ),
                DependentArray(
                    value_dtype=Dependency("T"),
                    size=Dependency("ITEMS_PER_THREAD"),
                    name="src",
                ),
            ]
        ]
        if num_valid_items is not None:
            parameters[0].append(Value(numba.types.int32, name="num_valid_items"))

        type_definitions = None
        if methods is not None:
            type_definitions = [numba_type_to_wrapper(dtype, methods=methods)]

        template = Algorithm(
            "WarpStore",
            "Store",
            "warp_store",
            ["cub/warp/warp_store.cuh"],
            template_parameters,
            parameters,
            self,
            type_definitions=type_definitions,
            threads=threads_in_warp,
            unique_id=unique_id,
        )

        self.algorithm = template
        self.specialization = template.specialize(
            {
                "T": dtype,
                "ITEMS_PER_THREAD": items_per_thread,
                "ALGORITHM": algorithm_cub,
                "LOGICAL_WARP_THREADS": threads_in_warp,
            }
        )

    @classmethod
    def create(
        cls,
        dtype,
        items_per_thread: int,
        threads_in_warp: int = 32,
        algorithm: Optional[WarpStoreAlgorithm] = None,
        num_valid_items: Optional[int] = None,
        methods: Optional[dict] = None,
    ):
        algo = cls(
            dtype=dtype,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=algorithm,
            num_valid_items=num_valid_items,
            methods=methods,
        )
        specialization = algo.specialization
        return Invocable(
            ltoir_files=specialization.get_lto_ir(threads=threads_in_warp),
            temp_storage_bytes=specialization.temp_storage_bytes,
            temp_storage_alignment=specialization.temp_storage_alignment,
            algorithm=specialization,
        )
