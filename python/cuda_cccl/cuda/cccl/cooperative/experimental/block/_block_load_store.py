# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import numba

from .._common import (
    normalize_dim_param,
    normalize_dtype_param,
)
from .._enums import (
    BlockLoadAlgorithm,
    BlockStoreAlgorithm,
)
from .._types import (
    Algorithm,
    BasePrimitive,
    Dependency,
    DependentArray,
    DependentPointer,
    Invocable,
    TemplateParameter,
    Value,
)
from .._typing import (
    DimType,
    DtypeType,
)

CUB_BLOCK_LOAD_ALGOS = {
    "direct": "::cub::BLOCK_LOAD_DIRECT",
    "striped": "::cub::BLOCK_LOAD_STRIPED",
    "vectorize": "::cub::BLOCK_LOAD_VECTORIZE",
    "transpose": "::cub::BLOCK_LOAD_TRANSPOSE",
    "warp_transpose": "::cub::BLOCK_LOAD_WARP_TRANSPOSE",
    "warp_transpose_timesliced": "::cub::BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED",
}

CUB_BLOCK_STORE_ALGOS = {
    "direct": "::cub::BLOCK_STORE_DIRECT",
    "striped": "::cub::BLOCK_STORE_STRIPED",
    "vectorize": "::cub::BLOCK_STORE_VECTORIZE",
    "transpose": "::cub::BLOCK_STORE_TRANSPOSE",
    "warp_transpose": "::cub::BLOCK_STORE_WARP_TRANSPOSE",
    "warp_transpose_timesliced": "::cub::BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED",
}


class BaseLoadStore(BasePrimitive):
    is_one_shot = True

    template_parameters = [
        TemplateParameter("T"),
        TemplateParameter("BLOCK_DIM_X"),
        TemplateParameter("ITEMS_PER_THREAD"),
        TemplateParameter("ALGORITHM"),
        TemplateParameter("BLOCK_DIM_Y"),
        TemplateParameter("BLOCK_DIM_Z"),
    ]

    def __init__(
        self,
        dtype: DtypeType,
        dim: DimType,
        items_per_thread: int,
        algorithm=None,
        num_valid_items=None,
        unique_id: int = None,
        temp_storage=None,
    ) -> None:
        self.dtype = normalize_dtype_param(dtype)
        self.dim = normalize_dim_param(dim)
        self.items_per_thread = items_per_thread
        self.num_valid_items = num_valid_items
        self.unique_id = unique_id
        (algorithm_cub, algorithm_enum) = self.resolve_cub_algorithm(
            algorithm,
        )

        input_is_array_pointer = items_per_thread > 1

        parameters = [
            [
                DependentPointer(
                    value_dtype=Dependency("T"),
                    restrict=True,
                    is_array_pointer=input_is_array_pointer,
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
        self.parameters = parameters

        self.algorithm = Algorithm(
            self.struct_name,
            self.method_name,
            self.c_name,
            self.includes,
            self.template_parameters,
            self.parameters,
            self,
            unique_id=unique_id,
        )
        self.specialization = self.algorithm.specialize(
            {
                "T": self.dtype,
                "BLOCK_DIM_X": self.dim[0],
                "ITEMS_PER_THREAD": items_per_thread,
                "ALGORITHM": algorithm_cub,
                "BLOCK_DIM_Y": self.dim[1],
                "BLOCK_DIM_Z": self.dim[2],
            }
        )
        self.temp_storage = temp_storage

    @classmethod
    def create(
        cls,
        dtype: DtypeType,
        threads_per_block: DimType,
        items_per_thread: int,
        algorithm=None,
    ):
        algo = cls(dtype, threads_per_block, items_per_thread, algorithm)
        specialization = algo.specialization

        return Invocable(
            ltoir_files=specialization.get_lto_ir(),
            temp_storage_bytes=specialization.temp_storage_bytes,
            temp_storage_alignment=specialization.temp_storage_alignment,
            algorithm=specialization,
        )


class load(BaseLoadStore):
    default_algorithm = BlockLoadAlgorithm.DIRECT
    cub_algorithm_map = CUB_BLOCK_LOAD_ALGOS
    struct_name = "BlockLoad"
    method_name = "Load"
    c_name = "block_load"
    includes = ["cub/block/block_load.cuh"]


def BlockLoad(
    dtype: DtypeType,
    threads_per_block: DimType,
    items_per_thread: int,
    algorithm=None,
):
    """
    Creates a block-wide load operation.
    """
    return load.create(dtype, threads_per_block, items_per_thread, algorithm)


class store(BaseLoadStore):
    default_algorithm = BlockStoreAlgorithm.DIRECT
    cub_algorithm_map = CUB_BLOCK_STORE_ALGOS
    struct_name = "BlockStore"
    method_name = "Store"
    c_name = "block_store"
    includes = ["cub/block/block_store.cuh"]


def BlockStore(
    dtype: DtypeType,
    threads_per_block: DimType,
    items_per_thread: int,
    algorithm=None,
):
    """
    Creates a block-wide store operation.
    """
    return store.create(dtype, threads_per_block, items_per_thread, algorithm)
