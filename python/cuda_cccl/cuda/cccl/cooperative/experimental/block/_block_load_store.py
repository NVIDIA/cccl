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
    Pointer,
    TemplateParameter,
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
    template_parameters = [
        TemplateParameter("T"),
        TemplateParameter("BLOCK_DIM_X"),
        TemplateParameter("ITEMS_PER_THREAD"),
        TemplateParameter("ALGORITHM"),
        TemplateParameter("BLOCK_DIM_Y"),
        TemplateParameter("BLOCK_DIM_Z"),
    ]

    parameters = [
        [
            Pointer(numba.uint8),
            DependentPointer(Dependency("T")),
            DependentArray(Dependency("T"), Dependency("ITEMS_PER_THREAD")),
        ]
    ]

    def __init__(
        self,
        dtype: DtypeType,
        dim: DimType,
        items_per_thread: int,
        algorithm=None,
        temp_storage=None,
    ) -> None:
        self.dtype = normalize_dtype_param(dtype)
        self.dim = normalize_dim_param(dim)
        self.items_per_thread = items_per_thread
        algorithm_enum = None
        if algorithm is not None:
            enum_class = self.default_algorithm.__class__
            if isinstance(algorithm, str):
                algorithm_cub = CUB_BLOCK_LOAD_ALGOS[algorithm]
            elif isinstance(algorithm, int):
                algorithm_enum = enum_class(algorithm)
                algorithm_cub = str(algorithm_enum)
            else:
                enum_class = self.default_algorithm.__class__
                if not isinstance(algorithm, enum_class):
                    raise ValueError(f"Invalid algorithm: {algorithm}")
                algorithm_cub = str(algorithm)
        else:
            algorithm_cub = str(self.default_algorithm)

        self.algorithm = Algorithm(
            self.struct_name,
            self.method_name,
            self.c_name,
            self.includes,
            self.template_parameters,
            self.parameters,
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

    def __call__(self, src, dst, items_per_thread=None):
        """
        Invokes the load/store operation on the specified source and
        destination.
        """

        if items_per_thread is None:
            items_per_thread = self.items_per_thread

        if not isinstance(src, numba.cuda.cudadrv.devicearray.DeviceNDArray):
            raise TypeError("Source must be a device array.")
        if not isinstance(dst, numba.cuda.cudadrv.devicearray.DeviceNDArray):
            raise TypeError("Destination must be a device array.")

        if src.dtype != self.dtype or dst.dtype != self.dtype:
            raise ValueError("Source and destination must have the same dtype.")

        return Invocable(
            ltoir_files=self.specialization.get_lto_ir(),
            temp_storage_bytes=self.specialization.temp_storage_bytes,
            temp_storage_alignment=self.specialization.temp_storage_alignment,
            algorithm=self.specialization,
        )

    @classmethod
    def create(
        cls,
        dtype: DtypeType,
        threads_per_block: DimType,
        items_per_thread: int,
        algorithm=None,
    ):
        """Creates a block-wide load operation."""
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
