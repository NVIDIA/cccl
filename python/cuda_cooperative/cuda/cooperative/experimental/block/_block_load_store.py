# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import numba

from cuda.cooperative.experimental._common import make_binary_tempfile, normalize_dim_param
from cuda.cooperative.experimental._types import (
    Algorithm,
    Dependency,
    DependentArray,
    DependentPointer,
    Invocable,
    Pointer,
    TemplateParameter,
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

def load(dtype, threads_per_block, items_per_thread=1, algorithm="direct"):
    dim = normalize_dim_param(threads_per_block)
    template = Algorithm(
        "BlockLoad",
        "Load",
        "block_load",
        ["cub/block/block_load.cuh"],
        [
            TemplateParameter("T"),
            TemplateParameter("BLOCK_DIM_X"),
            TemplateParameter("ITEMS_PER_THREAD"),
            TemplateParameter("ALGORITHM"),
            TemplateParameter("BLOCK_DIM_Y"),
            TemplateParameter("BLOCK_DIM_Z"),
        ],
        [
            [
                Pointer(numba.uint8),
                DependentPointer(Dependency("T")),
                DependentArray(Dependency("T"), Dependency("ITEMS_PER_THREAD")),
            ]
        ],
    )
    specialization = template.specialize(
        {
            "T": dtype,
            "BLOCK_DIM_X": dim[0],
            "ITEMS_PER_THREAD": items_per_thread,
            "ALGORITHM": CUB_BLOCK_LOAD_ALGOS[algorithm],
            "BLOCK_DIM_Y": dim[1],
            "BLOCK_DIM_Z": dim[2],
        }
    )
    return Invocable(
        temp_files=[
            make_binary_tempfile(ltoir, ".ltoir")
            for ltoir in specialization.get_lto_ir()
        ],
        temp_storage_bytes=specialization.get_temp_storage_bytes(),
        algorithm=specialization,
    )


def store(dtype, threads_per_block, items_per_thread=1, algorithm="direct"):
    dim = normalize_dim_param(threads_per_block)
    template = Algorithm(
        "BlockStore",
        "Store",
        "block_store",
        ["cub/block/block_store.cuh"],
        [
            TemplateParameter("T"),
            TemplateParameter("BLOCK_DIM_X"),
            TemplateParameter("ITEMS_PER_THREAD"),
            TemplateParameter("ALGORITHM"),
            TemplateParameter("BLOCK_DIM_Y"),
            TemplateParameter("BLOCK_DIM_Z"),
        ],
        [
            [
                Pointer(numba.uint8),
                DependentPointer(Dependency("T")),
                DependentArray(Dependency("T"), Dependency("ITEMS_PER_THREAD")),
            ]
        ],
    )
    specialization = template.specialize(
        {
            "T": dtype,
            "BLOCK_DIM_X": dim[0],
            "ITEMS_PER_THREAD": items_per_thread,
            "ALGORITHM": CUB_BLOCK_STORE_ALGOS[algorithm],
            "BLOCK_DIM_Y": dim[1],
            "BLOCK_DIM_Z": dim[2],
        }
    )
    return Invocable(
        temp_files=[
            make_binary_tempfile(ltoir, ".ltoir")
            for ltoir in specialization.get_lto_ir()
        ],
        temp_storage_bytes=specialization.get_temp_storage_bytes(),
        algorithm=specialization,
    )
