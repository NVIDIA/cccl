# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import numba

from .._common import (
    make_binary_tempfile,
    normalize_dim_param,
    normalize_dtype_param,
)
from .._types import (
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
    """Creates an operation that performs a block-wide load.

    Returns a callable object that can be linked to and invoked from device code. It can be
    invoked with the following signatures:

    - `(src: numba.types.Array, dest: numba.types.Array) -> None`: Each thread loads
        `items_per_thread` items from `src` into `dest`. `dest` must contain at least
        `items_per_thread` items.

    Different data movement strategies can be selected via the `algorithm` parameter:

    - `algorithm="direct"` (default): A blocked arrangement of data is read directly from memory.
    - `algorithm="striped"`: A striped arrangement of data is read directly from memory.
    - `algorithm="vectorize"`: A blocked arrangement of data is read directly from memory using CUDA's built-in vectorized loads as a coalescing optimization.
    - `algorithm="transpose"`: A striped arrangement of data is read directly from memory and is then locally transposed into a blocked arrangement.
    - `algorithm="warp_transpose"`: A warp-striped arrangement of data is read directly from memory and is then locally transposed into a blocked arrangement.
    - `algorithm="warp_transpose_timesliced"`: A warp-striped arrangement of data is read directly from memory and is then locally transposed into a blocked arrangement one warp at a time.

    For more details, [read the corresponding CUB C++ documentation](https://nvidia.github.io/cccl/cub/api/classcub_1_1BlockLoad.html).

    Args:
        dtype: Data type being loaded
        threads_per_block: The number of threads in a block, either an integer or a tuple of 2 or 3 integers
        items_per_thread: The number of items each thread loads
        algorithm: The data movement algorithm to use

    Example:
        The code snippet below illustrates a striped load and store of 128 integer items by 32 threads, with
        each thread handling 4 integers.

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_load_store_api.py
            :language: python
            :dedent:
            :start-after: example-begin imports
            :end-before: example-end imports

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_load_store_api.py
            :language: python
            :dedent:
            :start-after: example-begin load_store
            :end-before: example-end load_store
    """
    dim = normalize_dim_param(threads_per_block)
    dtype = normalize_dtype_param(dtype)

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
        temp_storage_bytes=specialization.temp_storage_bytes,
        temp_storage_alignment=specialization.temp_storage_alignment,
        algorithm=specialization,
    )


def store(dtype, threads_per_block, items_per_thread=1, algorithm="direct"):
    """Creates an operation that performs a block-wide store.

    Returns a callable object that can be linked to and invoked from device code. It can be
    invoked with the following signatures:

    - `(dest: numba.types.Array, src: numba.types.Array) -> None`: Each thread stores
        `items_per_thread` items from `src` into `dest`. `src` must contain at least
        `items_per_thread` items.

    Different data movement strategies can be selected via the `algorithm` parameter:

    - `algorithm="direct"` (default): A blocked arrangement of data is written directly to memory.
    - `algorithm="striped"`: A striped arrangement of data is written directly to memory.
    - `algorithm="vectorize"`: A blocked arrangement of data is written directly to memory using CUDA's built-in vectorized stores as a coalescing optimization.
    - `algorithm="transpose"`: A blocked arrangement is locally transposed into a striped arrangement which is then written to memory.
    - `algorithm="warp_transpose"`: A blocked arrangement is locally transposed into a warp-striped arrangement which is then written to memory.
    - `algorithm="warp_transpose_timesliced"`: A blocked arrangement is locally transposed into a warp-striped arrangement which is then written to memory. To reduce the shared memory requireent, only one warp’s worth of shared memory is provisioned and is subsequently time-sliced among warps.

    For more details, [read the corresponding CUB C++ documentation](https://nvidia.github.io/cccl/cub/api/classcub_1_1BlockStore.html).

    Args:
        dtype: Data type being stored
        threads_per_block: The number of threads in a block, either an integer or a tuple of 2 or 3 integers
        items_per_thread: The number of items each thread loads
        algorithm: The data movement algorithm to use

    Example:
        The code snippet below illustrates a striped load and store of 128 integer items by 32 threads, with
        each thread handling 4 integers.

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_load_store_api.py
            :language: python
            :dedent:
            :start-after: example-begin imports
            :end-before: example-end imports

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_load_store_api.py
            :language: python
            :dedent:
            :start-after: example-begin load_store
            :end-before: example-end load_store
    """
    dim = normalize_dim_param(threads_per_block)
    dtype = normalize_dtype_param(dtype)

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
        temp_storage_bytes=specialization.temp_storage_bytes,
        temp_storage_alignment=specialization.temp_storage_alignment,
        algorithm=specialization,
    )
