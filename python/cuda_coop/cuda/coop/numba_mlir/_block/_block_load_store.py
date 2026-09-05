# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from .. import _require_runtime

_require_runtime()

from cuda.coop._core.block import make_block_load_spec, make_block_store_spec

from .._common import (
    normalize_dim_param,
    normalize_dtype_param,
    resolve_threads_per_block_alias,
)
from .._core_adapter import NumbaMlirCoreAdapter
from .._enums import (
    BlockLoadAlgorithm,
    BlockStoreAlgorithm,
)
from .._types import make_invocable_from_specialization, numba_type_to_wrapper


def _resolve_block_algorithm(algorithm, enum_type, primitive_name):
    if isinstance(algorithm, enum_type):
        return str(algorithm)
    if isinstance(algorithm, int):
        return str(enum_type(algorithm))
    if isinstance(algorithm, str):
        if algorithm.startswith("::cub::"):
            return algorithm
        upper = algorithm.upper()
        if upper in enum_type.__members__:
            return str(enum_type[upper])
    allowed = sorted(member.name.lower() for member in enum_type)
    raise ValueError(
        f"Unsupported {primitive_name} algorithm {algorithm!r}; expected one of "
        f"{allowed} or {enum_type.__name__}."
    )


def _resolve_block_load_algorithm(algorithm):
    return _resolve_block_algorithm(
        algorithm,
        BlockLoadAlgorithm,
        "block load",
    )


def _resolve_block_store_algorithm(algorithm):
    return _resolve_block_algorithm(
        algorithm,
        BlockStoreAlgorithm,
        "block store",
    )


def load(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    algorithm="direct",
    num_valid_items=None,
    oob_default=None,
    dim=None,
):
    """Creates an operation that performs a block-wide load.

    Returns a callable object that can be linked to and invoked from device code. It can be
    invoked with the following signatures:

    - `(src: numba_cuda_mlir.types.Array, dest: numba_cuda_mlir.types.Array) -> None`: Each thread loads
        `items_per_thread` items from `src` into `dest`. `dest` must contain at least
        `items_per_thread` items.
    - `(src, dest, offset) -> None`: Loads from `src + offset`, where `offset`
        is counted in elements.
    - `(src, dest, num_valid_items) -> None`: Guards a partial tile and leaves
        out-of-range destination values unspecified.
    - `(src, dest, num_valid_items, oob_default) -> None`: Guards a partial tile
        and fills out-of-range destination values with `oob_default`.

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
        num_valid_items: Enables the CUB partial-tile overload.
        oob_default: Default value for out-of-bounds items. Requires
            `num_valid_items`.

    """
    if oob_default is not None and num_valid_items is None:
        raise ValueError("oob_default requires num_valid_items to be provided")
    threads_per_block = resolve_threads_per_block_alias(threads_per_block, dim)

    dim = normalize_dim_param(threads_per_block)
    dtype = normalize_dtype_param(dtype)
    algorithm = _resolve_block_load_algorithm(algorithm)
    core_spec = make_block_load_spec(
        dtype=dtype,
        block_dim=tuple(dim),
        items_per_thread=items_per_thread,
        algorithm=algorithm,
        valid_items=num_valid_items is not None,
        oob_default=oob_default is not None,
        include_full_tile=num_valid_items is not None,
        include_pointer_offset=True,
    )
    specialization = NumbaMlirCoreAdapter().materialize(
        core_spec.specialization,
        extra_type_definitions=(numba_type_to_wrapper(dtype),),
    )
    return make_invocable_from_specialization(specialization)


def store(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    algorithm="direct",
    num_valid_items=None,
    oob_default=None,
    dim=None,
):
    """Creates an operation that performs a block-wide store.

    Returns a callable object that can be linked to and invoked from device code. It can be
    invoked with the following signatures:

    - `(dest: numba_cuda_mlir.types.Array, src: numba_cuda_mlir.types.Array) -> None`: Each thread stores
        `items_per_thread` items from `src` into `dest`. `src` must contain at least
        `items_per_thread` items.
    - `(dest, src, offset) -> None`: Stores to `dest + offset`, where `offset`
        is counted in elements.
    - `(dest, src, num_valid_items) -> None`: Guards a partial tile and writes
        only the valid prefix.

    Different data movement strategies can be selected via the `algorithm` parameter:

    - `algorithm="direct"` (default): A blocked arrangement of data is written directly to memory.
    - `algorithm="striped"`: A striped arrangement of data is written directly to memory.
    - `algorithm="vectorize"`: A blocked arrangement of data is written directly to memory using CUDA's built-in vectorized stores as a coalescing optimization.
    - `algorithm="transpose"`: A blocked arrangement is locally transposed into a striped arrangement which is then written to memory.
    - `algorithm="warp_transpose"`: A blocked arrangement is locally transposed into a warp-striped arrangement which is then written to memory.
    - `algorithm="warp_transpose_timesliced"`: A blocked arrangement is locally transposed into a warp-striped arrangement which is then written to memory. To reduce the shared memory requireent, only one warp's worth of shared memory is provisioned and is subsequently time-sliced among warps.

    For more details, [read the corresponding CUB C++ documentation](https://nvidia.github.io/cccl/cub/api/classcub_1_1BlockStore.html).

    Args:
        dtype: Data type being stored
        threads_per_block: The number of threads in a block, either an integer or a tuple of 2 or 3 integers
        items_per_thread: The number of items each thread loads
        algorithm: The data movement algorithm to use
        num_valid_items: Enables the CUB partial-tile overload.
        oob_default: Reserved compatibility keyword for load/store factories;
            store rejects non-``None`` values.

    """
    if oob_default is not None:
        raise ValueError("oob_default is only valid for BlockLoad")
    threads_per_block = resolve_threads_per_block_alias(threads_per_block, dim)

    dim = normalize_dim_param(threads_per_block)
    dtype = normalize_dtype_param(dtype)
    algorithm = _resolve_block_store_algorithm(algorithm)
    core_spec = make_block_store_spec(
        dtype=dtype,
        block_dim=tuple(dim),
        items_per_thread=items_per_thread,
        algorithm=algorithm,
        valid_items=num_valid_items is not None,
        include_full_tile=num_valid_items is not None,
        include_pointer_offset=True,
    )
    specialization = NumbaMlirCoreAdapter().materialize(
        core_spec.specialization,
        extra_type_definitions=(numba_type_to_wrapper(dtype),),
    )
    return make_invocable_from_specialization(specialization)
