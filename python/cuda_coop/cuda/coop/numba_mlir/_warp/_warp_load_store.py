# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from .. import _require_runtime

_require_runtime()

from cuda.coop._core.warp import make_warp_load_spec, make_warp_store_spec

from .._common import normalize_dtype_param
from .._core_adapter import NumbaMlirCoreAdapter
from .._enums import WarpLoadAlgorithm, WarpStoreAlgorithm
from .._types import (
    make_invocable_from_specialization,
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


def _resolve_warp_load_algorithm(algorithm):
    if isinstance(algorithm, WarpLoadAlgorithm):
        return str(algorithm)
    if isinstance(algorithm, int):
        return str(WarpLoadAlgorithm(algorithm))
    if isinstance(algorithm, str):
        if algorithm.startswith("::cub::"):
            return algorithm
        if algorithm in CUB_WARP_LOAD_ALGOS:
            return CUB_WARP_LOAD_ALGOS[algorithm]
        upper = algorithm.upper()
        if upper in WarpLoadAlgorithm.__members__:
            return str(WarpLoadAlgorithm[upper])
    allowed = sorted(CUB_WARP_LOAD_ALGOS.keys())
    raise ValueError(
        "Unsupported warp load algorithm "
        f"{algorithm!r}; expected one of {allowed} or WarpLoadAlgorithm."
    )


def _resolve_warp_store_algorithm(algorithm):
    if isinstance(algorithm, WarpStoreAlgorithm):
        return str(algorithm)
    if isinstance(algorithm, int):
        return str(WarpStoreAlgorithm(algorithm))
    if isinstance(algorithm, str):
        if algorithm.startswith("::cub::"):
            return algorithm
        if algorithm in CUB_WARP_STORE_ALGOS:
            return CUB_WARP_STORE_ALGOS[algorithm]
        upper = algorithm.upper()
        if upper in WarpStoreAlgorithm.__members__:
            return str(WarpStoreAlgorithm[upper])
    allowed = sorted(CUB_WARP_STORE_ALGOS.keys())
    raise ValueError(
        "Unsupported warp store algorithm "
        f"{algorithm!r}; expected one of {allowed} or WarpStoreAlgorithm."
    )


def warp_load(
    dtype,
    items_per_thread=1,
    threads_in_warp=32,
    algorithm="direct",
    num_valid_items=None,
    oob_default=None,
    methods=None,
    threads_per_block=None,
):
    """Build a warp-wide load invocable.

    The invocable wraps CUB ``WarpLoad`` and loads ``items_per_thread`` values
    per lane into a local item array. ``num_valid_items`` enables partial-warp
    loads and ``oob_default`` supplies the default for lanes outside the valid
    item range.
    """
    if items_per_thread < 1:
        raise ValueError("items_per_thread must be greater than or equal to 1")
    if oob_default is not None and num_valid_items is None:
        raise ValueError("oob_default requires num_valid_items to be provided")

    dtype = normalize_dtype_param(dtype)
    algorithm = _resolve_warp_load_algorithm(algorithm)

    core_spec = make_warp_load_spec(
        dtype=dtype,
        items_per_thread=items_per_thread,
        threads_in_warp=threads_in_warp,
        algorithm=algorithm,
        valid_items=num_valid_items is not None,
        oob_default=oob_default is not None,
        include_full_tile=num_valid_items is not None,
        include_pointer_offset=True,
    )
    specialization = NumbaMlirCoreAdapter().materialize(
        core_spec.specialization,
        extra_type_definitions=(numba_type_to_wrapper(dtype, methods=methods),),
    )
    return make_invocable_from_specialization(
        specialization, threads=threads_in_warp, block_threads=threads_per_block
    )


def warp_store(
    dtype,
    items_per_thread=1,
    threads_in_warp=32,
    algorithm="direct",
    num_valid_items=None,
    methods=None,
    threads_per_block=None,
):
    """Build a warp-wide store invocable.

    The invocable wraps CUB ``WarpStore`` and writes ``items_per_thread``
    values per lane from a local item array to memory. ``num_valid_items``
    enables the partial-warp store overload.
    """
    if items_per_thread < 1:
        raise ValueError("items_per_thread must be greater than or equal to 1")

    dtype = normalize_dtype_param(dtype)
    algorithm = _resolve_warp_store_algorithm(algorithm)

    core_spec = make_warp_store_spec(
        dtype=dtype,
        items_per_thread=items_per_thread,
        threads_in_warp=threads_in_warp,
        algorithm=algorithm,
        valid_items=num_valid_items is not None,
        include_full_tile=num_valid_items is not None,
        include_pointer_offset=True,
    )
    specialization = NumbaMlirCoreAdapter().materialize(
        core_spec.specialization,
        extra_type_definitions=(numba_type_to_wrapper(dtype, methods=methods),),
    )
    return make_invocable_from_specialization(
        specialization, threads=threads_in_warp, block_threads=threads_per_block
    )


load = warp_load
store = warp_store
