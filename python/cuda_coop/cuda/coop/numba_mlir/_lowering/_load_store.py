# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Load and store provider lowering for Numba-CUDA-MLIR.

This module owns both block and warp provider construction for the movement family. Group planning selects a route here after resolving hierarchy, dtype, and launch facts.
"""

import operator

from cuda.coop._core import ArgumentBinding, BindingKind
from cuda.coop._core.block import make_block_load_spec, make_block_store_spec
from cuda.coop._core.warp import make_warp_load_spec, make_warp_store_spec

from .._compiler._parameters import (
    normalize_dim_param,
    normalize_dtype_param,
)
from .._enums import (
    BlockLoadAlgorithm,
    BlockStoreAlgorithm,
    WarpLoadAlgorithm,
    WarpStoreAlgorithm,
)
from .._types import make_invocable_from_specialization, numba_type_to_wrapper
from ._core import NumbaMlirCoreAdapter, _optional_binding


def _positive_int(value, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _resolve_algorithm(algorithm, enum_type, primitive_name: str) -> str:
    if isinstance(algorithm, enum_type):
        return str(algorithm)
    if isinstance(algorithm, bool):
        raise TypeError(f"{primitive_name} algorithm must not be bool")
    try:
        index = operator.index(algorithm)
    except TypeError:
        index = None
    if index is not None:
        try:
            return str(enum_type(index))
        except ValueError:
            pass
    if isinstance(algorithm, str):
        if algorithm.startswith("::cub::"):
            return algorithm
        upper = algorithm.upper()
        if upper in enum_type.__members__:
            return str(enum_type[upper])
    allowed = sorted(member.name.lower() for member in enum_type)
    raise ValueError(
        f"Unsupported {primitive_name} algorithm {algorithm!r}; expected one "
        f"of {allowed} or {enum_type.__name__}."
    )


def load(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    algorithm="direct",
    num_valid_items=None,
    oob_default=None,
    offset=None,
):
    """Build the block-load invocable selected by movement planning."""

    valid_items_binding = _optional_binding(num_valid_items)
    oob_default_binding = _optional_binding(oob_default)
    offset_binding = _optional_binding(offset)
    if (
        oob_default_binding.kind is not BindingKind.OMITTED
        and valid_items_binding.kind is BindingKind.OMITTED
    ):
        raise ValueError("oob_default requires num_valid_items to be provided")
    if threads_per_block is None:
        raise ValueError("threads_per_block must be provided")
    block_dim = normalize_dim_param(threads_per_block)
    dtype = normalize_dtype_param(dtype)
    items_per_thread = _positive_int(items_per_thread, name="items_per_thread")
    algorithm = _resolve_algorithm(algorithm, BlockLoadAlgorithm, "block load")
    core_spec = make_block_load_spec(
        dtype=dtype,
        block_dim=tuple(block_dim),
        items_per_thread=items_per_thread,
        algorithm=algorithm,
        valid_items=valid_items_binding,
        oob_default=oob_default_binding,
        include_full_tile=(
            not isinstance(num_valid_items, ArgumentBinding)
            and num_valid_items is not None
        ),
        include_pointer_offset=(
            offset_binding if isinstance(offset, ArgumentBinding) else True
        ),
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
    offset=None,
):
    """Build the block-store invocable selected by movement planning."""

    if oob_default is not None:
        raise ValueError("oob_default is only valid for BlockLoad")
    valid_items_binding = _optional_binding(num_valid_items)
    offset_binding = _optional_binding(offset)
    if threads_per_block is None:
        raise ValueError("threads_per_block must be provided")
    block_dim = normalize_dim_param(threads_per_block)
    dtype = normalize_dtype_param(dtype)
    items_per_thread = _positive_int(items_per_thread, name="items_per_thread")
    algorithm = _resolve_algorithm(algorithm, BlockStoreAlgorithm, "block store")
    core_spec = make_block_store_spec(
        dtype=dtype,
        block_dim=tuple(block_dim),
        items_per_thread=items_per_thread,
        algorithm=algorithm,
        valid_items=valid_items_binding,
        include_full_tile=(
            not isinstance(num_valid_items, ArgumentBinding)
            and num_valid_items is not None
        ),
        include_pointer_offset=(
            offset_binding if isinstance(offset, ArgumentBinding) else True
        ),
    )
    specialization = NumbaMlirCoreAdapter().materialize(
        core_spec.specialization,
        extra_type_definitions=(numba_type_to_wrapper(dtype),),
    )
    return make_invocable_from_specialization(specialization)


def warp_load(
    dtype,
    items_per_thread=1,
    threads_in_warp=32,
    algorithm="direct",
    num_valid_items=None,
    oob_default=None,
    offset=None,
    methods=None,
    threads_per_block=None,
):
    """Build the warp-load invocable selected by movement planning."""

    valid_items_binding = _optional_binding(num_valid_items)
    oob_default_binding = _optional_binding(oob_default)
    offset_binding = _optional_binding(offset)
    if (
        oob_default_binding.kind is not BindingKind.OMITTED
        and valid_items_binding.kind is BindingKind.OMITTED
    ):
        raise ValueError("oob_default requires num_valid_items to be provided")
    dtype = normalize_dtype_param(dtype)
    items_per_thread = _positive_int(items_per_thread, name="items_per_thread")
    threads_in_warp = _positive_int(threads_in_warp, name="threads_in_warp")
    algorithm = _resolve_algorithm(algorithm, WarpLoadAlgorithm, "warp load")
    core_spec = make_warp_load_spec(
        dtype=dtype,
        items_per_thread=items_per_thread,
        threads_in_warp=threads_in_warp,
        algorithm=algorithm,
        valid_items=valid_items_binding,
        oob_default=oob_default_binding,
        include_full_tile=(
            not isinstance(num_valid_items, ArgumentBinding)
            and num_valid_items is not None
        ),
        include_pointer_offset=(
            offset_binding if isinstance(offset, ArgumentBinding) else True
        ),
    )
    specialization = NumbaMlirCoreAdapter().materialize(
        core_spec.specialization,
        extra_type_definitions=(numba_type_to_wrapper(dtype, methods=methods),),
    )
    return make_invocable_from_specialization(
        specialization,
        threads=threads_in_warp,
        block_threads=threads_per_block,
    )


def warp_store(
    dtype,
    items_per_thread=1,
    threads_in_warp=32,
    algorithm="direct",
    num_valid_items=None,
    offset=None,
    methods=None,
    threads_per_block=None,
):
    """Build the warp-store invocable selected by movement planning."""

    dtype = normalize_dtype_param(dtype)
    valid_items_binding = _optional_binding(num_valid_items)
    offset_binding = _optional_binding(offset)
    items_per_thread = _positive_int(items_per_thread, name="items_per_thread")
    threads_in_warp = _positive_int(threads_in_warp, name="threads_in_warp")
    algorithm = _resolve_algorithm(algorithm, WarpStoreAlgorithm, "warp store")
    core_spec = make_warp_store_spec(
        dtype=dtype,
        items_per_thread=items_per_thread,
        threads_in_warp=threads_in_warp,
        algorithm=algorithm,
        valid_items=valid_items_binding,
        include_full_tile=(
            not isinstance(num_valid_items, ArgumentBinding)
            and num_valid_items is not None
        ),
        include_pointer_offset=(
            offset_binding if isinstance(offset, ArgumentBinding) else True
        ),
    )
    specialization = NumbaMlirCoreAdapter().materialize(
        core_spec.specialization,
        extra_type_definitions=(numba_type_to_wrapper(dtype, methods=methods),),
    )
    return make_invocable_from_specialization(
        specialization,
        threads=threads_in_warp,
        block_threads=threads_per_block,
    )
