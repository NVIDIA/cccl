# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Planner-private block exchange provider."""

import operator
from enum import IntEnum, auto

from cuda.coop._core.block import BlockExchangeMode, make_block_exchange_spec

from .._common import (
    normalize_dim_param,
    normalize_dtype_param,
)
from .._core_adapter import NumbaMlirCoreAdapter
from .._types import make_invocable_from_specialization, numba_type_to_wrapper


class BlockExchangeType(IntEnum):
    """CUB block-exchange data movement patterns."""

    StripedToBlocked = auto()
    BlockedToStriped = auto()
    WarpStripedToBlocked = auto()
    BlockedToWarpStriped = auto()
    ScatterToBlocked = auto()
    ScatterToStriped = auto()
    ScatterToStripedGuarded = auto()
    ScatterToStripedFlagged = auto()


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


def _normalize_exchange_type(value) -> BlockExchangeType:
    if isinstance(value, bool):
        raise TypeError("block_exchange_type must not be bool")
    if isinstance(value, BlockExchangeType):
        return value
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(
            "block_exchange_type must be a BlockExchangeType or integer"
        ) from exc
    try:
        return BlockExchangeType(value)
    except ValueError as exc:
        raise ValueError(f"invalid BlockExchangeType value: {value!r}") from exc


def exchange(
    block_exchange_type=BlockExchangeType.StripedToBlocked,
    dtype=None,
    threads_per_block=None,
    items_per_thread=1,
    warp_time_slicing=False,
    use_output_items=None,
    offset_dtype=None,
    valid_flag_dtype=None,
    methods=None,
):
    """Build the block-exchange invocable selected by movement planning."""

    if dtype is None:
        raise ValueError("dtype must be provided")
    if threads_per_block is None:
        raise ValueError("threads_per_block must be provided")
    if not isinstance(warp_time_slicing, bool):
        raise TypeError("warp_time_slicing must be a bool")
    if use_output_items is not None and not isinstance(use_output_items, bool):
        raise TypeError("use_output_items must be a bool or None")

    block_exchange_type = _normalize_exchange_type(block_exchange_type)
    mode = BlockExchangeMode.from_cub_method_name(block_exchange_type.name)
    dtype = normalize_dtype_param(dtype)
    block_dim = normalize_dim_param(threads_per_block)
    items_per_thread = _positive_int(items_per_thread, name="items_per_thread")

    if mode.uses_ranks:
        if offset_dtype is None:
            raise ValueError("offset_dtype is required for scatter exchange modes")
        offset_dtype = normalize_dtype_param(offset_dtype)
    elif offset_dtype is not None:
        raise ValueError("offset_dtype is valid only for scatter exchange modes")
    if mode.uses_valid_flags:
        if valid_flag_dtype is None:
            raise ValueError("valid_flag_dtype is required for ScatterToStripedFlagged")
        valid_flag_dtype = normalize_dtype_param(valid_flag_dtype)
    elif valid_flag_dtype is not None:
        raise ValueError("valid_flag_dtype is valid only for ScatterToStripedFlagged")

    value_form = (
        "both"
        if use_output_items is None
        else "out_of_place"
        if use_output_items
        else "in_place"
    )
    core_spec = make_block_exchange_spec(
        dtype=dtype,
        block_dim=tuple(block_dim),
        items_per_thread=items_per_thread,
        mode=mode,
        value_form=value_form,
        warp_time_slicing=warp_time_slicing,
        rank_dtype=offset_dtype,
        valid_flag_dtype=valid_flag_dtype,
    )
    specialization = NumbaMlirCoreAdapter().materialize(
        core_spec.specialization,
        extra_type_definitions=(numba_type_to_wrapper(dtype, methods=methods),),
    )
    return make_invocable_from_specialization(specialization)
