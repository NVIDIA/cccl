# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from enum import IntEnum, auto

from .. import _require_runtime

_require_runtime()

from cuda.coop._core.block import BlockExchangeMode, make_block_exchange_spec

from .._common import (
    normalize_dim_param,
    normalize_dtype_param,
    resolve_threads_per_block_alias,
)
from .._core_adapter import NumbaMlirCoreAdapter
from .._types import make_invocable_from_specialization, numba_type_to_wrapper


class BlockExchangeType(IntEnum):
    """CUB block-exchange data movement pattern."""

    StripedToBlocked = auto()
    BlockedToStriped = auto()
    WarpStripedToBlocked = auto()
    BlockedToWarpStriped = auto()
    ScatterToBlocked = auto()
    ScatterToStriped = auto()
    ScatterToStripedGuarded = auto()
    ScatterToStripedFlagged = auto()


def _normalize_exchange_type(block_exchange_type):
    if isinstance(block_exchange_type, int):
        block_exchange_type = BlockExchangeType(block_exchange_type)
    if not isinstance(block_exchange_type, BlockExchangeType):
        raise ValueError(
            "block_exchange_type must be a valid BlockExchangeType enum value; "
            f"got: {block_exchange_type!r}"
        )
    return block_exchange_type


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
    dim=None,
):
    """Build a block-wide item exchange invocable.

    The invocable wraps CUB ``BlockExchange`` for blocked/striped layout
    conversions and scatter variants. Scatter modes require ``offset_dtype``;
    ``ScatterToStripedFlagged`` also requires ``valid_flag_dtype``. When
    ``use_output_items`` is true the generated signature writes to a distinct
    output item array, otherwise it can operate in place.
    """
    if dtype is None and not isinstance(block_exchange_type, (BlockExchangeType, int)):
        dtype = block_exchange_type
        block_exchange_type = BlockExchangeType.StripedToBlocked

    threads_per_block = resolve_threads_per_block_alias(threads_per_block, dim)
    block_exchange_type = _normalize_exchange_type(block_exchange_type)
    core_mode = BlockExchangeMode.from_cub_method_name(block_exchange_type.name)
    if dtype is None:
        raise ValueError("dtype must be provided")
    if threads_per_block is None:
        raise ValueError("threads_per_block must be provided")
    if items_per_thread < 1:
        raise ValueError("items_per_thread must be greater than or equal to 1")
    if use_output_items is not None and not isinstance(use_output_items, bool):
        raise ValueError(
            f"use_output_items must be a boolean or None; got: {use_output_items!r}"
        )

    dim = normalize_dim_param(threads_per_block)
    dtype = normalize_dtype_param(dtype)
    # CUB's guarded overload uses negative ranks as the guard; only the flagged
    # overload takes a separate validity array.
    if core_mode.uses_ranks:
        if offset_dtype is None:
            raise ValueError(
                "offset_dtype must be provided for scatter block_exchange_type values"
            )
        offset_dtype = normalize_dtype_param(offset_dtype)
    elif offset_dtype is not None:
        raise ValueError(
            "offset_dtype is only supported for scatter block_exchange_type values"
        )
    if core_mode.uses_valid_flags:
        if valid_flag_dtype is None:
            raise ValueError(
                "valid_flag_dtype must be provided for ScatterToStripedFlagged"
            )
        valid_flag_dtype = normalize_dtype_param(valid_flag_dtype)
    elif valid_flag_dtype is not None:
        raise ValueError(
            "valid_flag_dtype is only supported for ScatterToStripedFlagged"
        )

    value_form = (
        "both"
        if use_output_items is None
        else "out_of_place"
        if use_output_items
        else "in_place"
    )
    core_spec = make_block_exchange_spec(
        dtype=dtype,
        block_dim=tuple(dim),
        items_per_thread=items_per_thread,
        mode=core_mode,
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
