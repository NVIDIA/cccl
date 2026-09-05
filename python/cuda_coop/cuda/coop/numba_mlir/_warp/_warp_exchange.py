# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from enum import IntEnum, auto

from .. import _require_runtime

_require_runtime()

from numba_cuda_mlir import types

from cuda.coop._core.warp import make_warp_exchange_spec

from .._common import normalize_dtype_param
from .._core_adapter import NumbaMlirCoreAdapter
from .._types import (
    make_invocable_from_specialization,
    numba_type_to_wrapper,
)


class WarpExchangeType(IntEnum):
    """CUB warp-exchange data movement pattern."""

    StripedToBlocked = auto()
    BlockedToStriped = auto()
    ScatterToStriped = auto()


def _normalize_exchange_type(warp_exchange_type):
    if isinstance(warp_exchange_type, int):
        warp_exchange_type = WarpExchangeType(warp_exchange_type)
    if not isinstance(warp_exchange_type, WarpExchangeType):
        raise ValueError(
            "warp_exchange_type must be a valid WarpExchangeType enum value; "
            f"got: {warp_exchange_type!r}"
        )
    return warp_exchange_type


def warp_exchange(
    dtype,
    items_per_thread=1,
    threads_in_warp=32,
    warp_exchange_type=WarpExchangeType.StripedToBlocked,
    offset_dtype=None,
    use_output_items=None,
    methods=None,
    threads_per_block=None,
):
    """Build a warp-wide item exchange invocable.

    The invocable wraps CUB ``WarpExchange`` for striped/blocked conversions
    and scatter-to-striped movement. Scatter mode accepts rank offsets through
    ``offset_dtype`` and may operate in place or write a distinct output array.
    """
    try:
        items_per_thread = int(items_per_thread)
    except (TypeError, ValueError) as e:
        raise ValueError("items_per_thread must be an integer") from e
    if items_per_thread < 1:
        raise ValueError("items_per_thread must be greater than or equal to 1")
    if use_output_items is not None and not isinstance(use_output_items, bool):
        raise ValueError(
            f"use_output_items must be a boolean or None; got: {use_output_items!r}"
        )

    warp_exchange_type = _normalize_exchange_type(warp_exchange_type)
    dtype = normalize_dtype_param(dtype)

    mode = {
        WarpExchangeType.StripedToBlocked: "striped_to_blocked",
        WarpExchangeType.BlockedToStriped: "blocked_to_striped",
        WarpExchangeType.ScatterToStriped: "scatter_to_striped",
    }[warp_exchange_type]

    if warp_exchange_type == WarpExchangeType.ScatterToStriped:
        if offset_dtype is None:
            offset_dtype = types.int32
        offset_dtype = normalize_dtype_param(offset_dtype)
        value_form = {
            None: "both",
            True: "out_of_place",
            False: "in_place",
        }[use_output_items]
    elif use_output_items is not None:
        raise ValueError("use_output_items is only supported for ScatterToStriped")
    elif offset_dtype is not None:
        raise ValueError("offset_dtype is only supported for ScatterToStriped")
    else:
        value_form = "out_of_place"
    core_spec = make_warp_exchange_spec(
        dtype=dtype,
        items_per_thread=items_per_thread,
        threads_in_warp=threads_in_warp,
        mode=mode,
        value_form=value_form,
        rank_dtype=offset_dtype,
    )
    specialization = NumbaMlirCoreAdapter().materialize(
        core_spec.specialization,
        extra_type_definitions=(numba_type_to_wrapper(dtype, methods=methods),),
    )

    return make_invocable_from_specialization(
        specialization, threads=threads_in_warp, block_threads=threads_per_block
    )


exchange = warp_exchange
