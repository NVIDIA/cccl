# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Planner-private warp exchange provider."""

import operator
from enum import IntEnum, auto

from numba_cuda_mlir import types

from cuda.coop._core.warp import make_warp_exchange_spec

from .._common import normalize_dtype_param
from .._core_adapter import NumbaMlirCoreAdapter
from .._types import make_invocable_from_specialization, numba_type_to_wrapper


class WarpExchangeType(IntEnum):
    """CUB warp-exchange data movement patterns."""

    StripedToBlocked = auto()
    BlockedToStriped = auto()
    ScatterToStriped = auto()


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


def _normalize_exchange_type(value) -> WarpExchangeType:
    if isinstance(value, bool):
        raise TypeError("warp_exchange_type must not be bool")
    if isinstance(value, WarpExchangeType):
        return value
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(
            "warp_exchange_type must be a WarpExchangeType or integer"
        ) from exc
    try:
        return WarpExchangeType(value)
    except ValueError as exc:
        raise ValueError(f"invalid WarpExchangeType value: {value!r}") from exc


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
    """Build the warp-exchange invocable selected by movement planning."""

    items_per_thread = _positive_int(items_per_thread, name="items_per_thread")
    threads_in_warp = _positive_int(threads_in_warp, name="threads_in_warp")
    if use_output_items is not None and not isinstance(use_output_items, bool):
        raise TypeError("use_output_items must be a bool or None")
    warp_exchange_type = _normalize_exchange_type(warp_exchange_type)
    dtype = normalize_dtype_param(dtype)
    mode = {
        WarpExchangeType.StripedToBlocked: "striped_to_blocked",
        WarpExchangeType.BlockedToStriped: "blocked_to_striped",
        WarpExchangeType.ScatterToStriped: "scatter_to_striped",
    }[warp_exchange_type]

    if warp_exchange_type is WarpExchangeType.ScatterToStriped:
        if offset_dtype is None:
            offset_dtype = types.int32
        offset_dtype = normalize_dtype_param(offset_dtype)
        value_form = {
            None: "both",
            True: "out_of_place",
            False: "in_place",
        }[use_output_items]
    elif use_output_items is not None:
        raise ValueError("use_output_items is valid only for ScatterToStriped")
    elif offset_dtype is not None:
        raise ValueError("offset_dtype is valid only for ScatterToStriped")
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
        specialization,
        threads=threads_in_warp,
        block_threads=threads_per_block,
    )
