# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shuffle provider lowering for Numba-CUDA-MLIR.

This module owns CUB shuffle materialization; group validation and typed fresh
result construction remain in compiler planning.
"""

import operator
from enum import IntEnum, auto

from cuda.coop._core import ArgumentBinding
from cuda.coop._core.block import BlockShuffleMode, make_block_shuffle_spec

from .._compiler._parameters import (
    normalize_dim_param,
    normalize_dtype_param,
)
from .._types import make_invocable_from_specialization, numba_type_to_wrapper
from ._core import NumbaMlirCoreAdapter


class BlockShuffleType(IntEnum):
    """CUB block-shuffle movement patterns."""

    Offset = auto()
    Rotate = auto()
    Up = auto()
    Down = auto()


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


def _integer(value, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


def _normalize_shuffle_type(value) -> BlockShuffleType:
    if isinstance(value, bool):
        raise TypeError("block_shuffle_type must not be bool")
    if isinstance(value, BlockShuffleType):
        return value
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(
            "block_shuffle_type must be a BlockShuffleType or integer"
        ) from exc
    try:
        return BlockShuffleType(value)
    except ValueError as exc:
        raise ValueError(f"invalid BlockShuffleType value: {value!r}") from exc


def shuffle(
    block_shuffle_type=BlockShuffleType.Up,
    dtype=None,
    threads_per_block=None,
    items_per_thread=None,
    distance=None,
    block_prefix=None,
    block_suffix=None,
    methods=None,
):
    """Build the block-shuffle invocable selected by movement planning."""

    if dtype is None:
        raise ValueError("dtype must be provided")
    if threads_per_block is None:
        raise ValueError("threads_per_block must be provided")
    block_shuffle_type = _normalize_shuffle_type(block_shuffle_type)
    block_dim = normalize_dim_param(threads_per_block)
    dtype = normalize_dtype_param(dtype)
    core_mode = BlockShuffleMode.from_cub_method_name(block_shuffle_type.name)
    array_form = (
        block_shuffle_type in (BlockShuffleType.Up, BlockShuffleType.Down)
        and items_per_thread is not None
    )

    if array_form:
        items_per_thread = _positive_int(
            items_per_thread,
            name="items_per_thread",
        )
        if distance is not None:
            raise ValueError("distance is not supported for Up/Down array shuffles")
        if block_shuffle_type is BlockShuffleType.Up and block_prefix is not None:
            raise ValueError("block_prefix is not valid for Up shuffles")
        if block_shuffle_type is BlockShuffleType.Down and block_suffix is not None:
            raise ValueError("block_suffix is not valid for Down shuffles")
        distance_binding = ArgumentBinding.omitted()
    else:
        if items_per_thread is not None:
            raise ValueError("items_per_thread is valid only for Up/Down shuffles")
        if block_prefix is not None or block_suffix is not None:
            raise ValueError(
                "block_prefix/block_suffix require an Up/Down array shuffle"
            )
        if distance is None:
            distance = 1
        distance = _integer(distance, name="distance")
        if block_shuffle_type is BlockShuffleType.Up:
            distance = -abs(distance)
            core_mode = BlockShuffleMode.OFFSET
        elif block_shuffle_type is BlockShuffleType.Down:
            distance = abs(distance)
            core_mode = BlockShuffleMode.OFFSET
        distance_binding = ArgumentBinding.static(distance)

    core_spec = make_block_shuffle_spec(
        dtype=dtype,
        block_dim=tuple(block_dim),
        mode=core_mode,
        items_per_thread=items_per_thread if array_form else None,
        distance=distance_binding,
        block_prefix=block_prefix is not None,
        block_suffix=block_suffix is not None,
    )
    specialization = NumbaMlirCoreAdapter().materialize(
        core_spec.specialization,
        extra_type_definitions=(numba_type_to_wrapper(dtype, methods=methods),),
    )
    return make_invocable_from_specialization(specialization)
