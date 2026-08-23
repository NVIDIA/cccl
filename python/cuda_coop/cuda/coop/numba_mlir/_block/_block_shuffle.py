# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from enum import IntEnum, auto

from .. import _require_runtime

_require_runtime()

from cuda.coop._core import ArgumentBinding
from cuda.coop._core.block import BlockShuffleMode, make_block_shuffle_spec

from .._common import (
    normalize_dim_param,
    normalize_dtype_param,
    resolve_threads_per_block_alias,
)
from .._core_adapter import NumbaMlirCoreAdapter
from .._types import make_invocable_from_specialization, numba_type_to_wrapper


class BlockShuffleType(IntEnum):
    """CUB block-shuffle movement pattern."""

    Offset = auto()
    Rotate = auto()
    Up = auto()
    Down = auto()


def _normalize_shuffle_type(block_shuffle_type):
    if isinstance(block_shuffle_type, int):
        block_shuffle_type = BlockShuffleType(block_shuffle_type)
    if not isinstance(block_shuffle_type, BlockShuffleType):
        raise ValueError(
            "block_shuffle_type must be a valid BlockShuffleType enum value; "
            f"got: {block_shuffle_type!r}"
        )
    return block_shuffle_type


def shuffle(
    block_shuffle_type=BlockShuffleType.Up,
    dtype=None,
    threads_per_block=None,
    items_per_thread=None,
    distance=None,
    block_prefix=None,
    block_suffix=None,
    methods=None,
    dim=None,
):
    """Build a block-wide shuffle invocable.

    Scalar shuffles use CUB ``BlockShuffle`` offset or rotate operations.
    Up/down array shuffles operate on per-thread item arrays and may exchange
    boundary items through ``block_prefix`` or ``block_suffix``.
    """
    if dtype is None and not isinstance(block_shuffle_type, (BlockShuffleType, int)):
        dtype = block_shuffle_type
        block_shuffle_type = BlockShuffleType.Up

    threads_per_block = resolve_threads_per_block_alias(threads_per_block, dim)
    block_shuffle_type = _normalize_shuffle_type(block_shuffle_type)
    if dtype is None:
        raise ValueError("dtype must be provided")
    if threads_per_block is None:
        raise ValueError("threads_per_block must be provided")

    dim = normalize_dim_param(threads_per_block)
    dtype = normalize_dtype_param(dtype)
    core_mode = BlockShuffleMode.from_cub_method_name(block_shuffle_type.name)

    use_array_inputs = (
        block_shuffle_type in (BlockShuffleType.Up, BlockShuffleType.Down)
        and items_per_thread is not None
    )

    if use_array_inputs:
        if items_per_thread < 1:
            raise ValueError("items_per_thread must be >= 1 for Up/Down shuffles")
        if distance is not None:
            raise ValueError("distance is not supported for Up/Down array shuffles")
        if block_shuffle_type == BlockShuffleType.Up and block_prefix is not None:
            raise ValueError("block_prefix is not valid for Up shuffles")
        if block_shuffle_type == BlockShuffleType.Down and block_suffix is not None:
            raise ValueError("block_suffix is not valid for Down shuffles")

        distance_binding = ArgumentBinding.omitted()
    else:
        if items_per_thread is not None and block_shuffle_type not in (
            BlockShuffleType.Up,
            BlockShuffleType.Down,
        ):
            raise ValueError("items_per_thread is only valid for Up/Down shuffles")
        if block_prefix is not None or block_suffix is not None:
            raise ValueError(
                "block_prefix/block_suffix are only valid for Up/Down array shuffles"
            )

        if distance is None:
            distance = 1
        if block_shuffle_type == BlockShuffleType.Up:
            distance = -abs(int(distance))
            core_mode = BlockShuffleMode.OFFSET
        elif block_shuffle_type == BlockShuffleType.Down:
            distance = abs(int(distance))
            core_mode = BlockShuffleMode.OFFSET
        distance_binding = ArgumentBinding.static(distance)

    core_spec = make_block_shuffle_spec(
        dtype=dtype,
        block_dim=tuple(dim),
        mode=core_mode,
        items_per_thread=items_per_thread if use_array_inputs else None,
        distance=distance_binding,
        block_prefix=block_prefix is not None,
        block_suffix=block_suffix is not None,
    )
    specialization = NumbaMlirCoreAdapter().materialize(
        core_spec.specialization,
        extra_type_definitions=(numba_type_to_wrapper(dtype, methods=methods),),
    )
    return make_invocable_from_specialization(specialization)
