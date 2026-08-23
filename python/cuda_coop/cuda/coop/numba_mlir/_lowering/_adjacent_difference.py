# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Adjacent-difference provider lowering for Numba-CUDA-MLIR.

This semantic module owns the physical-block CUB provider.  The group planner
owns hierarchy validation, payload provenance, and runtime-argument rewriting.
"""

import operator
from enum import IntEnum, auto

from .._compiler._activation import _require_runtime

_require_runtime()

from cuda.coop._core import Dependency, PythonOperator
from cuda.coop._core.block import (
    BlockAdjacentDifferenceDirection,
    make_block_adjacent_difference_spec,
)

from .._compiler._parameters import (
    normalize_dim_param,
    normalize_dtype_param,
)
from .._types import make_invocable_from_specialization, numba_type_to_wrapper
from ._core import NumbaMlirCoreAdapter


class BlockAdjacentDifferenceType(IntEnum):
    """Planner direction for block adjacent difference."""

    SubtractLeft = auto()
    SubtractRight = auto()


def _normalize_adjacent_difference_type(block_adjacent_difference_type):
    if isinstance(block_adjacent_difference_type, int):
        block_adjacent_difference_type = BlockAdjacentDifferenceType(
            block_adjacent_difference_type
        )
    if not isinstance(block_adjacent_difference_type, BlockAdjacentDifferenceType):
        raise ValueError(
            "block_adjacent_difference_type must be a valid "
            "BlockAdjacentDifferenceType enum value; "
            f"got: {block_adjacent_difference_type!r}"
        )
    return block_adjacent_difference_type


def adjacent_difference(
    block_adjacent_difference_type=BlockAdjacentDifferenceType.SubtractLeft,
    dtype=None,
    threads_per_block=None,
    items_per_thread=1,
    difference_op=None,
    methods=None,
    valid_items=None,
    tile_predecessor_item=None,
    tile_successor_item=None,
):
    """Build the direct CUB adjacent-difference invocable selected by planning."""
    block_adjacent_difference_type = _normalize_adjacent_difference_type(
        block_adjacent_difference_type
    )
    if dtype is None:
        raise ValueError("dtype must be provided")
    if threads_per_block is None:
        raise ValueError("threads_per_block must be provided")
    if isinstance(items_per_thread, bool):
        raise TypeError("items_per_thread must be an integer")
    try:
        items_per_thread = operator.index(items_per_thread)
    except TypeError as exc:
        raise TypeError("items_per_thread must be an integer") from exc
    if items_per_thread < 1:
        raise ValueError("items_per_thread must be a positive integer")
    if difference_op is None:
        raise ValueError("difference_op must be provided")
    if tile_predecessor_item is not None and tile_successor_item is not None:
        raise ValueError(
            "Only one of tile_predecessor_item or tile_successor_item may be set"
        )

    dim = normalize_dim_param(threads_per_block)
    dtype = normalize_dtype_param(dtype)

    if block_adjacent_difference_type == BlockAdjacentDifferenceType.SubtractLeft:
        if tile_successor_item is not None:
            raise ValueError("tile_successor_item is not valid for SubtractLeft")
        core_direction = BlockAdjacentDifferenceDirection.LEFT
    else:
        if tile_predecessor_item is not None:
            raise ValueError("tile_predecessor_item is not valid for SubtractRight")
        core_direction = BlockAdjacentDifferenceDirection.RIGHT

    core_spec = make_block_adjacent_difference_spec(
        dtype=dtype,
        block_dim=tuple(dim),
        items_per_thread=items_per_thread,
        direction=core_direction,
        difference_operator=PythonOperator(
            ret_dtype=Dependency("T"),
            arg_dtypes=(Dependency("T"), Dependency("T")),
            op=difference_op,
            name="difference_op",
        ),
        valid_items=valid_items,
        tile_predecessor_item=tile_predecessor_item,
        tile_successor_item=tile_successor_item,
    )

    specialization = NumbaMlirCoreAdapter().materialize(
        core_spec.specialization,
        extra_type_definitions=(numba_type_to_wrapper(dtype, methods=methods),),
    )
    return make_invocable_from_specialization(specialization)
