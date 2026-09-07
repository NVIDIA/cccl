# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from enum import IntEnum, auto

from .. import _require_runtime

_require_runtime()

from numba_cuda_mlir import types

from cuda.coop._core import Dependency, PythonOperator
from cuda.coop._core.block import (
    BlockDiscontinuityMode,
    make_block_discontinuity_spec,
)

from .._common import (
    normalize_dim_param,
    normalize_dtype_param,
    resolve_threads_per_block_alias,
)
from .._core_adapter import NumbaMlirCoreAdapter
from .._types import make_invocable_from_specialization, numba_type_to_wrapper


class BlockDiscontinuityType(IntEnum):
    """Output mode for ``block.discontinuity`` flag generation."""

    HEADS = auto()
    TAILS = auto()
    HEADS_AND_TAILS = auto()


def _normalize_block_discontinuity_type(block_discontinuity_type):
    if isinstance(block_discontinuity_type, int):
        block_discontinuity_type = BlockDiscontinuityType(block_discontinuity_type)
    if not isinstance(block_discontinuity_type, BlockDiscontinuityType):
        raise ValueError(
            "block_discontinuity_type must be a valid BlockDiscontinuityType enum "
            f"value; got: {block_discontinuity_type!r}"
        )
    return block_discontinuity_type


def discontinuity(
    dtype=None,
    threads_per_block=None,
    items_per_thread=1,
    flag_op=None,
    flag_dtype=None,
    block_discontinuity_type=BlockDiscontinuityType.HEADS,
    methods=None,
    tile_predecessor_item=None,
    tile_successor_item=None,
    dim=None,
):
    """Build a block-wide discontinuity flagging invocable.

    The invocable wraps CUB ``BlockDiscontinuity`` and marks item boundaries
    according to ``flag_op``. ``block_discontinuity_type`` selects heads, tails,
    or both; ``flag_dtype`` controls the element type of the generated flag
    arrays.
    """
    threads_per_block = resolve_threads_per_block_alias(threads_per_block, dim)
    block_discontinuity_type = _normalize_block_discontinuity_type(
        block_discontinuity_type
    )
    if dtype is None:
        raise ValueError("dtype must be provided")
    if threads_per_block is None:
        raise ValueError("threads_per_block must be provided")
    if items_per_thread < 1:
        raise ValueError("items_per_thread must be greater than or equal to 1")
    if flag_op is None:
        raise ValueError("flag_op must be provided for block discontinuity")

    dim = normalize_dim_param(threads_per_block)
    dtype = normalize_dtype_param(dtype)
    if flag_dtype is None:
        flag_dtype = types.boolean
    flag_dtype = normalize_dtype_param(flag_dtype)

    if block_discontinuity_type == BlockDiscontinuityType.HEADS:
        core_mode = BlockDiscontinuityMode.HEADS
    elif block_discontinuity_type == BlockDiscontinuityType.TAILS:
        core_mode = BlockDiscontinuityMode.TAILS
    else:
        core_mode = BlockDiscontinuityMode.HEADS_AND_TAILS

    core_spec = make_block_discontinuity_spec(
        dtype=dtype,
        flag_dtype=flag_dtype,
        block_dim=tuple(dim),
        items_per_thread=items_per_thread,
        mode=core_mode,
        flag_operator=PythonOperator(
            ret_dtype=Dependency("FlagT"),
            arg_dtypes=(Dependency("T"), Dependency("T")),
            op=flag_op,
            name="flag_op",
        ),
        tile_predecessor_item=tile_predecessor_item,
        tile_successor_item=tile_successor_item,
    )

    specialization = NumbaMlirCoreAdapter().materialize(
        core_spec.specialization,
        extra_type_definitions=(numba_type_to_wrapper(dtype, methods=methods),),
    )
    return make_invocable_from_specialization(specialization)
