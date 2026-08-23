# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Planner-private fused BlockRunLengthDecode provider."""

from .. import _require_runtime

_require_runtime()

from cuda.coop._core.block import (
    BlockRunLengthDecodeStage,
    make_block_run_length_decode_spec,
)

from .._common import normalize_dim_param, normalize_dtype_param
from .._core_adapter import NumbaMlirCoreAdapter
from .._types import make_invocable_from_specialization


def _group_run_length_decode(
    item_dtype,
    run_length_dtype,
    decoded_offset_dtype,
    total_decoded_size_dtype,
    threads_per_block,
    runs_per_thread,
    decoded_items_per_thread,
    with_relative_offsets=False,
    relative_offset_dtype=None,
):
    """Build one compiler-private fused BlockRunLengthDecode invocable."""

    dim = normalize_dim_param(threads_per_block)
    item_dtype = normalize_dtype_param(item_dtype)
    run_length_dtype = normalize_dtype_param(run_length_dtype)
    decoded_offset_dtype = normalize_dtype_param(decoded_offset_dtype)
    total_decoded_size_dtype = normalize_dtype_param(total_decoded_size_dtype)
    if relative_offset_dtype is not None:
        relative_offset_dtype = normalize_dtype_param(relative_offset_dtype)
    core_spec = make_block_run_length_decode_spec(
        item_dtype=item_dtype,
        run_length_dtype=run_length_dtype,
        decoded_offset_dtype=decoded_offset_dtype,
        total_decoded_size_dtype=total_decoded_size_dtype,
        block_dim=tuple(dim),
        runs_per_thread=runs_per_thread,
        decoded_items_per_thread=decoded_items_per_thread,
        stage=BlockRunLengthDecodeStage.FUSED,
        with_relative_offsets=with_relative_offsets,
        relative_offset_dtype=relative_offset_dtype,
        with_decoded_window_offset=True,
        returns_total_decoded_size=True,
    )
    specialization = NumbaMlirCoreAdapter().materialize(core_spec.specialization)
    return make_invocable_from_specialization(specialization)


__all__: tuple[str, ...] = ()
