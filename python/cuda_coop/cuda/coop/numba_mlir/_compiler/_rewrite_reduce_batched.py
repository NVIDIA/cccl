# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Payload checks for batched warp reduction providers."""

from ._parameters import _validate_common_numeric_dtype
from ._rewrite_support import CoopSinglePhaseRewriteError, _dtype_values_match


def infer_reduce_batched_payload(context, inference):
    input_var, input_spec = inference.array_candidate(0)
    output_var, output_spec = inference.array_candidate(1)
    if (
        input_spec is None
        or input_spec.items_per_thread is None
        or output_spec is None
        or output_spec.items_per_thread is None
    ):
        raise CoopSinglePhaseRewriteError(
            "reduce_batched requires fixed-size input and output payloads"
        )
    batches = input_spec.items_per_thread
    width = inference.factory_value("threads_in_warp")
    if not isinstance(width, int) or isinstance(width, bool) or width < 1:
        raise CoopSinglePhaseRewriteError("reduce_batched requires a static warp width")
    if output_spec.items_per_thread != (batches + width - 1) // width:
        raise CoopSinglePhaseRewriteError(
            "reduce_batched result extent does not match batches"
        )
    dtype = inference.inferred_array_dtype(input_var, input_spec)
    if dtype is None:
        dtype = inference.factory_value("dtype")
    dtype = _validate_common_numeric_dtype(dtype, operation="reduce_batched")
    output_dtype = inference.inferred_array_dtype(output_var, output_spec)
    if output_dtype is not None and not _dtype_values_match(dtype, output_dtype):
        raise CoopSinglePhaseRewriteError(
            "reduce_batched input and output dtypes must match"
        )
    inference.infer_kwarg("dtype", dtype)
    inference.infer_kwarg("batches", batches)
    context.record_thread_data_dtype(input_var, dtype)
    context.record_thread_data_dtype(output_var, dtype)


__all__ = ["infer_reduce_batched_payload"]
