# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Infer and check the unequal array extents of a batched-reduce provider.

Input extent is the batch count. Output extent is the per-lane share of the
aggregates, rounded up for batches that do not fill the warp. Both arrays
must use the same numeric dtype; the rewrite records that type for subsequent
ThreadData accesses as well as for provider specialization.
"""

from ._parameters import _validate_common_numeric_dtype
from ._rewrite_support import CoopSinglePhaseRewriteError, _dtype_values_match


def infer_reduce_batched_payload(context, inference):
    """Match fixed provider arrays to the batch count and static warp width.

    Take the batch count from the input extent and require an output extent
    of ceil(batches / width). Take the input dtype from array evidence or the
    explicit dtype keyword, and check any known output dtype against it. Fill
    the dtype and batches factory keywords, or reject explicit values that
    disagree with the arrays. Record the dtype on both operands so later
    indexing does not depend on type evidence from another call.
    """

    input_var, input_specification = inference.array_candidate(0)
    output_var, output_specification = inference.array_candidate(1)
    if (
        input_specification is None
        or input_specification.items_per_thread is None
        or output_specification is None
        or output_specification.items_per_thread is None
    ):
        raise CoopSinglePhaseRewriteError(
            "reduce_batched requires fixed-size input and output payloads"
        )
    batches = input_specification.items_per_thread
    width = inference.factory_value("threads_in_warp")
    if not isinstance(width, int) or isinstance(width, bool) or width < 1:
        raise CoopSinglePhaseRewriteError(
            "reduce_batched requires a static warp width"
        )
    if output_specification.items_per_thread != (batches + width - 1) // width:
        raise CoopSinglePhaseRewriteError(
            "reduce_batched result extent does not match batches"
        )
    dtype = inference.inferred_array_dtype(input_var, input_specification)
    if dtype is None:
        dtype = inference.factory_value("dtype")
    dtype = _validate_common_numeric_dtype(dtype, operation="reduce_batched")
    output_dtype = inference.inferred_array_dtype(
        output_var, output_specification
    )
    if output_dtype is not None and not _dtype_values_match(
        dtype, output_dtype
    ):
        raise CoopSinglePhaseRewriteError(
            "reduce_batched input and output dtypes must match"
        )
    inference.infer_kwarg("dtype", dtype)
    inference.infer_kwarg("batches", batches)
    context.record_thread_data_dtype(input_var, dtype)
    context.record_thread_data_dtype(output_var, dtype)


__all__ = ["infer_reduce_batched_payload"]
