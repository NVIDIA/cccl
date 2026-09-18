# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Validate radix payload extents and types before generating a provider."""

from numba_cuda_mlir import types

from ._rewrite_support import CoopSinglePhaseRewriteError


def infer_radix_payload(context, inference):
    rank = inference.op_name == "radix_rank"
    pairs = inference.op_name == "radix_sort_pairs"
    count = 2 if rank or pairs else 1
    extent = inference.factory_value("items_per_thread")
    for index in range(count):
        value, spec = inference.array_candidate(index)
        if spec is None or spec.items_per_thread is None:
            raise CoopSinglePhaseRewriteError(
                "radix operations require fixed-size array payloads"
            )
        if extent is None:
            extent = spec.items_per_thread
        if spec.items_per_thread != extent:
            raise CoopSinglePhaseRewriteError(
                "radix payloads must have the same items_per_thread"
            )
        dtype = inference.inferred_array_dtype(value, spec)
        parameter = "value_dtype" if index == 1 and pairs else "dtype"
        expected = (
            types.int32 if index == 1 and rank else inference.factory_value(parameter)
        )
        if dtype is None:
            dtype = expected
        if dtype is None or (expected is not None and dtype != expected):
            raise CoopSinglePhaseRewriteError(
                "radix payload dtype disagrees with its specialization"
            )
        if not (rank and index == 1):
            inference.infer_kwarg(parameter, dtype)
        context.record_thread_data_dtype(value, dtype)
    inference.infer_kwarg("items_per_thread", extent)
    if rank and len(inference.runtime_args) == 3:
        value, spec = inference.array_candidate(2)
        dtype = inference.inferred_array_dtype(value, spec)
        if spec is None or dtype not in {None, types.int32}:
            raise CoopSinglePhaseRewriteError(
                "exclusive_digit_prefix must be an int32 array"
            )
        context.record_thread_data_dtype(value, types.int32)


__all__ = ["infer_radix_payload"]
