# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Merge Sort payload inference for copied in-place provider operands."""

from ._parameters import _validate_common_numeric_dtype
from ._rewrite_support import CoopSinglePhaseRewriteError


def infer_merge_sort_payload(context, inference):
    names = ("keys", "values") if "pairs" in inference.op_name else ("keys",)
    extent = None
    for index, name in enumerate(names):
        value, spec = inference.array_candidate(index)
        if value is None or spec is None or spec.items_per_thread is None:
            raise CoopSinglePhaseRewriteError(
                f"Merge Sort {name} must have a fixed array extent"
            )
        if extent is not None and extent != spec.items_per_thread:
            raise CoopSinglePhaseRewriteError(
                "Merge Sort keys and values must have matching extents"
            )
        extent = spec.items_per_thread
        dtype_name = "key_dtype" if index == 0 else "value_dtype"
        dtype = inference.inferred_array_dtype(value, spec)
        if dtype is None:
            dtype = inference.factory_value(dtype_name)
        dtype = _validate_common_numeric_dtype(
            dtype, operation=inference.op_name, parameter=name
        )
        inference.infer_kwarg(dtype_name, dtype)
        context.record_thread_data_dtype(value, dtype)
    inference.infer_kwarg("items_per_thread", extent)


__all__: tuple[str, ...] = ()
