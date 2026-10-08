# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reconcile MergeSort payload types before provider specialization.

The group rewrite creates result arrays that CUB will sort in place. This
hook checks their fixed extents and records separate key and value dtypes
for allocation and factory arguments. Partial-tile scalar controls follow
the arrays and are prepared by the group rewrite.
"""

from ._parameters import _validate_common_numeric_dtype
from ._rewrite_support import CoopSinglePhaseRewriteError


def infer_merge_sort_payload(context, inference):
    """Infer matching extents and independent key and value dtypes.

    Inspect one array for keys-only sorting or two for pairs. Require fixed,
    equal extents, but infer each numeric dtype independently. Use an explicit
    factory dtype when payload provenance cannot provide one, and reconcile
    inferred keywords with any existing factory arguments.

    Record each dtype on its ThreadData payload so allocation and provider
    specialization agree. Partial-tile controls do not affect these shapes.
    """

    names = ("keys", "values") if "pairs" in inference.op_name else ("keys",)
    extent = None
    for index, name in enumerate(names):
        value, specification = inference.array_candidate(index)
        if (
            value is None
            or specification is None
            or specification.items_per_thread is None
        ):
            raise CoopSinglePhaseRewriteError(
                f"Merge Sort {name} must have a fixed array extent"
            )
        if extent is not None and extent != specification.items_per_thread:
            raise CoopSinglePhaseRewriteError(
                "Merge Sort keys and values must have matching extents"
            )
        extent = specification.items_per_thread
        dtype_name = "key_dtype" if index == 0 else "value_dtype"
        dtype = inference.inferred_array_dtype(value, specification)
        if dtype is None:
            dtype = inference.factory_value(dtype_name)
        dtype = _validate_common_numeric_dtype(
            dtype, operation=inference.op_name, parameter=name
        )
        inference.infer_kwarg(dtype_name, dtype)
        context.record_thread_data_dtype(value, dtype)
    inference.infer_kwarg("items_per_thread", extent)


__all__: tuple[str, ...] = ()
