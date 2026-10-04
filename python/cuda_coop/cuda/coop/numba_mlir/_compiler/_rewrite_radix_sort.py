# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check radix provider arrays and record their allocation dtypes.

Scalar inputs have already been boxed by group planning. Sort providers
receive copied keys and optional values; rank providers receive input keys
and a distinct int32 result array. These roles determine which factory dtype
each operand must match before specialization.
"""

import numba_cuda_mlir.numba_cuda.types as numba_types

from ._rewrite_support import CoopSinglePhaseRewriteError


def infer_radix_payload(context, inference):
    """Reconcile radix payload extents and the dtype of each provider operand.

    Require one common fixed extent for keys, associated values, and rank
    results. Key and value arrays retain independent factory dtypes; the rank
    result is always int32. Use factory metadata when a payload's dtype is
    unknown, reject conflicts, and record the resolved type for allocation.

    An optional rank digit-prefix array also receives int32 dtype. Its extent
    was checked during group planning and need not match the key item count.
    """

    rank = inference.op_name == "radix_rank_keys"
    pairs = inference.op_name == "radix_sort_pairs"
    count = 2 if rank or pairs else 1
    extent = inference.factory_value("items_per_thread")
    for index in range(count):
        value, specification = inference.array_candidate(index)
        if specification is None or specification.items_per_thread is None:
            raise CoopSinglePhaseRewriteError(
                "radix operations require fixed-size array payloads"
            )
        if extent is None:
            extent = specification.items_per_thread
        if specification.items_per_thread != extent:
            raise CoopSinglePhaseRewriteError(
                "radix payloads must have the same items_per_thread"
            )
        dtype = inference.inferred_array_dtype(value, specification)
        parameter = "value_dtype" if index == 1 and pairs else "dtype"
        expected = (
            numba_types.int32
            if index == 1 and rank
            else inference.factory_value(parameter)
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
        value, specification = inference.array_candidate(2)
        dtype = inference.inferred_array_dtype(value, specification)
        if specification is None or dtype not in {None, numba_types.int32}:
            raise CoopSinglePhaseRewriteError(
                "exclusive_digit_prefix must be an int32 array"
            )
        context.record_thread_data_dtype(value, numba_types.int32)


__all__ = ["infer_radix_payload"]
