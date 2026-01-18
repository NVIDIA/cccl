# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._bindings import _BINDINGS_AVAILABLE  # type: ignore[attr-defined]

if not _BINDINGS_AVAILABLE:
    __all__ = ["_BINDINGS_AVAILABLE"]

    def __getattr__(name):
        raise AttributeError(
            f"Cannot access 'cuda.compute.{name}' because CUDA bindings are not available."
            "This typically means you're running on a CPU-only machine without CUDA drivers installed."
        )
else:
    from ._caching import clear_all_caches
    from .algorithms import (
        DoubleBuffer,
        SortOrder,
        binary_transform,
        exclusive_scan,
        histogram_even,
        inclusive_scan,
        make_binary_transform,
        make_exclusive_scan,
        make_histogram_even,
        make_inclusive_scan,
        make_merge_sort,
        make_radix_sort,
        make_reduce_into,
        make_segmented_reduce,
        make_segmented_sort,
        make_select,
        make_three_way_partition,
        make_unary_transform,
        make_unique_by_key,
        merge_sort,
        radix_sort,
        reduce_into,
        segmented_reduce,
        segmented_sort,
        select,
        three_way_partition,
        unary_transform,
        unique_by_key,
    )
    from .determinism import Determinism
    from .iterators import (
        CacheModifiedInputIterator,
        ConstantIterator,
        CountingIterator,
        DiscardIterator,
        PermutationIterator,
        ReverseIterator,
        TransformIterator,
        TransformOutputIterator,
        ZipIterator,
    )
    from .op import OpKind
    from .struct import gpu_struct

    __all__ = [
        "_BINDINGS_AVAILABLE",
        "binary_transform",
        "clear_all_caches",
        "CacheModifiedInputIterator",
        "ConstantIterator",
        "CountingIterator",
        "DiscardIterator",
        "DoubleBuffer",
        "exclusive_scan",
        "gpu_struct",
        "histogram_even",
        "inclusive_scan",
        "make_binary_transform",
        "make_exclusive_scan",
        "make_select",
        "make_histogram_even",
        "make_inclusive_scan",
        "make_merge_sort",
        "make_radix_sort",
        "make_reduce_into",
        "make_segmented_reduce",
        "make_segmented_sort",
        "make_three_way_partition",
        "make_unary_transform",
        "make_unique_by_key",
        "merge_sort",
        "OpKind",
        "Determinism",
        "PermutationIterator",
        "radix_sort",
        "reduce_into",
        "ReverseIterator",
        "segmented_reduce",
        "segmented_sort",
        "select",
        "SortOrder",
        "TransformIterator",
        "TransformOutputIterator",
        "three_way_partition",
        "unary_transform",
        "unique_by_key",
        "ZipIterator",
    ]
