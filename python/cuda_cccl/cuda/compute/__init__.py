# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from . import types
from ._caching import clear_all_caches
from ._iterators import (
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
from .compiled import CompiledIterator
from .determinism import Determinism
from .op import CompiledOp, OpKind
from .struct import gpu_struct

__all__ = [
    # BYOC (Bring Your Own Compiler) APIs - no Numba required
    "CompiledIterator",
    "CompiledOp",
    "types",
    # Algorithms
    "binary_transform",
    "clear_all_caches",
    "exclusive_scan",
    "histogram_even",
    "inclusive_scan",
    "make_binary_transform",
    "make_exclusive_scan",
    "make_histogram_even",
    "make_inclusive_scan",
    "make_merge_sort",
    "make_radix_sort",
    "make_reduce_into",
    "make_segmented_reduce",
    "make_segmented_sort",
    "make_select",
    "make_three_way_partition",
    "make_unary_transform",
    "make_unique_by_key",
    "merge_sort",
    "radix_sort",
    "reduce_into",
    "segmented_reduce",
    "segmented_sort",
    "select",
    "three_way_partition",
    "unary_transform",
    "unique_by_key",
    # Iterators (Numba-based)
    "CacheModifiedInputIterator",
    "ConstantIterator",
    "CountingIterator",
    "DiscardIterator",
    "PermutationIterator",
    "ReverseIterator",
    "TransformIterator",
    "TransformOutputIterator",
    "ZipIterator",
    # Operators
    "OpKind",
    # Sorting
    "DoubleBuffer",
    "SortOrder",
    # Determinism
    "Determinism",
    # Structs (Numba-based)
    "gpu_struct",
]
