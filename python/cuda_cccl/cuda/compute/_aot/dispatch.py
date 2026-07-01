# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Public, free-standing ahead-of-time (AoT) serialize/deserialize entry points.

The per-algorithm builder classes (``_Reduce``, ``_Scan``, ...) are private, so
their ``serialize``/``deserialize`` are exposed here as module-level functions.
``deserialize`` dispatches on the algorithm tag embedded in the blob, so a single
function reconstructs any built algorithm with no objects supplied.
"""

from __future__ import annotations

from typing import Any, Callable

from ..algorithms._binary_search import _ALGO_BINARY_SEARCH, _BinarySearch
from ..algorithms._histogram import _ALGO_HISTOGRAM, _Histogram
from ..algorithms._reduce import _ALGO_REDUCE, _Reduce
from ..algorithms._scan import _ALGO_SCAN, _Scan
from ..algorithms._segmented_reduce import _ALGO_SEGMENTED_REDUCE, _SegmentedReduce
from ..algorithms._sort._merge_sort import _ALGO_MERGE_SORT, _MergeSort
from ..algorithms._sort._radix_sort import _ALGO_RADIX_SORT, _RadixSort
from ..algorithms._sort._segmented_sort import _ALGO_SEGMENTED_SORT, _SegmentedSort
from ..algorithms._three_way_partition import (
    _ALGO_THREE_WAY_PARTITION,
    _ThreeWayPartition,
)
from ..algorithms._transform import (
    _ALGO_BINARY_TRANSFORM,
    _ALGO_UNARY_TRANSFORM,
    _BinaryTransform,
    _UnaryTransform,
)
from ..algorithms._unique_by_key import _ALGO_UNIQUE_BY_KEY, _UniqueByKey
from . import serde

# Algorithm tag -> reconstructor (each takes only the blob). binary_search's
# private _deserialize accepts the blob and validates nothing extra when called
# without an expected mode (the mode is read from the blob itself).
_DESERIALIZERS: dict[int, Callable[[bytes], Any]] = {
    _ALGO_REDUCE: _Reduce.deserialize,
    _ALGO_SCAN: _Scan.deserialize,
    _ALGO_SEGMENTED_REDUCE: _SegmentedReduce.deserialize,
    _ALGO_UNARY_TRANSFORM: _UnaryTransform.deserialize,
    _ALGO_BINARY_TRANSFORM: _BinaryTransform.deserialize,
    _ALGO_BINARY_SEARCH: _BinarySearch._deserialize,
    _ALGO_MERGE_SORT: _MergeSort.deserialize,
    _ALGO_RADIX_SORT: _RadixSort.deserialize,
    _ALGO_SEGMENTED_SORT: _SegmentedSort.deserialize,
    _ALGO_THREE_WAY_PARTITION: _ThreeWayPartition.deserialize,
    _ALGO_UNIQUE_BY_KEY: _UniqueByKey.deserialize,
    _ALGO_HISTOGRAM: _Histogram.deserialize,
}


def serialize(algorithm: Any) -> bytes:
    """Serialize a built algorithm into a self-contained AoT blob.

    Args:
        algorithm: An object returned by a ``make_*`` factory (e.g.
            :func:`make_reduce_into`, :func:`make_exclusive_scan`).

    Returns:
        A versioned, self-describing byte blob. Reconstruct it with
        :func:`deserialize` — no objects required at load time.
    """
    try:
        return algorithm.serialize()
    except AttributeError as e:
        raise TypeError(
            f"{type(algorithm).__name__} is not an AoT-serializable algorithm "
            "(expected an object from a make_* factory)."
        ) from e


def deserialize(blob: bytes):
    """Reconstruct a built algorithm from a blob produced by :func:`serialize`.

    Takes only the blob: the algorithm kind is read from the blob's header and
    its iterator/operator/value descriptors are rebuilt from the embedded
    sidecar, so no objects need to be supplied. The returned object is callable
    exactly like the one from the corresponding ``make_*`` factory.

    Raises:
        ValueError: if the blob is malformed or its algorithm tag is unknown.
    """
    tag = serde.peek_algo(blob)
    try:
        reconstruct = _DESERIALIZERS[tag]
    except KeyError:
        raise ValueError(f"AoT blob: unknown algorithm tag {tag}") from None
    return reconstruct(blob)
