# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Primitive-family lowering factories for Numba-CUDA-MLIR.

Each semantic module owns both block and warp routes where the primitive has
both.  This package registers exact factory identities for the compiler; it
does not recognize providers by their module or function names.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .._compiler._operations import register_factory
from ._adjacent_difference import (  # noqa: F401
    BlockAdjacentDifferenceType,
    adjacent_difference,
)
from ._discontinuity import (  # noqa: F401
    BlockDiscontinuityType,
    discontinuity,
)
from ._exchange import (  # noqa: F401
    BlockExchangeType,
    WarpExchangeType,
    exchange,
    warp_exchange,
)
from ._histogram import _group_histogram
from ._load_store import load, store, warp_load, warp_store
from ._merge_sort import (
    merge_sort_keys,
    merge_sort_pairs,
    warp_merge_sort_keys,
    warp_merge_sort_pairs,
)
from ._radix import (
    _common_radix_rank,
    _common_radix_sort_keys,
    _common_radix_sort_pairs,
    radix_rank,
    radix_sort_keys,
    radix_sort_keys_descending,
    radix_sort_pairs,
    radix_sort_pairs_descending,
)
from ._reduce import (
    block_reduce_builtin,
    group_reduce,
    reduce,
    sum,
    warp_reduce,
    warp_reduce_builtin,
    warp_sum,
)
from ._run_length_decode import _group_run_length_decode
from ._scan import (
    scan,
    warp_exclusive_scan,
    warp_exclusive_sum,
    warp_inclusive_scan,
    warp_inclusive_sum,
)
from ._shuffle import BlockShuffleType, shuffle  # noqa: F401
from ._topk import (
    _common_topk_max_keys,
    _common_topk_max_pairs,
    _common_topk_min_keys,
    _common_topk_min_pairs,
    _qualified_group_topk_max_keys,
    _qualified_group_topk_max_pairs,
    _qualified_group_topk_min_keys,
    _qualified_group_topk_min_pairs,
    topk_max_keys,
    topk_max_pairs,
    topk_min_keys,
    topk_min_pairs,
)


def _register(namespace: str, *factories: Callable[..., Any]) -> None:
    for factory in factories:
        register_factory(
            factory,
            operation=factory.__name__,
            namespace=namespace,
        )


_register(
    "block",
    adjacent_difference,
    block_reduce_builtin,
    discontinuity,
    exchange,
    load,
    merge_sort_keys,
    merge_sort_pairs,
    radix_rank,
    radix_sort_keys,
    radix_sort_keys_descending,
    radix_sort_pairs,
    radix_sort_pairs_descending,
    reduce,
    scan,
    shuffle,
    store,
    sum,
    topk_max_keys,
    topk_max_pairs,
    topk_min_keys,
    topk_min_pairs,
    _common_radix_rank,
    _common_radix_sort_keys,
    _common_radix_sort_pairs,
    _common_topk_max_keys,
    _common_topk_max_pairs,
    _common_topk_min_keys,
    _common_topk_min_pairs,
    _qualified_group_topk_max_keys,
    _qualified_group_topk_max_pairs,
    _qualified_group_topk_min_keys,
    _qualified_group_topk_min_pairs,
    _group_histogram,
    _group_run_length_decode,
)
_register(
    "group",
    group_reduce,
)
_register(
    "warp",
    warp_exchange,
    warp_exclusive_scan,
    warp_exclusive_sum,
    warp_inclusive_scan,
    warp_inclusive_sum,
    warp_load,
    warp_merge_sort_keys,
    warp_merge_sort_pairs,
    warp_reduce,
    warp_reduce_builtin,
    warp_store,
    warp_sum,
)

__all__: tuple[str, ...] = ()
