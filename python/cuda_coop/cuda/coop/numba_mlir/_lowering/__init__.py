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
from ._exchange import BlockExchangeType as BlockExchangeType
from ._exchange import WarpExchangeType as WarpExchangeType
from ._exchange import exchange as exchange
from ._exchange import warp_exchange as warp_exchange
from ._load_store import load as load
from ._load_store import store as store
from ._load_store import warp_load as warp_load
from ._load_store import warp_store as warp_store
from ._merge_sort import merge_sort_keys as merge_sort_keys
from ._merge_sort import merge_sort_pairs as merge_sort_pairs
from ._merge_sort import warp_merge_sort_keys as warp_merge_sort_keys
from ._merge_sort import warp_merge_sort_pairs as warp_merge_sort_pairs
from ._reduce import block_reduce_builtin as block_reduce_builtin
from ._reduce import group_reduce as group_reduce
from ._reduce import reduce as reduce
from ._reduce import sum as sum
from ._reduce import warp_reduce as warp_reduce
from ._reduce import warp_reduce_builtin as warp_reduce_builtin
from ._reduce import warp_sum as warp_sum
from ._scan import scan as scan
from ._scan import warp_exclusive_scan as warp_exclusive_scan
from ._scan import warp_exclusive_sum as warp_exclusive_sum
from ._scan import warp_inclusive_scan as warp_inclusive_scan
from ._scan import warp_inclusive_sum as warp_inclusive_sum
from ._shuffle import BlockShuffleType as BlockShuffleType
from ._shuffle import shuffle as shuffle


def _register(namespace: str, *factories: Callable[..., Any]) -> None:
    for factory in factories:
        register_factory(
            factory,
            operation=factory.__name__,
            namespace=namespace,
        )


_register(
    "block",
    block_reduce_builtin,
    exchange,
    load,
    merge_sort_keys,
    merge_sort_pairs,
    reduce,
    scan,
    shuffle,
    store,
    sum,
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
