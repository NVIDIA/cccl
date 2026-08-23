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
    exchange,
    load,
    shuffle,
    store,
)
_register(
    "warp",
    warp_exchange,
    warp_load,
    warp_store,
)

__all__: tuple[str, ...] = ()
