# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Load/Store selectors shared by group planning and lowering."""

from enum import Enum

from cuda.coop._core.block.load_store import BlockLoadStoreAlgorithm
from cuda.coop._core.group.load_store import (
    _STORAGE_FREE_ALGORITHMS as _CORE_STORAGE_FREE_ALGORITHMS,
)
from cuda.coop._core.warp.load_store import WarpLoadStoreAlgorithm

_BLOCK_LOAD_STORE_ALGORITHMS = frozenset(
    item.value for item in BlockLoadStoreAlgorithm
)
_WARP_LOAD_STORE_ALGORITHMS = frozenset(
    item.value for item in WarpLoadStoreAlgorithm
)
_STORAGE_FREE_ALGORITHMS = frozenset(
    item.value for item in _CORE_STORAGE_FREE_ALGORITHMS
)


def _resolve_algorithm(
    algorithm, allowed_algorithms, primitive_name: str
) -> str:
    if not isinstance(algorithm, str) or isinstance(algorithm, Enum):
        raise TypeError(f"{primitive_name} algorithm must be a string")
    token = algorithm.strip().lower().replace("-", "_")
    if token in allowed_algorithms:
        return token
    choices = ", ".join(sorted(allowed_algorithms))
    raise ValueError(
        f"Unsupported {primitive_name} algorithm {algorithm!r}; expected one "
        f"of: {choices}"
    )
