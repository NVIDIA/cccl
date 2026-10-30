# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Keep Numba Load/Store selectors consistent with the shared core.

Allowed names come from the core algorithm enums. Factories use the
storage-free subset to check that an algorithm without shared scratch
uses the factory without a scratch pointer. Callers pass strings, which
are normalized; enum objects are rejected.
"""

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
    """Normalize a selector and check the primitive's supported choices.

    Ignore surrounding whitespace and letter case; treat hyphens as
    underscores. Reject enum objects even when they inherit from ``str`` so
    the API has one consistent selector form.
    """

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


__all__ = [
    "_BLOCK_LOAD_STORE_ALGORITHMS",
    "_STORAGE_FREE_ALGORITHMS",
    "_WARP_LOAD_STORE_ALGORITHMS",
    "_resolve_algorithm",
]
