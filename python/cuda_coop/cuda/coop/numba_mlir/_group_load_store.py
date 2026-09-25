# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group-first load and store markers for Numba-CUDA-MLIR.

This module owns public movement signatures.  Compiler planning and CUB
provider materialization live in ``_compiler`` and ``_lowering`` respectively.
"""

from __future__ import annotations

from typing import Any

from .._core.api import ThreadDataLike
from ._compiler._operations import group_operation
from ._group_marker import group_primitive_marker
from ._thread_group import ThreadGroup


@group_operation(
    "load",
    family_module="cuda.coop.numba_mlir._compiler._group_load_store",
)
def load(
    group: ThreadGroup,
    source: Any,
    output: ThreadDataLike[Any],
    /,
    *,
    algorithm: Any = "direct",
    valid_items: Any = None,
    oob_default: Any = None,
    offset: Any = None,
    temp_storage: Any = None,
) -> None:
    """Load a block or warp tile with the Numba-CUDA-MLIR backend.

    Parameters, algorithm choices, tile addressing, and the ``None`` return
    follow :func:`cuda.coop.load`. In this backend, ``output`` may also be a
    supported fixed-size Numba local array. It must be writable; an untyped
    local array can infer its dtype from the source.

    See :ref:`per-thread payloads <coop-thread-data>` and
    :ref:`temporary storage <coop-temp-storage>` for allocation rules.
    The executable example in :func:`cuda.coop.load` activates this backend
    explicitly and shows a guarded final tile.

    See Also
    --------
    :cpp:class:`cub::BlockLoad`, :cpp:class:`cub::WarpLoad`
        C++ Load primitives used for these group scopes.
    """

    group_primitive_marker(
        "load",
        group,
        source,
        output,
        algorithm=algorithm,
        valid_items=valid_items,
        oob_default=oob_default,
        offset=offset,
        temp_storage=temp_storage,
    )


@group_operation(
    "store",
    family_module="cuda.coop.numba_mlir._compiler._group_load_store",
)
def store(
    group: ThreadGroup,
    destination: Any,
    value: Any,
    /,
    *,
    algorithm: Any = "direct",
    valid_items: Any = None,
    offset: Any = None,
    temp_storage: Any = None,
) -> None:
    """Store a block or warp tile with the Numba-CUDA-MLIR backend.

    Parameters, algorithm choices, tile addressing, and the ``None`` return
    follow :func:`cuda.coop.store`. ``value`` may be a numeric scalar or a
    supported fixed-size Numba local array. Store preserves that input,
    including for transpose algorithms.

    See :ref:`per-thread payloads <coop-thread-data>` and
    :ref:`temporary storage <coop-temp-storage>` for allocation rules, and
    :func:`cuda.coop.store` for an executable partial-tile example.

    See Also
    --------
    :cpp:class:`cub::BlockStore`, :cpp:class:`cub::WarpStore`
        C++ Store primitives used for these group scopes.
    """

    group_primitive_marker(
        "store",
        group,
        destination,
        value,
        algorithm=algorithm,
        valid_items=valid_items,
        offset=offset,
        temp_storage=temp_storage,
    )


__all__ = ["load", "store"]
