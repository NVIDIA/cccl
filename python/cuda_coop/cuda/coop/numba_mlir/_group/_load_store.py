# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Expose qualified Load and Store markers to Numba-CUDA-MLIR kernels.

The operation decorator identifies each call and its lowering family. The
whole-function planner replaces supported calls before device compilation.
Calling a marker directly in Python raises the ``RuntimeError`` from
``_marker.group_primitive_marker`` because only the planner can replace it.
"""

from __future__ import annotations

from typing import Any

try:
    import numpy
except ModuleNotFoundError as exc:
    if exc.name != "numpy":
        raise


from cuda.coop._typing import _CommonNumericT

from ..._core.api import ThreadDataLike
from ..._typing import (
    BlockLoadStoreAlgorithm,
    CommonThreadDataLike,
    IntegerValue,
    TempStorageLike,
    ValidItems,
    WarpLoadStoreAlgorithm,
)
from .._compiler._operations import group_operation
from .._thread_group import BlockGroup, WarpGroup
from ._marker import group_primitive_marker


@group_operation(
    "load",
    family_module="cuda.coop.numba_mlir._compiler._group_load_store",
)
def load(
    group: BlockGroup | WarpGroup,
    source: object,
    output: ThreadDataLike[_CommonNumericT] | numpy.ndarray | None = None,
    /,
    *,
    items_per_thread: int | None = None,
    alignment: int | None = None,
    algorithm: BlockLoadStoreAlgorithm | WarpLoadStoreAlgorithm = "direct",
    valid_items: ValidItems | None = None,
    oob_default: _CommonNumericT | float | None = None,
    offset: IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[Any] | None:
    """Load a block or warp tile with the Numba-CUDA-MLIR backend.

    Parameters, algorithm choices, tile addressing, and result ownership
    follow :func:`cuda.coop.load`. Prefer ``items_per_thread`` to return a
    fresh payload with the source dtype. Sources may be contiguous global
    arrays or shared arrays. The common API supports these memory sources too.

    To fill existing per-thread storage, pass ``output`` and omit
    ``items_per_thread`` and ``alignment``. This form returns ``None``.
    In this backend, ``output`` may also be a supported fixed-size,
    one-dimensional Numba local array. A directly constructed ``ThreadData``
    can infer its dtype from the source.

    To load directly into shared memory, pass a native contiguous shared
    array as ``output`` together with ``items_per_thread``. Each physical
    thread writes its own consecutive row of that many elements. The array
    must have a statically known capacity of at least the enclosing block's
    thread count times ``items_per_thread``; a contiguous slice selects the
    destination origin. Use ``shared[start:stop]``; explicit slice steps,
    including one, are unsupported by this backend. ``offset`` still selects
    the source origin. This form returns ``None`` and does not accept
    ``alignment``. Synchronize before another thread reads the shared result.
    Guarded loads preserve CUB's unspecified invalid slots when
    ``oob_default`` is omitted.

    See :ref:`per-thread payloads <coop-thread-data>` and
    :ref:`temporary storage <coop-temp-storage>` for allocation rules.
    The executable example in :func:`cuda.coop.load` activates this backend
    explicitly and shows a guarded final tile.

    See Also
    --------
    :cpp:class:`cub::BlockLoad`, :cpp:class:`cub::WarpLoad`
        C++ Load primitives used for these group scopes.

    Examples
    --------
    Fill an existing Numba local array while copying a partial final tile.
    For ordinary fresh payloads, prefer the returning example in
    :func:`cuda.coop.load`.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_qualified_load_store_examples.py
        :language: python
        :start-after: # qualified-load-store-example-begin
        :end-before: # qualified-load-store-example-end
        :dedent: 4
    """

    group_primitive_marker(
        "load",
        group,
        source,
        output,
        items_per_thread=items_per_thread,
        alignment=alignment,
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
    group: BlockGroup | WarpGroup,
    destination: object,
    value: _CommonNumericT
    | CommonThreadDataLike[_CommonNumericT]
    | numpy.ndarray,
    /,
    *,
    algorithm: BlockLoadStoreAlgorithm | WarpLoadStoreAlgorithm = "direct",
    valid_items: ValidItems | None = None,
    offset: IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> None:
    """Store a block or warp tile with the Numba-CUDA-MLIR backend.

    Parameters, algorithm choices, tile addressing, and the ``None`` return
    follow :func:`cuda.coop.store`. ``value`` may be a numeric scalar or a
    supported fixed-size Numba local array. Transpose algorithms may rearrange
    that input in place; copy values before Store if they are needed later.
    ``destination`` may be a contiguous global or shared array. Synchronize
    before another thread reads a shared-memory value written by this call.

    See :ref:`per-thread payloads <coop-thread-data>` and
    :ref:`temporary storage <coop-temp-storage>` for allocation rules, and
    :func:`cuda.coop.store` for an executable partial-tile example.

    See Also
    --------
    :cpp:class:`cub::BlockStore`, :cpp:class:`cub::WarpStore`
        C++ Store primitives used for these group scopes.

    Examples
    --------
    Store only the valid prefix of the final tile.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_qualified_load_store_examples.py
        :language: python
        :start-after: # qualified-load-store-example-begin
        :end-before: # qualified-load-store-example-end
        :dedent: 4
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
