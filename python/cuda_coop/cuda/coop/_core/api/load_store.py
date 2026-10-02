# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from typing import Any

from ..thread_group import CoopCompilerContextRequiredError, ThreadGroup
from ._dispatch import (
    _common_group_operation,
)
from ._payload import (
    TempStorageLike,
    ThreadDataLike,
)


@_common_group_operation(
    "load",
    group_kinds=("block", "warp", "threads_within_warp"),
)
def load(
    group: ThreadGroup,
    source: object,
    output: ThreadDataLike[Any],
    /,
    *,
    algorithm: str = "direct",
    valid_items: object = None,
    oob_default: object = None,
    offset: object = None,
    temp_storage: TempStorageLike | None = None,
) -> None:
    """Load a group tile from memory into per-thread values.

    Parameters
    ----------
    group : cuda.coop.ThreadGroup
        Participating threads; see :ref:`thread groups <coop-thread-groups>`.
        Supports blocks and physical or logical warps. Warp loads require
        an enclosing block size divisible by 32.
    source : array
        One-dimensional contiguous source array in device-accessible memory.
        Its element dtype must match ``output``; an untyped ``ThreadData``
        infers its dtype from this array. The array must contain all elements
        selected by ``offset`` and ``valid_items``.
    output : cuda.coop.ThreadDataLike
        Writable :ref:`per-thread payload <coop-thread-data>`.
        Load populates this payload in place. The group's tile contains
        ``group_size * items_per_thread`` elements.
    algorithm : str, optional
        Compile-time load algorithm, default ``"direct"``. ``"direct"`` gives
        each thread consecutive elements (blocked order); ``"striped"`` gives
        neighboring threads neighboring elements at each item index.
        ``"vectorize"`` uses vector accesses when possible, and ``"transpose"``
        uses shared scratch to rearrange striped accesses into blocked order.
        Both return blocked order. Blocks also support ``"warp_transpose"``
        and ``"warp_transpose_timesliced"``, which return blocked order and
        require a block size divisible by 32.
    valid_items : int or integer scalar, optional
        Number of valid elements in the group's tile, shared by all threads
        in that group. Supply a value between zero and the tile size,
        inclusive. ``None`` loads the full tile. Slots beyond this valid
        prefix have unspecified values unless ``oob_default`` is given.
    oob_default : numeric scalar, optional
        Value written to slots beyond ``valid_items``. Requires an explicit
        ``valid_items`` count. For example, use zero to pad a partial tile
        before summing it. A runtime value must have the payload dtype and
        be uniform across the group. With ``None``, those slots are unspecified,
        even if initialized before the Load; assign them before reading them.
    offset : int or integer scalar, optional
        Nonnegative offset in elements from the start of ``source``, uniform
        across the group. ``None`` means zero. For block tiles, supply the
        block's starting offset explicitly. For Warp tiles, the backend adds
        ``(linear_thread_rank // group_size) * tile_size`` automatically;
        do not include that within-block group offset a second time.
    temp_storage : cuda.coop.TempStorageLike, optional
        :ref:`Scratch descriptor <coop-temp-storage>` for block
        transpose-family algorithms. ``None`` uses automatic scratch.
        Direct, striped, and vectorized loads need no shared scratch.
        Warp loads require ``None``.

    Returns
    -------
    None
        The call populates ``output`` in place.

    See Also
    --------
    :cpp:class:`cub::BlockLoad`, :cpp:class:`cub::WarpLoad`
        C++ block and warp Load primitives.

    Examples
    --------
    Copy an array with Numba-CUDA-MLIR, using 128 threads and a kernel argument
    for the values per thread. Each block loads up to
    ``128 * items_per_thread`` elements. The last block pads its
    missing values with zero and stores only the valid prefix.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_load_example.py
        :language: python
        :start-after: # example-begin
        :end-before: # example-end
        :dedent: 4

    The qualified import activates the Numba-CUDA-MLIR backend even if another
    module imported ``cuda.coop`` first. Use the qualified
    ``cuda.coop.<backend>`` API for backend-specific behavior.
    """

    raise CoopCompilerContextRequiredError(
        "cuda.coop.load must be called from a supported GPU kernel."
    )


@_common_group_operation(
    "store",
    group_kinds=("block", "warp", "threads_within_warp"),
)
def store(
    group: ThreadGroup,
    destination: object,
    value: object,
    /,
    *,
    algorithm: str = "direct",
    valid_items: object = None,
    offset: object = None,
    temp_storage: TempStorageLike | None = None,
) -> None:
    """Store a group tile from per-thread values into memory.

    Parameters
    ----------
    group : cuda.coop.ThreadGroup
        Participating threads; see :ref:`thread groups <coop-thread-groups>`.
        Supports blocks and physical or logical warps. Warp stores require
        an enclosing block size divisible by 32.
    destination : array
        Writable one-dimensional contiguous array in device-accessible
        memory, with the same element dtype as ``value``. It must contain
        every element selected by ``offset`` and ``valid_items``.
    value : numeric scalar or cuda.coop.ThreadDataLike
        This thread's value or readable :ref:`payload <coop-thread-data>`.
        Initialize every item that will be stored. The tile contains
        ``group_size * items_per_thread`` elements, with one item per thread
        for a scalar. As in CUB, transpose algorithms may rearrange the payload
        in place. Do not rely on its contents after Store; copy values before
        the call if they are needed later.
    algorithm : str, optional
        Compile-time store algorithm, default ``"direct"``. ``"direct"``
        expects blocked values; ``"striped"`` expects striped values.
        ``"vectorize"`` and ``"transpose"`` also expect blocked values and
        use vector accesses or shared-memory rearrangement, respectively.
        Blocks additionally support ``"warp_transpose"`` and
        ``"warp_transpose_timesliced"``, both requiring a block size divisible
        by 32. See :ref:`data layouts <coop-data-layouts>` before pairing
        different Load and Store algorithms.
    valid_items : int or integer scalar, optional
        Number of valid elements in the group's tile, uniform across the
        group and between zero and the tile size, inclusive. ``None`` stores
        the entire tile. Elements outside the valid prefix are not written.
    offset : int or integer scalar, optional
        Nonnegative offset in elements from the start of ``destination``,
        uniform across the group. ``None`` means zero. Supply each block's
        origin explicitly. Warp stores also add the within-block group
        origin automatically, using the same addressing rule as
        :func:`cuda.coop.load`.
    temp_storage : cuda.coop.TempStorageLike, optional
        :ref:`Scratch descriptor <coop-temp-storage>` for block
        transpose-family algorithms. ``None`` uses automatic scratch.
        Direct, striped, and vectorized stores need no shared scratch.
        Warp stores require ``None``.

    Returns
    -------
    None
        The call writes to ``destination``. The input payload may be rearranged.

    See Also
    --------
    :cpp:class:`cub::BlockStore`, :cpp:class:`cub::WarpStore`
        C++ block and warp Store primitives.

    Examples
    --------
    Store a partial tile at an element offset. The untouched prefix and
    suffix keep their sentinel values. The input payload is not used after
    the transpose Store.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_store_example.py
        :language: python
        :start-after: # example-begin
        :end-before: # example-end
        :dedent: 4

    See :ref:`participation and synchronization <coop-participation>` for
    control-flow requirements at primitive calls.
    """

    raise CoopCompilerContextRequiredError(
        "cuda.coop.store must be called from a supported GPU kernel."
    )


__all__ = ["load", "store"]
