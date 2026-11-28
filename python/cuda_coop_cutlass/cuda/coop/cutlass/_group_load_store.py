# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Load and Store move a group's tile between memory and registers.

:func:`~cuda.coop.cutlass.load` reads a tile and returns each thread's items
as a payload, and :func:`~cuda.coop.cutlass.store` writes each thread's
payload to a tile. Values that are already in registers, in a CuTe register
tensor or a ``TensorSSA``, become a payload through the conversions of
:ref:`ThreadData <coop-thread-data>`.

A :term:`tile` is the run of consecutive elements that one group reads or
writes in one call, ``items_per_thread`` elements for each of its threads:

.. list-table::
   :header-rows: 1
   :widths: 30 32 38

   * - Group
     - Elements in a tile
     - Where the tile starts
   * - ``this_block()``
     - Block size times ``items_per_thread``
     - At ``offset``
   * - ``this_warp()``
     - 32 times ``items_per_thread``
     - At ``offset`` plus the tiles of the block's earlier warps
   * - ``this_warp().group_by(n)``
     - ``n`` times ``items_per_thread``
     - At ``offset`` plus the tiles of the block's earlier groups

``offset`` counts elements from the start of the memory. On a warp or a
group of lanes the compiler adds the tiles of the earlier groups: with ``G``
threads in the group and ``K`` items per thread it adds ``(b // G) * G * K``,
where ``b = x + block_x * (y + block_y * z)`` is the thread's rank in the
block. The warps of a block, or its groups of lanes, can therefore all pass
the same ``offset``. Every block of a grid works on the same tiles when it
passes the same ``offset``, so a kernel with more than one block computes
``offset`` from ``cute.arch.block_idx()``.

``algorithm`` chooses how the threads reach the tile. It decides the layout
of the payload and whether the call uses
:ref:`scratch <coop-temp-storage>`:

.. list-table::
   :header-rows: 1
   :widths: 26 10 11 10 43

   * - ``algorithm``
     - Groups
     - Payload
     - Scratch
     - Memory access
   * - ``"direct"``
     - All
     - Blocked
     - No
     - Each thread reads or writes its own consecutive elements.
   * - ``"striped"``
     - All
     - Striped
     - No
     - The thread with rank ``t`` in the group reads or writes elements
       ``t``, ``t + G``, ``t + 2 * G`` and so on.
   * - ``"vectorize"``
     - All
     - Blocked
     - No
     - As ``"direct"``, with vector instructions when the call has no
       ``valid_items``, ``items_per_thread`` is even and the tile's address
       is aligned for them.
   * - ``"transpose"``
     - All
     - Blocked
     - Yes
     - As ``"striped"``. The threads exchange items through scratch, after
       they read or before they write.
   * - ``"warp_transpose"``
     - Blocks
     - Blocked
     - Yes
     - As ``"transpose"``, with each warp of the block working by itself on
       its own part of the tile.
   * - ``"warp_transpose_timesliced"``
     - Blocks
     - Blocked
     - Yes
     - As ``"warp_transpose"``, with scratch for one warp that the warps
       use in turn.

Item ``i`` of the thread with rank ``t`` in its group is tile element
``t * K + i`` in a :term:`blocked` payload and ``t + i * G`` in a
:term:`striped` one. On a warp or a group of lanes, ``t`` is ``b % G``. A
payload keeps no record of its layout, and ``store`` reads it in the layout
of its own ``algorithm``. In a group of more than one thread with more than
one item each, a tile loaded with ``"striped"`` and stored with ``"direct"``
is therefore written in another order.

.. figure:: visualizations/load-transpose.svg
   :alt: Two rows of eight threads with two elements each, and a bar named
         shared scratch between them. In the top row thread 0 holds
         elements 0 and 8, and thread 7 holds 7 and 15. In the bottom row
         thread 0 holds elements 0 and 1, and thread 7 holds 14 and 15.
   :width: 100%

   ``load`` with ``algorithm="transpose"`` on eight threads with two items
   each. The threads read in striped order, the top row, where thread 0
   holds elements 0 and 8. The exchange through scratch gives the blocked
   payload of the bottom row, where thread 0 holds elements 0 and 1.
   ``algorithm="striped"`` returns the top row.

``load`` takes the element type from the memory and the item count from
``items_per_thread``. ``store`` takes both from the payload. CuTe reports
unsigned integers behind a ``cute.Pointer`` argument as signed, and
``dtype`` on both calls declares them unsigned.
"""

# Implementation notes.
#
# Validate block and warp Load/Store calls for CuTe lowering.
#
# Exact launch facts and shared group planning determine participation and
# algorithm support. Argument bindings mark each control as omitted, as a
# constant embedded in generated C++, or as a runtime argument. Block calls
# can use explicit scratch descriptors; physical and logical warps use
# compiler-managed scratch when their algorithm needs it. Load materializes a
# temporary register tensor and copies its scalar values into ThreadData.

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any

from cuda.coop.cutlass._planning import ArgumentBinding, GroupLoadStoreAlgorithm
from cuda.coop.cutlass._planning.thread_group import (
    ThreadGroup as PlanningThreadGroup,
)
from cuda.coop.cutlass._typing import (
    BlockLoadStoreAlgorithm,
    IntegerValue,
    ReadableThreadDataLike,
    TempStorageLike,
    ValidItems,
    WarpLoadStoreAlgorithm,
    _NumericT,
)
from cuda.coop.cutlass._validation import _validate_temp_storage

from ._thread_data import ThreadData
from ._thread_group import (
    _require_complete_warp_partition,
    _resolve_primitive_group_from_launch,
)
from ._typing import BlockGroup, WarpGroup

_SCOPE = "cuda.coop.cutlass"
_MAX_STATIC_OFFSET = (1 << 63) - 1


def _resolve_group(group, algorithm, temp_storage, operation):
    """Resolve block or warp groups against exact launch dimensions.

    Physical and logical warps require a block made only of complete 32-thread
    warps, and they reject explicit scratch descriptors. Normalize the
    algorithm and validate any explicit block scratch descriptor. Shared
    planning checks the logical width and algorithm support. Return the group,
    launch facts, and normalized algorithm for lowering.
    """

    if not isinstance(group, PlanningThreadGroup):
        raise TypeError(f"{_SCOPE}.{operation} group must be a ThreadGroup")
    if group.kind not in {"block", "warp", "threads_within_warp"}:
        raise NotImplementedError(
            f"{_SCOPE}.{operation} requires a block, physical warp, "
            "or logical warp group"
        )
    if group.kind != "block" and temp_storage is not None:
        raise NotImplementedError(
            f"{_SCOPE}.{operation} explicit TempStorage is supported only "
            "for block groups"
        )
    algorithm = _normalize_algorithm(algorithm)
    if temp_storage is not None:
        _validate_temp_storage(operation, temp_storage)
        from ._compiler._lifetime import register_use
        from ._temp_storage import TempStorage

        if isinstance(temp_storage, TempStorage):
            register_use(temp_storage, operation)
    from ._compiler._launch import current_kernel_launch_facts

    launch = current_kernel_launch_facts()
    resolved = _resolve_primitive_group_from_launch(
        group, launch, feature=operation
    )
    _require_complete_warp_partition(
        resolved, feature=operation, exact_block_dim=launch.exact_block_dim
    )
    return resolved, launch, algorithm


def load(
    group: BlockGroup | WarpGroup,
    source: object,
    /,
    *,
    items_per_thread: int,
    dtype: object | None = None,
    alignment: int | None = None,
    algorithm: BlockLoadStoreAlgorithm | WarpLoadStoreAlgorithm = "direct",
    valid_items: ValidItems | None = None,
    oob_default: _NumericT | float | None = None,
    offset: IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadData:
    """Read a group's tile from memory into a payload in each thread.

    Parameters
    ----------
    group : ThreadGroup
        ``this_block()``, ``this_warp()``, or ``this_warp().group_by(n)``
        with ``n`` of 1, 2, 4, 8, 16 or 32. Every thread of the group makes
        the call and passes the same ``valid_items``, ``oob_default`` and
        ``offset``. A warp or a group of lanes needs a block whose size is
        a multiple of 32. The block's other warps or groups of lanes can
        skip the call, or make it with other values of those three.
    source : CuTe pointer or tensor
        The memory to read, global or shared. It is a CuTe pointer, such as
        a kernel argument made with ``cutlass.cute.runtime.make_ptr``, or a
        tensor whose shape and strides are Python values and form a compact
        layout: ``cute.make_tensor(pointer, cute.make_layout(n))``, a
        ``cute.recast_tensor`` view of it, a shared tensor from CuTe's
        ``SmemAllocator``, or a tensor from
        :meth:`TempStorage.reserve <cuda.coop.cutlass.TempStorage.reserve>`.
        The pointer must be aligned to the element size in bytes. Use a
        plain layout and an unswizzled pointer for linear memory access.
        Its element type is one of the ten listed under
        :ref:`ThreadData <coop-thread-data>`. A register tensor, a tensor
        with a runtime shape and any other element type are rejected when
        the kernel is compiled. For unsigned integers behind a
        ``cute.Pointer`` argument, pass ``dtype``.

        ``source`` must hold every element the call reads. A tensor that
        is too small is rejected when the kernel is compiled, as long as
        ``offset`` and ``valid_items`` are Python values or left out. On a
        warp or a group of lanes, the check covers the tiles of all such
        groups of the block, including groups that skip the call. A
        pointer has no size, so nothing is checked.

        If other threads wrote the elements to shared memory, put a
        barrier such as ``block.sync()`` between those writes and the call.
    items_per_thread : int
        The number of items each thread receives, a positive Python
        integer or a kernel argument declared ``cutlass.Constexpr``. A
        CuTe value is rejected when the kernel is compiled.
    dtype : CUTLASS or NumPy type, optional
        Declares the integers in ``source`` unsigned. CuTe reports integer
        memory behind a ``cute.Pointer`` argument as signed, including when
        the host made the pointer with an unsigned type, and a
        ``cute.make_tensor`` view of any pointer is signed too. ``Uint32``
        data then loads as ``Int32`` with no error, and a maximum, a sort
        or a conversion to a wider type gives the signed answer.

        Pass the unsigned type of the same width, such as
        ``dtype=cutlass.Uint32`` for memory that CuTe reports as ``Int32``.
        The payload has that element type, and ``oob_default`` is checked
        against it. ``dtype`` can also repeat the type CuTe reports. Any
        other type is rejected when the kernel is compiled, so ``dtype``
        cannot change the width or turn integers into floats.

        ``dtype`` is not needed for a ``source`` that already reports the
        unsigned type, such as a ``cute.recast_tensor`` view, a tensor from
        :meth:`TempStorage.reserve <cuda.coop.cutlass.TempStorage.reserve>`,
        or a pointer the kernel made itself with ``cute.make_ptr`` and
        passes unchanged.
    alignment : int, optional
        The smallest byte alignment, a power of two, of register storage
        made from the returned payload, as for ``alignment`` of
        :class:`~cuda.coop.cutlass.ThreadData`. It is a Python integer, or
        ``None`` (the default) to ask for none. The items, their order and
        the way ``source`` is read stay the same.
    algorithm : str, optional
        A Python string, one of the six names in the ``algorithm`` table
        of :ref:`Load & Store <memory-operations>`. The default is
        ``"direct"``. A block takes all six. A warp or a group of lanes
        takes ``"direct"``, ``"striped"``, ``"vectorize"`` and
        ``"transpose"``. ``"warp_transpose"`` and
        ``"warp_transpose_timesliced"`` need a block size that is a
        multiple of 32. Any other choice is rejected when the kernel is
        compiled.
    valid_items : int or CuTe integer, optional
        The number of elements to read, counted from the start of the
        group's tile, from 0 to the size of that tile. Left out, the whole
        tile is read. An item whose element is at or past the count
        receives ``oob_default``. Without ``oob_default`` its value is
        undefined, so assign it before anything reads it.

        A Python count out of range is rejected when the kernel is
        compiled, and a runtime count out of range traps on the device.
    oob_default : number or CuTe scalar, optional
        The value for items at or past the count. It needs ``valid_items``.
        A Python or NumPy number is fixed when the kernel is compiled. It
        must be finite and fit the element type. A float needs ``Float32``
        or ``Float64`` memory. An integer for such memory must also lie
        between ``-2**63`` and ``2**64 - 1``, so write a default outside
        that range as a float. A CuTe scalar is computed while the kernel
        runs and has exactly the payload's element type. Anything else is
        rejected when the kernel is compiled.
    offset : int or CuTe integer, optional
        The number of elements from the start of ``source`` to the tile, 0
        by default. On a warp or a group of lanes it leads to the tile of
        the block's first group, and the compiler adds the tiles of the
        groups before the calling one. A negative Python integer is
        rejected when the kernel is compiled, and a negative runtime value
        traps on the device. A call reads the same tile every time it runs
        with the same offset, so a loop over tiles computes a new offset
        on each trip.
    temp_storage : TempStorage, optional
        For blocks. ``"transpose"``, ``"warp_transpose"`` and
        ``"warp_transpose_timesliced"`` use scratch. Leave this out and
        the compiler provides the scratch and a block barrier after the
        call. Pass a :ref:`TempStorage <coop-temp-storage>` to share the
        bytes with other calls. A barrier then follows the call only if
        the ``TempStorage`` was created with ``auto_sync=True``. The other
        three algorithms use no scratch and no barrier, with or without a
        ``TempStorage``.

        Warps and groups of lanes take ``None``. There, ``"transpose"``
        uses scratch from the compiler, one slice for each warp or group
        of lanes, and a warp barrier for that group's lanes follows the
        call.

    Returns
    -------
    cuda.coop.cutlass.ThreadData
        A new payload in each thread, with ``items_per_thread`` items of
        the element type of ``source``, or of ``dtype`` when it is given,
        in the layout that ``algorithm`` gives. With ``items_per_thread=1``
        the result is a payload of one item. ``source`` is only read, and
        every call returns its own payload.

    See Also
    --------
    :cpp:struct:`cub::BlockLoad`, :cpp:struct:`cub::WarpLoad`
        The CUB classes behind the call.

    Examples
    --------
    On a block of 64 threads, copy a tile without its last three elements.
    ``load`` starts five elements into ``source`` and sets the three items
    at or past the count to 0. ``store`` writes the same number of
    elements, starting two elements into ``destination``.

    `Full example and test on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_load_store_documentation.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_load_store_documentation.py
        :language: python
        :start-after: # cutlass-load-store-example-begin
        :end-before: # cutlass-load-store-example-end
        :dedent: 4

    The two warps of a block copy one tile each. ``_BLOCK`` is
    ``(8, 4, 2)``, which is 64 threads. ``warp`` is 0 in the first warp
    and 1 in the second. Each warp reads its tile without the last
    ``7 + warp * 11`` elements, a count computed while the kernel runs,
    and the items at or past the count are set to -1. Both warps pass the
    same offsets, a ``_SOURCE_OFFSET`` of 3 and a ``_DESTINATION_OFFSET``
    of 5, and the compiler puts the second warp's tile after the first.
    ``store`` writes each warp's whole tile.

    `Full example on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/warp_load_store.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/examples/warp_load_store.py
        :language: python
        :start-after: docs: start cutlass-warp-load-store
        :end-before: docs: end cutlass-warp-load-store
        :dedent: 4

    The same copy with groups of eight lanes, where ``_WIDTH`` is 8. The
    block's 64 threads form eight groups with one tile each.
    ``group_index`` numbers the groups from 0, and each group reads its
    tile without the last ``1 + group_index`` elements.

    `Full example on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/logical_warp_load_store.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/examples/logical_warp_load_store.py
        :language: python
        :start-after: docs: start cutlass-logical-warp-load-store
        :end-before: docs: end cutlass-logical-warp-load-store
        :dedent: 4

    Sort ``Uint32`` keys from one pointer argument into another. CuTe
    reports both pointers as ``Int32``, so the kernel passes
    ``dtype=cutlass.Uint32`` to ``load`` and ``store``. Without ``dtype``
    the keys with the top bit set would sort first, as negative numbers.

    `Full example and test on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_load_store_documentation.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_load_store_documentation.py
        :language: python
        :start-after: # cutlass-load-store-unsigned-example-begin
        :end-before: # cutlass-load-store-unsigned-example-end
        :dedent: 4
    """

    output = ThreadData(items_per_thread, alignment=alignment)
    if oob_default is not None and valid_items is None:
        raise ValueError(f"{_SCOPE}.load oob_default requires valid_items")
    group, launch, algorithm = _resolve_group(
        group, algorithm, temp_storage, "load"
    )
    from ._lowering._load_store import provider_load

    provider_load(
        group=group,
        launch=launch,
        source=source,
        output=output,
        dtype=dtype,
        algorithm=algorithm,
        valid_items=valid_items,
        valid_items_binding=_classify_integer_binding(
            valid_items, name="valid_items"
        ),
        oob_default=oob_default,
        oob_default_binding=_classify_oob_default(oob_default),
        offset=offset,
        offset_binding=_classify_integer_binding(offset, name="offset"),
        temp_storage=temp_storage,
    )
    return output


def store(
    group: BlockGroup | WarpGroup,
    destination: object,
    value: _NumericT | ReadableThreadDataLike[_NumericT],
    /,
    *,
    dtype: object | None = None,
    algorithm: BlockLoadStoreAlgorithm | WarpLoadStoreAlgorithm = "direct",
    valid_items: ValidItems | None = None,
    offset: IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> None:
    """Write each thread's payload to a group's tile in memory.

    Parameters
    ----------
    group : ThreadGroup
        The groups that :func:`~cuda.coop.cutlass.load` takes, under the
        same rules. Every thread of the group makes the call and passes
        the same ``valid_items`` and ``offset``.
    destination : CuTe pointer or tensor
        The memory to write, global or shared, in the forms that ``load``
        takes as ``source`` and with the same check of its size. Its
        element type, as CuTe reports it or as ``dtype`` declares it, is
        the element type of ``value``. Nothing is converted, and two
        different types are rejected when the kernel is compiled.
    value : ThreadData or CuTe scalar
        This thread's payload. Every item must be assigned, and a payload
        with an unassigned item is rejected when the kernel is compiled.
        One scalar per thread, such as ``cutlass.Int32(x)``, is stored as
        a payload of one item. To store a CuTe register tensor or a
        ``TensorSSA``, convert it first with ``ThreadData``'s
        :meth:`~cuda.coop.cutlass.ThreadData.from_register_tensor` or
        :meth:`~cuda.coop.cutlass.ThreadData.from_vector`.

        ``"direct"``, ``"striped"`` and ``"vectorize"`` leave the payload
        as it was. After ``"transpose"``, ``"warp_transpose"`` or
        ``"warp_transpose_timesliced"`` its items are undefined, so copy
        the ones the kernel still needs before the call.
    dtype : CUTLASS or NumPy type, optional
        Declares the integers in ``destination`` unsigned, under the rules
        of ``dtype`` of ``load``. To store an unsigned payload through a
        ``cute.Pointer`` argument, pass the payload's element type, such as
        ``dtype=cutlass.Uint32``.
    algorithm : str, optional
        A Python string, a name from the ``algorithm`` table of
        :ref:`Load & Store <memory-operations>`. It states the layout of
        ``value`` and decides how the tile is written. ``"striped"`` takes
        a striped payload, and the other five take a blocked one. The
        default is ``"direct"``. Groups and block sizes limit the choice
        as they do for ``load``.
    valid_items : int or CuTe integer, optional
        The number of elements to write, counted from the start of the
        group's tile, from 0 to the size of that tile. Left out, the whole
        tile is written. Elements of ``destination`` at or past the count
        keep their values, so a count of 0 writes nothing. A count out of
        range is handled as it is for ``load``.
    offset : int or CuTe integer, optional
        The number of elements from the start of ``destination`` to the
        tile, 0 by default, under the rules of ``offset`` of ``load``.
    temp_storage : TempStorage, optional
        For blocks, the scratch of the three transpose algorithms, under
        the rules of ``temp_storage`` of ``load``.

    Returns
    -------
    None

    See Also
    --------
    :cpp:struct:`cub::BlockStore`, :cpp:struct:`cub::WarpStore`
        The CUB classes behind the call.

    Examples
    --------
    ``store`` below writes a block's tile without its last three elements,
    starting two elements into ``destination``. Those three elements of
    ``destination`` keep the values they had.

    `Full example and test on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_load_store_documentation.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_load_store_documentation.py
        :language: python
        :start-after: # cutlass-load-store-example-begin
        :end-before: # cutlass-load-store-example-end
        :dedent: 4
    """

    group, launch, algorithm = _resolve_group(
        group, algorithm, temp_storage, "store"
    )
    from ._lowering._load_store import provider_store

    provider_store(
        group=group,
        launch=launch,
        destination=destination,
        value=value,
        dtype=dtype,
        algorithm=algorithm,
        valid_items=valid_items,
        valid_items_binding=_classify_integer_binding(
            valid_items, name="valid_items"
        ),
        offset=offset,
        offset_binding=_classify_integer_binding(offset, name="offset"),
        temp_storage=temp_storage,
    )


def _normalize_algorithm(algorithm: Any) -> GroupLoadStoreAlgorithm:
    """Resolve an enum or normalized Load/Store algorithm name.

    Strip surrounding whitespace and accept case and hyphen variants before
    matching the shared selector. Report available choices for unknown names.
    """

    token = getattr(algorithm, "value", algorithm)
    if isinstance(token, str):
        token = token.strip().lower().replace("-", "_")
    try:
        return GroupLoadStoreAlgorithm(token)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(item.value for item in GroupLoadStoreAlgorithm)
        raise ValueError(
            f"{_SCOPE}.load/store algorithm must be one of {choices}"
        ) from exc


def _is_boolean(value: Any) -> bool:
    """Recognize Python, NumPy, and DSL booleans before integer checks."""

    if isinstance(value, bool):
        return True
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        if isinstance(value, np.bool_):
            return True
    from cutlass.base_dsl.typing import Boolean

    return isinstance(value, Boolean)


def _classify_integer_binding(value: Any, *, name: str) -> ArgumentBinding:
    """Separate omitted, embedded, and runtime integer controls.

    Reject booleans. Validate static offset bounds here; shared planning
    checks static valid counts against the tile size. DSL integer values
    remain runtime bindings for the provider ABI.
    """

    if value is None:
        return ArgumentBinding.omitted()
    if _is_boolean(value):
        raise TypeError(f"{_SCOPE}.load/store {name} must be an integer")
    if isinstance(value, Integral):
        normalized = int(value)
        if name == "offset" and normalized < 0:
            raise ValueError(f"{_SCOPE}.load/store offset must be non-negative")
        if name == "offset" and normalized > _MAX_STATIC_OFFSET:
            raise ValueError(
                f"{_SCOPE}.load/store offset must fit a signed 64-bit integer"
            )
        return ArgumentBinding.static(normalized)
    from cutlass.base_dsl.typing import Integer

    if isinstance(value, Integer):
        return ArgumentBinding.runtime()
    raise TypeError(
        f"{_SCOPE}.load/store {name} must be an integer, "
        f"not {type(value).__name__}"
    )


def _classify_oob_default(value: Any) -> ArgumentBinding:
    """Embed host numeric defaults and retain DSL inputs for runtime.

    Embed Python and NumPy numbers as plain int or float constants, and keep
    DSL values as runtime inputs. Reject booleans and non-finite floats here.
    The provider later checks range and float-to-integer compatibility with
    the memory dtype.
    """

    if value is None:
        return ArgumentBinding.omitted()
    if _is_boolean(value):
        raise TypeError(
            f"{_SCOPE}.load oob_default must be numeric, not boolean"
        )
    if isinstance(value, Integral):
        return ArgumentBinding.static(int(value))
    if isinstance(value, Real):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(f"{_SCOPE}.load oob_default must be finite")
        return ArgumentBinding.static(normalized)
    from cutlass.base_dsl.typing import Numeric

    if isinstance(value, Numeric):
        return ArgumentBinding.runtime()
    raise TypeError(
        f"{_SCOPE}.load oob_default must be a numeric scalar, not "
        f"{type(value).__name__}"
    )


__all__ = [
    "_is_boolean",
    "load",
    "store",
]
