# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""``exchange`` and ``shuffle`` move values between the threads of a group.

Both work on values the threads already hold in registers, such as a payload
from :func:`~cuda.coop.cutlass.load`, and the results stay in registers.

``exchange`` changes which thread holds each item of a :term:`tile`. In a
:term:`blocked` layout each thread holds neighboring elements of the tile,
and in a :term:`striped` layout neighboring threads do.

.. figure:: visualizations/exchange-striped.svg
   :alt: Two rows of eight threads with two items each, joined by a band
         of shared scratch. In the top row thread 0 holds 0 and 8, thread 1
         holds 1 and 9, and so on up to thread 7 with 7 and 15. In the
         bottom row thread 0 holds 0 and 1, thread 1 holds 2 and 3, and so
         on up to thread 7 with 14 and 15.
   :width: 100%

   ``exchange`` with ``mode="striped_to_blocked"`` on eight threads with
   two items each. Each number is an item's position in the tile. Thread 0
   holds positions 0 and 8 before the call and 0 and 1 after it.

For a group of ``G`` threads with ``K`` items each, item ``i`` of the thread
with rank ``t`` holds this tile position:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Layout
     - Tile position
   * - Blocked
     - ``t * K + i``
   * - Striped
     - ``t + i * G``
   * - Warp-striped
     - Striped inside each warp of a block: ``w * 32 * K + l + i * 32``,
       for lane ``l`` of warp ``w``

On a block, the tile is the block's ``G * K`` items. Each warp or group of
lanes has a tile of its own, with ``G`` the group's size and ``t`` the
thread's rank in its group.

The four layout modes convert a payload from one layout to another. The four
scatter modes send items to the tile positions that ``ranks`` gives for
them:

.. list-table::
   :header-rows: 1
   :widths: 38 14 48

   * - ``mode``
     - Groups
     - What it does
   * - ``"striped_to_blocked"`` (default)
     - All
     - Turns a striped payload into a blocked one.
   * - ``"blocked_to_striped"``
     - All
     - Turns a blocked payload into a striped one.
   * - ``"warp_striped_to_blocked"``
     - Blocks
     - Turns a warp-striped payload into a blocked one.
   * - ``"blocked_to_warp_striped"``
     - Blocks
     - Turns a blocked payload into a warp-striped one.
   * - ``"scatter_to_blocked"``
     - Blocks
     - Sends each item to the position in ``ranks``. The result is blocked.
   * - ``"scatter_to_striped"``
     - Blocks
     - Sends each item to the position in ``ranks``. The result is striped.
   * - ``"scatter_to_striped_guarded"``
     - Blocks
     - The same as ``"scatter_to_striped"`` for the items whose rank in
       ``ranks`` is zero or more. It skips the items with a negative rank.
   * - ``"scatter_to_striped_flagged"``
     - Blocks
     - The same as ``"scatter_to_striped"`` for the items whose flag in
       ``valid_flags`` is nonzero. It skips the items with a zero flag.

Nothing records which layout a payload has, and ``exchange`` moves the items
as the mode says. In a group of more than one thread with more than one item
each, ``"striped_to_blocked"`` on a payload that is already blocked returns
the items in the wrong order, without an error. A payload loaded with
``algorithm="striped"`` is striped, and a scan reads a blocked one, so a
kernel that does both calls ``exchange`` in between.
"""

# Implementation notes.
#
# Validate Exchange operands and adapt register payloads.
#
# Adapt register containers to initialized ThreadData and classify their
# dtypes. The plan selects the block or warp specialization and its owned
# scratch; lowering emits the CuTe call and C++ wrapper.

from __future__ import annotations

from enum import Enum
from typing import Literal, TypeVar

from cuda.coop.cutlass._planning import GroupExchangeMode
from cuda.coop.cutlass._planning.thread_group import ThreadGroup
from cuda.coop.cutlass._typing import (
    BlockExchangeMode,
    IntegralScalar,
    SignedIntegerScalar,
)

from ._thread_data import (
    CutlassTensorSample,
    CutlassTensorSSASample,
    ThreadData,
    _coerce_thread_payload,
)
from ._thread_group import (
    _require_complete_warp_partition,
    _resolve_primitive_group_from_launch,
)
from ._typing import (
    BlockGroup,
    ExchangeMode,
    NumericScalar,
    ReadableThreadDataLike,
    WarpGroup,
)

_ItemT = TypeVar("_ItemT", bound=NumericScalar)


_SCOPE = "cuda.coop.cutlass"
_BLOCK_MODES = frozenset(mode.value for mode in GroupExchangeMode)
_WARP_MODES = frozenset({"striped_to_blocked", "blocked_to_striped"})


def _normalize_exchange_mode(mode, *, group_kind):
    """Resolve a mode within the group's supported Exchange family.

    Block groups expose layout conversions and scatter forms. Physical and
    logical warps expose only blocked/striped conversions; other restrictions
    are checked by planning.
    """

    if not isinstance(mode, str) or isinstance(mode, Enum):
        raise TypeError(f"{_SCOPE}.exchange mode must be a compile-time string")
    token = mode.strip().lower().replace("-", "_")
    allowed = _BLOCK_MODES if group_kind == "block" else _WARP_MODES
    if token not in allowed:
        raise ValueError(
            f"{_SCOPE}.exchange mode for {group_kind} groups must be one of: "
            + ", ".join(sorted(allowed))
        )
    return GroupExchangeMode(token)


def _payload(value, *, name):
    """Adapt a register container and require a per-thread payload.

    Calls accept readable ThreadData or a CuTe
    register tensor or TensorSSA value. Convert these inputs with
    ``ThreadData.from_payload``.
    """

    value = _coerce_thread_payload(
        value,
        scope=_SCOPE,
        primitive_name="exchange",
        arg_name=name,
    )
    if not isinstance(value, ThreadData):
        raise TypeError(
            f"{_SCOPE}.exchange {name} must be a fixed-size ThreadData"
        )
    return value


def exchange(
    group: BlockGroup | WarpGroup,
    value: ReadableThreadDataLike[_ItemT]
    | CutlassTensorSample
    | CutlassTensorSSASample,
    /,
    *,
    mode: BlockExchangeMode
    | ExchangeMode
    | Literal[
        "scatter_to_striped_guarded", "scatter_to_striped_flagged"
    ] = "striped_to_blocked",
    ranks: ReadableThreadDataLike[SignedIntegerScalar]
    | CutlassTensorSample
    | CutlassTensorSSASample
    | None = None,
    valid_flags: ReadableThreadDataLike[IntegralScalar]
    | CutlassTensorSample
    | CutlassTensorSSASample
    | None = None,
    warp_time_slicing: bool = False,
) -> ThreadData:
    """Rearrange the items that the threads of a group hold.

    Each thread passes its payload and receives a new one. The table of
    modes at the top of :ref:`Data rearrangement <coop-cutlass-exchange>`
    says what each mode does.

    Parameters
    ----------
    group : ThreadGroup
        ``this_block()``, ``this_warp()``, or ``this_warp().group_by(n)``
        with ``n`` of 1, 2, 4, 8, 16 or 32. Any other group is rejected
        when the kernel is compiled. Every thread of the group makes the
        call, on every trip of a loop that contains it, including a thread
        whose items a scatter skips. On a warp or a group of lanes, the
        other groups of the block can skip the call. A warp group needs a
        block whose size is a multiple of 32.
    value : ThreadData, CuTe register tensor or TensorSSA
        A payload. A register tensor or ``TensorSSA`` is read as a payload,
        and the result is a ``ThreadData`` that ``to_register_tensor`` or
        ``to_tensor_ssa`` converts back. One scalar per thread is rejected
        when the kernel is compiled: pass a payload of one item. A tensor
        in global or shared memory is rejected when the kernel is compiled,
        so read it with :func:`~cuda.coop.cutlass.load` first. An item has
        one of the ten element types listed under
        :ref:`ThreadData <coop-thread-data>`.
    mode : str, optional
        A Python string. The default is ``"striped_to_blocked"``. A block
        takes all eight modes. ``"warp_striped_to_blocked"`` and
        ``"blocked_to_warp_striped"`` need a block size that is a multiple
        of 32, and the other six take a block of any size. A warp or a
        group of lanes takes ``"striped_to_blocked"`` and
        ``"blocked_to_striped"``. A mode that the group does not take is
        rejected when the kernel is compiled.
    ranks : ThreadData, CuTe register tensor or TensorSSA, optional
        For ``"scatter_to_blocked"``, ``"scatter_to_striped"``,
        ``"scatter_to_striped_guarded"`` and
        ``"scatter_to_striped_flagged"``, which need it. The four layout
        modes take ``None``.

        A payload with the item count of ``value``. Item ``i`` of ``ranks``
        is the tile position that item ``i`` of ``value`` goes to. The
        element type is ``Int8``, ``Int16``, ``Int32`` or ``Int64``,
        whatever the element type of ``value``. Another item count or
        element type is rejected when the kernel is compiled. Choose a type
        that holds ``G * K - 1`` for a block of ``G`` threads with ``K``
        items each: ``Int8`` reaches 128 tile positions.

        ``"scatter_to_blocked"`` and ``"scatter_to_striped"`` write every
        item. ``"scatter_to_striped_guarded"`` writes the items whose rank
        is zero or more, and ``"scatter_to_striped_flagged"`` writes the
        items whose flag is nonzero. The ranks of the written items must be
        distinct and from 0 to ``G * K - 1``, so with the first two modes
        they name each tile position once.

        Nothing checks the rank values, when the kernel is compiled or
        while it runs. Two written items with the same rank leave the
        result undefined. Without ``warp_time_slicing``, a written item
        whose rank is outside the range is stored in shared memory outside
        the call's scratch. That includes a rank of ``G * K`` or more with
        ``"scatter_to_striped_guarded"``.
    valid_flags : ThreadData, CuTe register tensor or TensorSSA, optional
        For ``"scatter_to_striped_flagged"``, which needs it. Every other
        mode takes ``None``. A payload with the item count of ``value`` and
        one of the eight integer element types. Another item count or
        element type is rejected when the kernel is compiled. That includes
        ``Boolean``, the type of a comparison such as ``x > 0``. Item ``i``
        of ``value`` is written when item ``i`` of ``valid_flags`` is
        nonzero, and its rank is ignored when the flag is zero.
    warp_time_slicing : bool, optional
        For blocks. A Python bool, ``False`` by default. A block's scratch
        holds the whole tile. With ``True`` it holds the items of one warp,
        and the warps of the block take turns, with a block barrier between
        turns. The result is the same, and the call uses less shared memory
        on a block of more than 32 threads.
        ``"scatter_to_striped_guarded"``, ``"scatter_to_striped_flagged"``,
        warps and groups of lanes take ``False``.

    Returns
    -------
    cuda.coop.cutlass.ThreadData
        A new payload with the item count, element type and alignment of
        ``value``. ``value``, ``ranks`` and ``valid_flags`` keep their
        values. A layout mode defines every item. After a scatter, a result
        item is undefined when no item was sent to its tile position.
        Assign it before reading it.

    Notes
    -----
    Every mode uses scratch, and ``exchange`` takes no ``temp_storage``. The
    compiler provides the scratch and a barrier after the call. On a block,
    that is a block barrier. On a warp or a group of lanes, each group has
    scratch of its own and the barrier is a warp barrier for the lanes of
    that group.

    See Also
    --------
    :cpp:struct:`cub::BlockExchange`, :cpp:struct:`cub::WarpExchange`
        The CUB classes behind the call.

    Examples
    --------
    Turn a blocked payload into a striped one. The kernel then shifts the
    result with :func:`~cuda.coop.cutlass.shuffle`. ``_BLOCK`` is
    ``(8, 4, 2)``, a block of 64 threads, and ``_THREADS`` is 64. With
    three items per thread and a ``source`` that counts from 0 to 191,
    thread 0 holds 0, 1, 2 in ``payload`` and 0, 64, 128 in ``striped``.

    `Full example on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/exchange_shuffle.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/examples/exchange_shuffle.py
        :language: python
        :start-after: docs: start cutlass-exchange-shuffle
        :end-before: docs: end cutlass-exchange-shuffle
        :dedent: 4

    Reverse a block's tile with ``"scatter_to_blocked"``. The block has 64
    threads, so the tile has ``64 * items_per_thread`` positions. Each
    item's rank is ``64 * items_per_thread - 1`` minus its own position,
    ``thread * items_per_thread + item``. ``ranks`` is made without a
    ``dtype`` and takes ``Int32`` from its items.

    `Full example and test on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_collective_documentation.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_collective_documentation.py
        :language: python
        :start-after: # cutlass-scatter-example-begin
        :end-before: # cutlass-scatter-example-end
        :dedent: 4
    """
    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_SCOPE}.exchange group must be a ThreadGroup")
    if group.kind not in {"block", "warp", "threads_within_warp"}:
        raise NotImplementedError(
            f"{_SCOPE}.exchange requires a block or warp group"
        )
    mode = _normalize_exchange_mode(mode, group_kind=group.kind)
    if not isinstance(warp_time_slicing, bool):
        raise TypeError(
            f"{_SCOPE}.exchange warp_time_slicing must be a compile-time bool"
        )
    if warp_time_slicing and group.kind != "block":
        raise ValueError(
            f"{_SCOPE}.exchange warp_time_slicing applies only to blocks"
        )
    if mode.uses_ranks != (ranks is not None):
        requirement = "requires" if mode.uses_ranks else "does not accept"
        raise ValueError(f"{_SCOPE}.exchange {mode.value} {requirement} ranks")
    if mode.uses_valid_flags != (valid_flags is not None):
        requirement = "requires" if mode.uses_valid_flags else "does not accept"
        raise ValueError(
            f"{_SCOPE}.exchange {mode.value} {requirement} valid_flags"
        )
    value = _payload(value, name="value")
    ranks = None if ranks is None else _payload(ranks, name="ranks")
    valid_flags = (
        None
        if valid_flags is None
        else _payload(valid_flags, name="valid_flags")
    )

    from ._compiler._launch import current_kernel_launch_facts
    from ._lowering import _exchange

    launch = current_kernel_launch_facts()
    group = _resolve_primitive_group_from_launch(
        group, launch, feature="exchange"
    )
    _require_complete_warp_partition(
        group, feature="exchange", exact_block_dim=launch.exact_block_dim
    )
    value_type, rank_type, flag_type = (
        _exchange._resolve_exchange_operand_types(
            value=value, ranks=ranks, valid_flags=valid_flags
        )
    )
    plan = _exchange._make_group_exchange_plan(
        group=group,
        launch=launch,
        dtype=value_type,
        items_per_thread=value.items_per_thread,
        mode=mode.value,
        rank_dtype=rank_type,
        valid_flag_dtype=flag_type,
        warp_time_slicing=warp_time_slicing,
    ).require_supported()
    return _exchange.provider_exchange(
        plan=plan, value=value, ranks=ranks, valid_flags=valid_flags
    )


__all__ = ["exchange"]
