# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""``shuffle`` shifts a block's values along the order of its threads.

Given a payload, it shifts the block's items by one position, across thread
boundaries. Given one scalar per thread, it gives each thread the scalar of
the thread ``distance`` ranks ahead of it.

.. figure:: visualizations/shuffle-up.svg
   :alt: Eight threads hold the values 0 to 15, two each. A line leads from
         each of the values 0 to 14 to the next position in the row below.
         That row holds a question mark and 0 in thread 0, 1 and 2 in
         thread 1, and so on up to 13 and 14 in thread 7.
   :width: 100%

   ``shuffle`` with ``mode="up"`` on a block of eight threads with two
   items each. Thread 1 holds 2 and 3 before the call and 1 and 2 after it.
   The first item of thread 0 has no source and is undefined.

In this table a block has ``G`` threads with ``K`` items each. Position
``t * K + i`` is item ``i`` of the thread with rank ``t``:

.. list-table::
   :header-rows: 1
   :widths: 24 16 60

   * - ``mode``
     - ``value``
     - What it does
   * - ``"down"`` (default)
     - Payload
     - Position ``j`` receives the item from position ``j + 1``.
   * - ``"up"``
     - Payload
     - Position ``j`` receives the item from position ``j - 1``.
   * - ``"offset"``
     - Scalar
     - Rank ``t`` receives the scalar from rank ``t + distance``.
   * - ``"rotate"``
     - Scalar
     - Rank ``t`` receives the scalar from rank ``(t + distance) % G``.

``"rotate"`` wraps around the block. The other three modes stop at its
ends, and a result with no source is undefined.

``"down"`` and ``"up"`` follow the order in which the threads hold the
items, whichever layout the payload has. On a :term:`blocked` payload they
shift the :term:`tile` by one element.
"""

# Implementation notes.
#
# Validate block Shuffle calls and adapt register payloads.
#
# Payload up/down shifts and scalar offset/rotate calls have different result
# ABIs. Planning checks the selected form and distance before the
# provider emits a call; register inputs are converted to ThreadData.

from __future__ import annotations

from enum import Enum
from typing import Literal, TypeVar

from cuda.coop.cutlass._planning.thread_group import ThreadGroup
from cuda.coop.cutlass._typing import IntegerValue

from ._thread_data import (
    CutlassTensorSample,
    CutlassTensorSSASample,
    ThreadData,
    _coerce_thread_payload,
)
from ._typing import (
    ArrayShuffleMode,
    BlockGroup,
    NumericScalar,
    ReadableThreadDataLike,
    ScalarShuffleMode,
)

_ItemT = TypeVar("_ItemT", bound=NumericScalar)


def shuffle(
    group: BlockGroup,
    value: ReadableThreadDataLike[_ItemT]
    | CutlassTensorSample
    | CutlassTensorSSASample
    | _ItemT,
    /,
    *,
    mode: ArrayShuffleMode | ScalarShuffleMode = "down",
    distance: Literal[1] | IntegerValue = 1,
) -> ThreadData | _ItemT:
    """Shift a block's items one position, or pass scalars between threads.

    The :ref:`table of modes <coop-cutlass-shuffle>` says what each mode
    does.

    Parameters
    ----------
    group : ThreadGroup
        ``this_block()``. Any other group is rejected when the kernel is
        compiled. Every thread of the block makes the call, on every trip
        of a loop that contains it, including a thread that does not use
        its result.
    value : scalar, ThreadData, CuTe register tensor or TensorSSA
        A payload for ``"down"`` and ``"up"``, or one scalar per thread for
        ``"offset"`` and ``"rotate"``. A register tensor or ``TensorSSA``
        is read as a payload, and so is a ``ThreadData`` of one item. A
        tensor in global or shared memory is rejected when the kernel is
        compiled: read a payload with :func:`~cuda.coop.cutlass.load`, or
        index the tensor to get one scalar. A scalar or an item has one of
        the ten element types listed under
        :ref:`ThreadData <coop-thread-data>`.
    mode : {"down", "up", "offset", "rotate"}, optional
        A Python string. The default is ``"down"``, so a call that passes a
        scalar names ``"offset"`` or ``"rotate"``. A scalar with ``"down"``
        or ``"up"`` is rejected when the kernel is compiled, and so is a
        payload with ``"offset"`` or ``"rotate"``.
    distance : int or CuTe integer, optional
        For ``"offset"`` and ``"rotate"``, how many ranks ahead of a thread
        its source is. The default is 1. With a payload, leave ``distance``
        at 1. Any other value, or a CuTe integer, is then rejected when the
        kernel is compiled.

        A Python or NumPy integer is fixed when the kernel is compiled. A
        CuTe integer is read while the kernel runs, and each thread can
        pass a different one. Its type is ``Int8`` to ``Int64`` or
        ``Uint8`` to ``Uint32``, and another type is rejected when the
        kernel is compiled.

        ``"offset"`` takes any distance that fits a signed 32-bit integer,
        zero and negative distances included. ``"rotate"`` takes 1 to
        ``G - 1`` on a block of ``G`` threads, so it needs a block of at
        least two threads. A Python distance outside the range is rejected
        when the kernel is compiled, and a runtime distance outside it
        traps on the device.

    Returns
    -------
    CuTe scalar or cuda.coop.cutlass.ThreadData
        A payload gives a new payload with the same item count, element
        type and alignment. A scalar gives a CuTe scalar of the same type.
        The input keeps its values.

        ``"rotate"`` gives every thread a defined result. ``"down"`` leaves
        the last item of the thread with the highest rank undefined, and
        ``"up"`` leaves item 0 of the thread with rank 0 undefined.
        ``"offset"`` leaves the result undefined in a thread whose rank
        plus ``distance`` lies outside the block. Assign an undefined item
        of a payload before reading it, as the first example below does.

    Notes
    -----
    Every mode uses scratch, one item for each thread of the block, and
    ``shuffle`` takes no ``temp_storage``. The compiler provides the scratch
    and a block barrier after the call.

    CUB's ``Up`` can also give every thread the block's last item, and its
    ``Down`` the block's first item. Those overloads have no counterpart
    here.

    See Also
    --------
    :cpp:struct:`cub::BlockShuffle`
        The CUB class behind the call.

    Examples
    --------
    Shift a payload down by one position. ``_BLOCK`` is ``(8, 4, 2)``, a
    block of 64 threads, and ``_THREADS`` is 64. With three items per
    thread and a ``source`` that counts from 0 to 191, thread 0 holds 0,
    64, 128 in ``striped`` and 64, 128, 1 in ``shifted``. The last item of
    thread 63 has no source, and the kernel assigns it before ``store``.

    `Full example on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/exchange_shuffle.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/examples/exchange_shuffle.py
        :language: python
        :start-after: docs: start cutlass-exchange-shuffle
        :end-before: docs: end cutlass-exchange-shuffle
        :dedent: 4

    Give each thread of a block of 64 the scalar of the thread seven ranks
    ahead, wrapping around the block. Thread 0 receives the scalar of
    thread 7, and thread 57 receives that of thread 0.

    `Full example and test on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_collective_documentation.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_collective_documentation.py
        :language: python
        :start-after: # cutlass-rotate-example-begin
        :end-before: # cutlass-rotate-example-end
        :dedent: 4
    """
    if not isinstance(group, ThreadGroup):
        raise TypeError("cuda.coop.cutlass.shuffle group must be a ThreadGroup")
    if group.kind != "block":
        raise NotImplementedError(
            "cuda.coop.cutlass.shuffle requires a block group"
        )
    if not isinstance(mode, str) or isinstance(mode, Enum):
        raise TypeError("cuda.coop.cutlass.shuffle mode must be a string")
    mode = mode.strip().lower().replace("-", "_")
    if mode not in {"up", "down", "offset", "rotate"}:
        raise ValueError(
            "cuda.coop.cutlass.shuffle mode must be up, down, offset, or rotate"
        )
    value = _coerce_thread_payload(
        value,
        scope="cuda.coop.cutlass",
        primitive_name="shuffle",
        arg_name="value",
    )
    from ._compiler._launch import current_kernel_launch_facts
    from ._lowering._shuffle import provider_shuffle

    return provider_shuffle(
        group=group,
        launch=current_kernel_launch_facts(),
        value=value,
        mode=mode,
        distance=distance,
    )


__all__ = ["shuffle"]
