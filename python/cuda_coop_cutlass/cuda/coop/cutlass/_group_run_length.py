# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run-length decoding repeats each run's value as many times as its length.

A run is a value and a length: the values ``[7, 9]`` with the lengths
``[3, 2]`` decode to ``7, 7, 7, 9, 9``. Each thread passes its runs as a
payload of values and a payload of lengths. The runs of a block, taken
thread by thread, decode to one sequence of items, the block's decoded
stream.

.. figure:: visualizations/run-length-decode.svg
   :alt: Four threads hold one run each. The run values are 7, 9, 2 and 5,
         and the run lengths are 3, 2, 0 and 0. A row of starts reads 0, 3,
         5 and 5. A row of window values reads 7, 9, 9 and 0.
   :width: 100%

   ``run_length_decode`` on a block of four threads with one run each,
   ``decoded_items_per_thread=1`` and ``decoded_window_offset=2``. The
   stream is 7, 7, 7, 9, 9, and the two runs of length zero add nothing to
   it. The third row is the position in the stream where each run starts,
   counted from 0. The window is positions 2 to 5. Position 5 is past the
   end of the stream, so thread 3 receives zero.

The two functions decode the same runs and differ in where the items go:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * -
     - ``run_length_decode``
     - ``run_length_decode_into``
   * - Decodes
     - One window of the stream: ``decoded_items_per_thread`` items for
       each thread, from ``decoded_window_offset`` on
     - The whole stream
   * - The items go to
     - A new payload in each thread
     - A tensor in global memory, from ``destination_offset`` on
   * - Returns
     - The payload
     - The length of the stream, to every thread
   * - Past the end of the stream
     - The items hold zero
     - Nothing is written

Every call starts from the runs: it checks the lengths and finds where each
run starts before it decodes. Nothing is kept from one call to the next,
even when the calls share a :ref:`TempStorage <coop-temp-storage>`. A loop
of ``run_length_decode`` calls does that work on every trip, and the kernel
can use the items of each window. One ``run_length_decode_into`` call does
the work once and stores the items itself.
"""

# Implementation notes.
#
# CUTLASS entry points for block run-length decoding.
#
# Each call copies the per-thread run values and lengths from ThreadData, CuTe
# register tensors, or TensorSSA values, so inputs stay unchanged. Values and
# lengths must have the same per-thread count. run_length_decode returns one
# decoded window as ThreadData. run_length_decode_into writes the whole stream
# to a global tensor and returns its length.

from __future__ import annotations

from typing import TypeVar

from cuda.coop.cutlass._planning.thread_group import ThreadGroup
from cuda.coop.cutlass._typing import CompilerIntegerLike

from ._temp_storage import TempStorage
from ._thread_data import (
    CutlassTensorSample,
    CutlassTensorSSASample,
    ThreadData,
    _snapshot_readable_payload,
)
from ._typing import (
    BlockGroup,
    IntegralScalar,
    NumericScalar,
    ReadableThreadDataLike,
    TempStorageLike,
)

_LengthT = TypeVar("_LengthT", bound=IntegralScalar)

_ItemT = TypeVar("_ItemT", bound=NumericScalar)


def _decode(
    group,
    run_values,
    run_lengths,
    *,
    decoded_items_per_thread,
    offset,
    destination,
    bulk,
    temp_storage,
):
    """Validate the block group and copy both run payloads.

    Values and lengths must have the same per-thread extent. The decoded
    extent is a separate compile-time control. The provider resolves the
    value and length types, the static or runtime offset type, and the
    destination checks. It then selects the window or bulk C++ wrapper.
    """

    primitive = "run_length_decode_into" if bulk else "run_length_decode"
    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{primitive} group must be a ThreadGroup")
    if group.kind != "block":
        raise NotImplementedError(
            f"{primitive} requires a complete this_block() group"
        )
    if temp_storage is not None and not isinstance(temp_storage, TempStorage):
        raise TypeError(f"{primitive} temp_storage must be CUTLASS TempStorage")
    values = _snapshot_readable_payload(
        run_values, name="run_values", primitive=primitive
    )
    lengths = _snapshot_readable_payload(
        run_lengths, name="run_lengths", primitive=primitive
    )
    if values.items_per_thread != lengths.items_per_thread:
        raise ValueError(
            f"{primitive} run inputs must have matching items_per_thread"
        )

    from ._compiler._launch import current_kernel_launch_facts
    from ._lowering._run_length import provider_run_length_decode

    return provider_run_length_decode(
        group=group,
        launch=current_kernel_launch_facts(),
        values=values,
        lengths=lengths,
        decoded_items_per_thread=decoded_items_per_thread,
        offset=offset,
        destination=destination,
        bulk=bulk,
        temp_storage=temp_storage,
    )


def run_length_decode(
    group: BlockGroup,
    run_values: ReadableThreadDataLike[_ItemT]
    | CutlassTensorSample
    | CutlassTensorSSASample,
    run_lengths: ReadableThreadDataLike[_LengthT]
    | CutlassTensorSample
    | CutlassTensorSSASample,
    /,
    *,
    decoded_items_per_thread: int,
    decoded_window_offset: IntegralScalar = 0,
    temp_storage: TempStorageLike | None = None,
) -> ThreadData:
    """Decode one window of a block's runs into a payload.

    A window is the block size times ``decoded_items_per_thread``
    consecutive positions of the decoded stream. Positions count from 0,
    and each thread receives ``decoded_items_per_thread`` of them.

    Parameters
    ----------
    group : ThreadGroup
        ``this_block()``, on a one-dimensional block of any size. Every
        thread of the block makes the call, on every trip of a loop that
        contains it, including a thread whose run lengths are all zero. Any
        other group is rejected when the kernel is compiled, and so is a
        block with two or three dimensions.
    run_values : ThreadData, CuTe register tensor or TensorSSA
        The values of this thread's runs, in a payload. The element type is
        one of the ten listed under :ref:`ThreadData <coop-thread-data>`.
        A register tensor or ``TensorSSA`` is read as a payload. For one
        run per thread, pass a payload of one item. A scalar is rejected
        when the kernel is compiled, and so is a tensor in global or shared
        memory, which :func:`~cuda.coop.cutlass.load` reads into a payload.
    run_lengths : ThreadData, CuTe register tensor or TensorSSA
        The lengths of this thread's runs, in a payload with the item count
        of ``run_values``. Item ``i`` is the length of the run whose value
        is item ``i`` of ``run_values``. The element type is a signed or
        unsigned integer of 8, 16, 32 or 64 bits, whatever the element type
        of ``run_values``. Another type or a different item count is
        rejected when the kernel is compiled.

        The runs of the block are in :term:`blocked` order. With ``r`` runs
        per thread, item ``i`` of the thread with rank ``t`` in the block
        is run ``t * r + i``. In that order every positive length comes
        before every zero. A run of length zero is padding. It lets a block
        decode fewer runs than its payloads hold, and its value never
        appears in the output. A block whose lengths are all zero has an
        empty stream. The lengths of a block add up to at most
        4,294,967,295.

        A negative length, a positive length after a zero, and a sum above
        that limit each trap on the device before anything is decoded.
    decoded_items_per_thread : int
        The number of items each thread receives, a positive Python
        integer. A value that is not a positive integer is rejected when
        the kernel is compiled. The number is independent of the number of
        runs per thread.
    decoded_window_offset : int or CuTe integer, optional
        The position in the stream of the window's first item. The default
        is 0. Positions count decoded items, so a window can start in the
        middle of a run. Item ``i`` of the thread with rank ``t`` holds
        position
        ``decoded_window_offset + t * decoded_items_per_thread + i``.

        The offset is a Python or NumPy integer from 0 to ``2**64 - 1``, or
        a CuTe integer computed while the kernel runs. A CuTe integer can
        be signed or unsigned, of 8, 16, 32 or 64 bits. Every thread of the
        block passes the same value. Any other Python value, ``True`` and
        ``False`` included, is rejected when the kernel is compiled. A
        negative CuTe value traps on the device. An offset equal to the
        length of the stream or larger gives a window of zeros.
    temp_storage : TempStorage, optional
        The call uses scratch. Leave this out and the compiler provides the
        scratch and a barrier after the call. Pass a
        :ref:`TempStorage <coop-temp-storage>` to share the bytes with
        other calls. A barrier then follows the call only if the
        ``TempStorage`` was created with ``auto_sync=True``. Without it,
        call :meth:`~cuda.coop.cutlass.TempStorage.sync` before the bytes
        are used again, which in a loop is before the next trip.

    Returns
    -------
    cuda.coop.cutlass.ThreadData
        A new payload of ``decoded_items_per_thread`` items in the element
        type of ``run_values``. Every item is defined, and an item past the
        end of the stream is zero. Zero can also be a run value, and the
        call does not return the length of the stream, so the kernel works
        out from the run lengths how many items of a window were decoded.
        ``run_values`` and ``run_lengths`` keep their values. The payload
        has the alignment of a ``ThreadData`` passed as ``run_values``.

    Notes
    -----
    In both functions the block size times ``decoded_items_per_thread``
    must be less than ``2**31``, and so must the block size times the number
    of runs per thread. A larger product is rejected when the kernel is
    compiled.

    CUB's class is constructed from the runs once and then decodes window
    after window. Here every call is complete by itself, as the top of the
    :ref:`Run Length Decode <coop-cutlass-run-length>` section says. CUB
    can also give each item's position within its run, which has no
    counterpart here.

    See Also
    --------
    :cpp:struct:`cub::BlockRunLengthDecode`
        The CUB class behind the call.

    Examples
    --------
    A window that starts inside a run. With ``items_per_thread=1`` the 32
    threads hold the values 10 to 41, each with length 2, so the stream has
    64 items. The window starts at position 3, the second item of the
    second run, and thread 0 receives 11 and 12. The last three positions
    of the window are past the end of the stream: thread 30 receives 41 and
    0, and thread 31 receives 0 and 0. With ``items_per_thread=4`` the
    stream has 256 items and the whole window is inside it.
    ``run_length_decode_into`` then writes the whole stream to
    ``destination`` from element 5 on. The example under
    :func:`~cuda.coop.cutlass.run_length_decode_into` decodes a whole stream
    with a loop of window calls.

    `Full example and test on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_run_length_examples.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_run_length_examples.py
        :language: python
        :start-after: # example-begin
        :end-before: # example-end
        :dedent: 4
    """
    return _decode(
        group,
        run_values,
        run_lengths,
        decoded_items_per_thread=decoded_items_per_thread,
        offset=decoded_window_offset,
        destination=None,
        bulk=False,
        temp_storage=temp_storage,
    )


def run_length_decode_into(
    group: BlockGroup,
    run_values: ReadableThreadDataLike[_ItemT]
    | CutlassTensorSample
    | CutlassTensorSSASample,
    run_lengths: ReadableThreadDataLike[_LengthT]
    | CutlassTensorSample
    | CutlassTensorSSASample,
    destination: CutlassTensorSample,
    /,
    *,
    decoded_items_per_thread: int,
    destination_offset: IntegralScalar = 0,
    temp_storage: TempStorageLike | None = None,
) -> CompilerIntegerLike:
    """Decode all of a block's runs into a tensor in global memory.

    The call checks the lengths once. It then decodes the stream window by
    window and stores each window, until the whole stream is written. A
    window is the block size times ``decoded_items_per_thread`` items.

    Parameters
    ----------
    group : ThreadGroup
        ``this_block()``, on a one-dimensional block, as in
        :func:`~cuda.coop.cutlass.run_length_decode`.
    run_values : ThreadData, CuTe register tensor or TensorSSA
        The values of this thread's runs, as in
        :func:`~cuda.coop.cutlass.run_length_decode`.
    run_lengths : ThreadData, CuTe register tensor or TensorSSA
        The lengths of this thread's runs, with the rules of
        :func:`~cuda.coop.cutlass.run_length_decode`. The lengths that trap
        on the device in that function trap in this one before anything is
        written.
    destination : cute.Tensor
        The memory that receives the stream: a tensor in global memory with
        one dimension, stride 1 and the element type of ``run_values``,
        such as ``cute.make_tensor(pointer, cute.make_layout(n))``. Its
        number of elements, ``n`` here, is a Python integer or a CuTe
        integer. A pointer has no size and is rejected when the kernel is
        compiled. So are a tensor in shared memory or in registers, a
        tensor with two dimensions or another stride, and a tensor of
        another element type.

        Every thread of the block passes the same tensor. In a kernel with
        more than one block, each block writes its own stream. Give each
        block its own part of memory, through the tensor or through
        ``destination_offset``.
    decoded_items_per_thread : int
        The number of items each thread decodes and stores in one window, a
        positive Python integer. A value that is not a positive integer is
        rejected when the kernel is compiled. The last window stores only
        the items that remain. A larger number means fewer and larger
        windows. Each thread holds its ``decoded_items_per_thread`` items
        of a window in registers until it stores them. The number changes
        neither what is written nor the size of the scratch.
    destination_offset : int or CuTe integer, optional
        The element of ``destination`` that receives the first item of the
        stream. The default is 0. The whole stream follows from there, so
        ``destination`` needs at least ``destination_offset`` plus the
        length of the stream in elements. With fewer, the call traps on the
        device before anything is written. The check is made while the
        kernel runs, for Python integers too, and an empty stream still
        needs ``destination_offset`` elements.

        The offset is a Python or NumPy integer from 0 to ``2**64 - 1``, or
        a CuTe integer computed while the kernel runs. A CuTe integer can
        be signed or unsigned, of 8, 16, 32 or 64 bits. Every thread of the
        block passes the same value. Any other Python value, ``True`` and
        ``False`` included, is rejected when the kernel is compiled. A
        negative CuTe value traps on the device.
    temp_storage : TempStorage, optional
        Scratch for the call, as in
        :func:`~cuda.coop.cutlass.run_length_decode`.

    Returns
    -------
    cutlass.Uint32
        The length of the decoded stream, the same in every thread of the
        block. The stream is in elements ``destination_offset`` to
        ``destination_offset + length - 1`` of ``destination``, and every
        other element is as it was. An empty stream writes nothing and
        returns 0. ``run_values`` and ``run_lengths`` keep their values.

    See Also
    --------
    :cpp:struct:`cub::BlockRunLengthDecode`
        The CUB class behind the call.

    Examples
    --------
    A loop of ``run_length_decode`` calls and one ``run_length_decode_into``
    call write the same stream. With ``items_per_thread=1`` there are 32
    runs, 31 of length 3 and the last of length 1, so the stream has 94
    items. The loop makes three trips, for the windows that start at 0, 32
    and 64, and ``valid_items`` limits the last store to 30 items. Both
    write the stream from element 5 on, and every thread receives 94 in
    ``total``. With ``items_per_thread=4`` the stream has 382 items and the
    loop makes 12 trips.

    `Full example and test on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_run_length_documentation.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_run_length_documentation.py
        :language: python
        :start-after: # cutlass-run-length-example-begin
        :end-before: # cutlass-run-length-example-end
        :dedent: 4
    """
    return _decode(
        group,
        run_values,
        run_lengths,
        decoded_items_per_thread=decoded_items_per_thread,
        offset=destination_offset,
        destination=destination,
        bulk=True,
        temp_storage=temp_storage,
    )


__all__ = ["run_length_decode", "run_length_decode_into"]
