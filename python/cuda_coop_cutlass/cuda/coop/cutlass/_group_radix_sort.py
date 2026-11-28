# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A radix sort orders a block's keys by their bits, one digit at a time.

A digit is a run of neighboring bits of a key. A sort takes the digits from
the lowest bits up, and each pass moves the keys of the :term:`tile` into
the order of one digit. ``radix_rank_keys`` looks at one digit and returns
the position each key would take in a sort by that digit.

.. figure:: visualizations/radix-sort.svg
   :alt: Three rows of eight threads, T0 to T7, with one key and one value
         in each thread. The input row holds the keys 35, 18, 33, 2, 35,
         17, 1, 18 with the values 0 to 7. After a pass on bits 0 to 3 the
         keys are 33, 17, 1, 18, 2, 18, 35, 35. After a pass on bits 4 to 7
         they are 1, 2, 17, 18, 18, 33, 35, 35 with the values 6, 3, 5, 1,
         7, 2, 0, 4.
   :width: 100%

   ``radix_sort_pairs`` with ``end_bit=8`` on eight threads with one key
   each. The value under each key is its position in the input. The middle
   row is the tile in the order of the low four bits, and the bottom row is
   the result. ``radix_rank_keys`` with ``radix_bits=4`` returns the
   position of each input key in the middle row: 6, 3, 0, 4, 7, 1, 2, 5.

The three functions differ in what they return and in the bits they
compare:

.. list-table::
   :header-rows: 1
   :widths: 24 40 36

   * - Function
     - Returns
     - Compares
   * - ``radix_sort_keys``
     - The keys in sorted order.
     - The bits ``[begin_bit, end_bit)``, which is the whole key by
       default.
   * - ``radix_sort_pairs``
     - The keys in sorted order and, at the position of each key, its
       value.
     - The same bits of the keys. The values take no part.
   * - ``radix_rank_keys``
     - For each key, its position in the sorted order, as an ``Int32``.
       The keys stay where they are.
     - One digit of one to eight bits, the low four by default.

All three run on ``this_block()`` and work on the whole tile of that block.
In a launch with several blocks, each block's tile is sorted or ranked by
itself. The order is stable: keys that are equal in the compared bits keep
the order they have in the input, in ascending and in descending order. The
inputs keep their values.

Bit 0 is the lowest bit. Signed and floating-point keys are compared in an
ordered form. ``begin_bit`` and ``end_bit`` are bit positions in that form:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key type
     - Ordered form
   * - ``Uint32``, ``Uint64``
     - The key itself.
   * - ``Int32``, ``Int64``
     - The key with its sign bit flipped. In this form, bit 31 of an
       ``Int32`` key is 0 for a negative key and 1 for every other key.
   * - ``Float32``, ``Float64``
     - The key with its sign bit flipped when that bit is clear, and with
       every bit flipped when it is set. ``-0.0`` is compared as ``+0.0``.
       The two sorts take these keys, and ``radix_rank_keys`` takes
       integer keys only.

A sort returns each key with the bits it had in the input.

The functions take these controls:

.. list-table::
   :header-rows: 1
   :widths: 28 22 50

   * - Control
     - Functions
     - What it does
   * - ``begin_bit``, ``end_bit``
     - All three
     - Select the bits that are compared.
   * - ``radix_bits``
     - ``radix_rank_keys``
     - Sets the width of the digit.
   * - ``descending``
     - All three
     - Sorts or ranks the keys with the largest compared bits first.
   * - ``blocked_to_striped``
     - The two sorts
     - Returns the sorted items in :term:`striped` order.
   * - ``temp_storage``
     - The two sorts
     - Puts the call's scratch in a :ref:`TempStorage <coop-temp-storage>`.
   * - ``exclusive_digit_prefix``
     - ``radix_rank_keys``
     - Receives, for each value of the digit, the number of keys that are
       ranked before the keys with that value.
"""

# Implementation notes.
#
# Adapt radix inputs and preserve public result shapes.
#
# Sort accepts scalar or readable/register payloads and returns fresh keys
# and optional associated values. Rank returns Int32 positions and can also
# write a separate bin-prefix payload. All forms require a complete block.

from __future__ import annotations

from cuda.coop.cutlass._planning.thread_group import ThreadGroup
from cuda.coop.cutlass._typing import CompilerIntegerLike, CompilerScalarLike

from ._temp_storage import TempStorage
from ._thread_data import ThreadData, _snapshot_readable_payload

try:
    import numpy as np
except ModuleNotFoundError as exc:
    if exc.name != "numpy":
        raise
from typing import TypeVar

from cuda.coop.cutlass._typing import IntegerValue

from ._thread_data import CutlassTensorSample, CutlassTensorSSASample
from ._typing import (
    BlockGroup,
    NumericScalar,
    ReadableThreadDataLike,
)

_KeyT = TypeVar(
    "_KeyT",
    bound=(
        "int | np.int32"
        " | np.uint32 | np.int64"
        " | np.uint64 | CompilerIntegerLike"
        " | float | np.float32"
        " | np.float64 | CompilerScalarLike"
    ),
)

_ValueT = TypeVar("_ValueT", bound=NumericScalar)

_RankKeyT = TypeVar(
    "_RankKeyT",
    bound=(
        "int | np.int32"
        " | np.uint32 | np.int64"
        " | np.uint64 | CompilerIntegerLike"
    ),
)


def _validate_group(group):
    """Require the physical block used by CUB radix collectives."""

    if not isinstance(group, ThreadGroup):
        raise TypeError("cuda.coop.cutlass radix group must be a ThreadGroup")
    if group.kind != "block":
        raise NotImplementedError(
            "cuda.coop.cutlass radix requires a complete physical block"
        )


def _input(value, name, primitive):
    """Copy readable payloads and adapt register containers.

    A scalar key or value is adapted as a one-item payload.
    """

    return _snapshot_readable_payload(
        value, name=name, primitive=primitive, allow_scalar=True
    )


def _sort(
    group,
    keys,
    values,
    *,
    begin_bit,
    end_bit,
    descending,
    blocked_to_striped,
    temp_storage,
):
    """Validate sort controls and matching key/value shapes.

    Adapt operands before comparing scalar versus payload form and item count.
    The provider then resolves independent dtypes, bit bounds, exact scratch
    layout, and the blocked or striped output method.
    """

    _validate_group(group)
    primitive = "radix_sort_keys" if values is None else "radix_sort_pairs"
    for name, value in (
        ("descending", descending),
        ("blocked_to_striped", blocked_to_striped),
    ):
        if not isinstance(value, bool):
            raise TypeError(
                f"cuda.coop.cutlass.{primitive} {name} must be "
                "a compile-time bool"
            )
    if temp_storage is not None and not isinstance(temp_storage, TempStorage):
        raise TypeError(
            "cuda.coop.cutlass radix temp_storage must be TempStorage"
        )
    keys = _input(keys, "keys", primitive)
    if values is not None:
        values = _input(values, "values", primitive)
        if isinstance(keys, ThreadData) != isinstance(values, ThreadData):
            raise TypeError(
                "radix_sort_pairs keys and values must have matching "
                "scalar or array shapes"
            )
        if (
            isinstance(keys, ThreadData)
            and keys.items_per_thread != values.items_per_thread
        ):
            raise ValueError(
                "radix_sort_pairs keys and values must have matching "
                "items_per_thread"
            )
    from ._compiler._launch import current_kernel_launch_facts
    from ._lowering._radix_sort import provider_radix_sort

    return provider_radix_sort(
        group=group,
        launch=current_kernel_launch_facts(),
        keys=keys,
        values=values,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        blocked_to_striped=blocked_to_striped,
        temp_storage=temp_storage,
    )


def radix_sort_keys(
    group: BlockGroup,
    keys: ReadableThreadDataLike[_KeyT]
    | CutlassTensorSample
    | CutlassTensorSSASample
    | _KeyT,
    /,
    *,
    begin_bit: IntegerValue = 0,
    end_bit: IntegerValue | None = None,
    descending: bool = False,
    temp_storage: TempStorage | None = None,
    blocked_to_striped: bool = False,
) -> ThreadData | _KeyT:
    """Sort the keys of a block's tile by their bits.

    The call orders the keys by the bits ``[begin_bit, end_bit)`` of each
    key, which is the whole key by default. The order is ascending, or
    descending with ``descending=True``. Keys that are equal in those bits
    keep their input order, whatever their other bits are.

    Parameters
    ----------
    group : ThreadGroup
        ``this_block()``, on a block of any shape and size, such as the 48
        threads of ``block=(8, 3, 2)``. Every thread of the block makes the
        call, on every trip of a loop that contains it. The block size
        times the item count is at most 65,535. A larger tile is rejected
        when the kernel is compiled, and so is any other group.
    keys : scalar, ThreadData, CuTe register tensor or TensorSSA
        This thread's keys: one scalar or a payload. The element type is
        ``Int32``, ``Uint32``, ``Int64``, ``Uint64``, ``Float32`` or
        ``Float64``, and another type is rejected when the kernel is
        compiled. The tile is read in :term:`blocked` order: item ``i`` of
        the thread with rank ``t`` is at position
        ``t * items_per_thread + i``.

        A register tensor or ``TensorSSA`` is read as a payload, and the
        result is a ``ThreadData``. An object of your own that has
        ``items_per_thread``, ``dtype``, ``len()`` and integer indexing is
        read as a payload too. A tensor in global or shared memory is
        rejected when the kernel is compiled, so read it with
        :func:`~cuda.coop.cutlass.load` first. A ``ThreadData`` of one item
        is a payload. For the scalar form, pass ``keys[0]``.
    begin_bit : int or CuTe integer, optional
        The lowest bit that is compared. The default is 0. Bit 0 is the
        lowest bit of the key's ordered form, which the top of the
        :ref:`Radix sorting and ranking <coop-cutlass-radix>` section
        describes.
    end_bit : int or CuTe integer, optional
        The bit after the highest bit that is compared. The default is the
        width of the key, 32 or 64 bits, so ``begin_bit=3`` by itself
        compares bit 3 and every bit above it. The two bounds satisfy
        ``0 <= begin_bit < end_bit <= width``.

        Each bound is a Python or NumPy integer, fixed when the kernel is
        compiled, or a CuTe integer of type ``Int8`` to ``Int64`` or
        ``Uint8`` to ``Uint32``, read while the kernel runs. One bound can
        be a Python value while the other is a CuTe value. Python bounds
        that break the rule are rejected when the kernel is compiled. A
        CuTe bound is checked while the kernel runs, and a call whose
        bounds break the rule traps on the device.

        Every thread of the block passes the same bounds. The package
        checks only that each thread's bounds are in range.
    descending : bool, optional
        A Python ``bool``. ``True`` puts the keys with the largest compared
        bits first. Tied keys keep their input order in both directions, so
        the result can differ from an ascending result read backwards. The
        default is ``False``.
    temp_storage : TempStorage, optional
        Every sort needs scratch. Leave this out and the compiler provides
        the scratch and a barrier after the call. Pass a
        :ref:`TempStorage <coop-temp-storage>` to share the bytes with
        other calls. A barrier then follows the call only if the
        ``TempStorage`` was created with ``auto_sync=True``.
    blocked_to_striped : bool, optional
        A Python ``bool``. The default, ``False``, returns the result in
        blocked order: item ``i`` of the thread with rank ``t`` holds
        sorted position ``t * items_per_thread + i``. ``True`` returns it
        in :term:`striped` order: on a block of ``G`` threads, that item
        holds sorted position ``t + i * G``. Write a striped result with
        ``store(..., algorithm="striped")``. The input is read in blocked
        order with either setting.

    Returns
    -------
    CuTe scalar or cuda.coop.cutlass.ThreadData
        This thread's part of the sorted tile, in the element type of
        ``keys``. A scalar gives a scalar. A payload gives a new payload
        with the same item count and alignment. Every thread's result is
        defined, and ``keys`` keeps its values.

    Notes
    -----
    A sort makes one pass for every four bits it compares, and one more for
    any bits left over. A whole 32-bit key takes eight passes, a whole
    64-bit key sixteen, and a range of six bits two.

    With ``Float32`` or ``Float64`` keys, ``-0.0`` and ``+0.0`` are equal
    and keep their input order. NaN keys are compared by the bits of their
    ordered form. In an ascending sort of the whole key, a NaN with the
    sign bit set comes before negative infinity, and a NaN with the sign
    bit clear comes after positive infinity.

    See Also
    --------
    :cpp:struct:`cub::BlockRadixSort`
        The CUB class behind the call.

    Examples
    --------
    ``sort_digits`` sorts one payload, ``keys``, twice on a block of 64
    threads. ``ordered`` holds the keys in ascending order. The pair sort
    orders the same keys by their low four bits, largest first, with each
    key's position in the tile as its value. Its two results are striped,
    so their ``store`` calls use ``algorithm="striped"``.

    `Full example and test on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_radix_sort_documentation.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_radix_sort_documentation.py
        :language: python
        :start-after: # cutlass-radix-sort-example-begin
        :end-before: # cutlass-radix-sort-example-end
        :dedent: 4
    """
    return _sort(
        group,
        keys,
        None,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        blocked_to_striped=blocked_to_striped,
        temp_storage=temp_storage,
    )


def radix_sort_pairs(
    group: BlockGroup,
    keys: ReadableThreadDataLike[_KeyT]
    | CutlassTensorSample
    | CutlassTensorSSASample
    | _KeyT,
    values: ReadableThreadDataLike[_ValueT]
    | CutlassTensorSample
    | CutlassTensorSSASample
    | _ValueT,
    /,
    *,
    begin_bit: IntegerValue = 0,
    end_bit: IntegerValue | None = None,
    descending: bool = False,
    temp_storage: TempStorage | None = None,
    blocked_to_striped: bool = False,
) -> tuple[ThreadData, ThreadData] | tuple[_KeyT, _ValueT]:
    """Sort the keys of a block's tile and move each key's value with it.

    The same as :func:`~cuda.coop.cutlass.radix_sort_keys` with a second
    operand, ``values``. The other parameters are described there.

    Parameters
    ----------
    values : scalar, ThreadData, CuTe register tensor or TensorSSA
        This thread's values, one for each key: item ``i`` of ``values``
        belongs to item ``i`` of ``keys``. Pass two scalars, or two
        payloads with the same item count. A scalar with a payload is
        rejected when the kernel is compiled, and so are two payloads with
        different item counts.

        A value has one of the ten element types listed under
        :ref:`ThreadData <coop-thread-data>`. The value type can differ
        from the key type. A payload of values takes the forms that
        ``keys`` takes, and the two can have different forms, such as a
        ``ThreadData`` and a register tensor.

    Returns
    -------
    tuple of CuTe scalars or tuple of cuda.coop.cutlass.ThreadData
        ``(sorted_keys, sorted_values)``. The keys are placed as
        ``radix_sort_keys`` places them, and each value is at the position
        of its key. With ``blocked_to_striped=True``, both results are
        striped. Each result has the element type of its input. Two
        scalars give two scalars. Two payloads give two new payloads, each
        with the item count and alignment of its input. ``keys`` and
        ``values`` keep their values.

    See Also
    --------
    :cpp:struct:`cub::BlockRadixSort`
        The CUB class behind the call.

    Examples
    --------
    ``pair_keys`` below holds the keys in descending order of their low
    four bits, and ``pair_indices`` holds the position in the tile that
    each of those keys came from. Both results are striped, so the two
    ``store`` calls that write them use ``algorithm="striped"``.

    `Full example and test on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_radix_sort_documentation.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_radix_sort_documentation.py
        :language: python
        :start-after: # cutlass-radix-sort-example-begin
        :end-before: # cutlass-radix-sort-example-end
        :dedent: 4
    """
    if values is None:
        raise TypeError(
            "radix_sort_pairs values must be a numeric scalar or payload"
        )
    return _sort(
        group,
        keys,
        values,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        blocked_to_striped=blocked_to_striped,
        temp_storage=temp_storage,
    )


def radix_rank_keys(
    group: BlockGroup,
    keys: ReadableThreadDataLike[_RankKeyT]
    | CutlassTensorSample
    | CutlassTensorSSASample
    | _RankKeyT,
    /,
    *,
    begin_bit: int = 0,
    end_bit: int | None = None,
    radix_bits: int | None = None,
    descending: bool = False,
    exclusive_digit_prefix: ThreadData | None = None,
) -> ThreadData | CompilerIntegerLike:
    """Find where each key of a block's tile goes in a sort by one digit.

    The digit is the bits ``[begin_bit, end_bit)`` of each key, one to
    eight bits wide. A key's rank is the number of keys in the tile whose
    digit is smaller, or greater with ``descending=True``, plus the number
    of keys with the same digit that come before it in the input. The ranks
    of a tile are the numbers from 0 to the tile size less one, each once.
    Moving every key to its rank sorts the tile by that digit.

    Parameters
    ----------
    group : ThreadGroup
        ``this_block()``, under the rules that
        :func:`~cuda.coop.cutlass.radix_sort_keys` gives for its group.
    keys : scalar, ThreadData, CuTe register tensor or TensorSSA
        This thread's keys, in the forms and the order that
        ``radix_sort_keys`` takes. The element type is ``Int32``,
        ``Uint32``, ``Int64`` or ``Uint64``, and another type is rejected
        when the kernel is compiled.
    begin_bit : int, optional
        The lowest bit of the digit. The default is 0. Bit 0 is the lowest
        bit of the key's ordered form, which the top of the
        :ref:`Radix sorting and ranking <coop-cutlass-radix>` section
        describes.
    end_bit : int, optional
        The bit after the highest bit of the digit. The default is
        ``begin_bit + radix_bits``.
    radix_bits : int, optional
        The width of the digit in bits, from 1 to 8. The default is
        ``end_bit - begin_bit``, or 4 when ``end_bit`` is left out as well.
        When both are given, ``radix_bits`` must equal
        ``end_bit - begin_bit``.

        ``begin_bit``, ``end_bit`` and ``radix_bits`` are Python or NumPy
        integers. A CuTe value is rejected when the kernel is compiled, and
        so is a digit that is wider than eight bits or reaches past the
        key. With ``begin_bit=30`` on a 32-bit key, the default width of 4
        reaches past the key, so pass ``radix_bits`` of 1 or 2.
    descending : bool, optional
        A Python ``bool``. ``True`` ranks the keys with the largest digit
        first. Keys with the same digit are ranked in input order in both
        directions. The default is ``False``.
    exclusive_digit_prefix : ThreadData, optional
        A payload that the call fills with one count for each value of the
        digit: the number of keys with a smaller digit, or with a greater
        digit when ``descending=True``. The count is the rank at which the
        keys with that digit begin.

        For a digit of ``R`` bits on a block of ``G`` threads, create the
        payload with ``P = max(1, ceil(2**R / G))`` items, as in
        ``coop.ThreadData(P)``. Leave its ``dtype`` out or give ``Int32``.
        It needs no values before the call. The thread with rank ``t`` in
        the block receives the count for digit value ``t * P + i`` in item
        ``i``, in both directions. On a block of 64 threads, a four-bit
        digit takes one item per thread, and threads 0 to 15 receive the
        sixteen counts. Items that stand for a digit value of ``2**R`` or
        more are undefined, so write the payload with
        ``store(..., valid_items=2**R)``.

        Pass a ``ThreadData`` other than ``keys``. A register tensor is
        rejected when the kernel is compiled. When the call is inside a
        runtime ``if`` or loop, create the payload there and read its
        items there, as in :ref:`this FAQ <coop-faq-scan-loop-aggregate>`.

    Returns
    -------
    CuTe scalar or cuda.coop.cutlass.ThreadData
        The ``Int32`` rank of each of this thread's keys: item ``i`` of the
        result is the rank of item ``i`` of ``keys``. A scalar gives a
        scalar. A payload gives a new payload with the same item count and
        alignment. Every thread's result is defined, and ``keys`` keeps its
        values.

    Notes
    -----
    The call provides its own scratch and a barrier after it, so it takes
    no ``temp_storage``. The scratch grows with the width of the digit.
    With CUB from CCCL 3.7, a digit of ``R`` bits needs ``2**(R - 1) + 1``
    four-byte counters for every thread, as well as the scratch of a block
    scan. That is 36 bytes of counters per thread for four bits and 516 for
    eight.

    See Also
    --------
    :cpp:struct:`cub::BlockRadixRank`
        The CUB class behind the call.

    Examples
    --------
    ``rank_digits`` ranks the keys of a block of 64 threads by their low
    four bits. A four-bit digit has sixteen values, so ``prefixes`` has one
    item per thread and threads 0 to 15 receive the sixteen counts. The
    second ``store`` writes those sixteen items.

    `Full example and test on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_radix_rank_documentation.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_radix_rank_documentation.py
        :language: python
        :start-after: # cutlass-radix-rank-example-begin
        :end-before: # cutlass-radix-rank-example-end
        :dedent: 4
    """
    _validate_group(group)
    if not isinstance(descending, bool):
        raise TypeError(
            "cuda.coop.cutlass.radix_rank_keys descending must be "
            "a compile-time bool"
        )
    if exclusive_digit_prefix is keys:
        raise ValueError(
            "radix_rank_keys exclusive_digit_prefix must be distinct from keys"
        )
    keys = _input(keys, "keys", "radix_rank_keys")
    from ._compiler._launch import current_kernel_launch_facts
    from ._lowering._radix_sort import provider_radix_rank

    return provider_radix_rank(
        group=group,
        launch=current_kernel_launch_facts(),
        keys=keys,
        begin_bit=begin_bit,
        end_bit=end_bit,
        radix_bits=radix_bits,
        descending=descending,
        exclusive_digit_prefix=exclusive_digit_prefix,
    )


__all__ = ["radix_rank_keys", "radix_sort_keys", "radix_sort_pairs"]
