# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Provide common block Radix Sort and Radix Rank calls.

Common calls take readable integer-key payloads in blocked order. Qualified
operations document their additional payload forms and key dtypes.
Decorators register these functions so compilers can recognize their calls.
A CuTe DSL trace executes the Python bodies, which check common restrictions
before calling the backend. Numba-CUDA-MLIR replaces calls during compilation
and checks its typed operands separately.

The static bound helper resolves Rank's digit interval for the common check
and both compiler frontends. Sort bounds may be runtime values; a separate
range check and a device trap cover them.
"""

from __future__ import annotations

from numbers import Integral

from cuda.coop._typing import CompilerIntegerLike

from .._bindings import ArgumentBinding
from ..block.radix import make_radix_bit_range
from ._dispatch import (
    _backend_module_name,
    _common_group_operation,
    _group_primitive_marker,
    _validate_common_operation_group,
)
from ._payload import (
    TempStorageLike,
    _common_thread_data_extent,
    _validate_common_integer_value,
    _validate_common_numeric_value,
    _validate_common_temp_storage,
)

try:
    import numpy
except ModuleNotFoundError as exc:
    if exc.name != "numpy":
        raise
from typing import TypeVar

from cuda.coop._typing import (
    CommonNumericScalar,
    CommonThreadDataLike,
    IntegerValue,
    ThreadDataLike,
)

from .thread_group import BlockGroup

_ValueT = TypeVar("_ValueT", bound=CommonNumericScalar)

_KeyT = TypeVar(
    "_KeyT",
    bound=(
        "int | numpy.int32"
        " | numpy.uint32 | numpy.int64"
        " | numpy.uint64 | CompilerIntegerLike"
    ),
)

_RankKeyT = TypeVar(
    "_RankKeyT",
    bound=(
        "int | numpy.int32"
        " | numpy.uint32 | numpy.int64"
        " | numpy.uint64 | CompilerIntegerLike"
    ),
)


def _radix_bounds(operation, key_width, begin_bit, end_bit, radix_bits=None):
    """Resolve static radix defaults and check the common API's interval.

    Current callers use this helper only for Rank. Rank defaults to four bits
    from begin, unless radix_bits or end is supplied, and permits at most
    eight selected bits. An explicit radix_bits must agree with the resolved
    interval. The helper also defines a full-key-width default for Sort, but
    does not handle runtime Sort bounds.
    """

    for name, value in (
        ("begin_bit", begin_bit),
        ("end_bit", end_bit),
        ("radix_bits", radix_bits),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, Integral)
        ):
            raise TypeError(
                f"cuda.coop.{operation} {name} must be a compile-time integer"
            )
    if radix_bits is not None and radix_bits <= 0:
        raise ValueError("radix_bits must be positive")
    if end_bit is None:
        end_bit = (
            begin_bit + (4 if radix_bits is None else radix_bits)
            if operation == "radix_rank_keys"
            else key_width
        )
    if radix_bits is not None and end_bit - begin_bit != radix_bits:
        raise ValueError("radix_bits must match end_bit - begin_bit")
    make_radix_bit_range(
        begin_bit=begin_bit, end_bit=end_bit, bit_width=key_width
    )
    if operation == "radix_rank_keys" and end_bit - begin_bit > 8:
        raise ValueError("radix_rank_keys bit width must be <= 8")
    return int(begin_bit), int(end_bit)


def _validate(
    operation,
    group,
    keys,
    values,
    begin_bit,
    end_bit,
    descending,
    temp_storage,
    radix_bits=None,
):
    """Check common Radix operands while a CuTe DSL trace runs this body.

    Require supported integer-key payloads, matching pair extents, and a
    compile-time bool for descending. Sort bounds may be runtime integers;
    Rank requires a static digit interval. With no active Python tracing
    backend, return at once; the dispatch call that follows reports the
    missing compiler context. Numba-CUDA-MLIR replaces these calls during
    compilation and checks its typed operands.
    """

    if _backend_module_name() is None:
        return
    _validate_common_operation_group(operation, group)
    name = _validate_common_numeric_value(
        operation,
        "keys",
        keys,
        require_thread_data=True,
        allow_readonly_thread_data=True,
    )
    if name not in {"int32", "uint32", "int64", "uint64"}:
        raise TypeError(
            f"cuda.coop.{operation} keys require "
            "int32, uint32, int64, or uint64"
        )
    if operation == "radix_sort_pairs":
        _validate_common_numeric_value(
            operation,
            "values",
            values,
            require_thread_data=True,
            allow_readonly_thread_data=True,
        )
        if _common_thread_data_extent(
            operation, "keys", keys
        ) != _common_thread_data_extent(operation, "values", values):
            raise ValueError(
                "keys and values must have the same items_per_thread"
            )
    if not isinstance(descending, bool):
        raise TypeError(
            f"cuda.coop.{operation} descending must be a compile-time bool"
        )
    width = int(name[-2:])
    if operation == "radix_rank_keys":
        _radix_bounds(operation, width, begin_bit, end_bit, radix_bits)
    else:
        begin = _validate_common_integer_value(
            operation, "begin_bit", begin_bit
        )
        end = (
            width
            if end_bit is None
            else _validate_common_integer_value(operation, "end_bit", end_bit)
        )
        make_radix_bit_range(
            begin_bit=ArgumentBinding.runtime() if begin is None else begin,
            end_bit=ArgumentBinding.runtime() if end is None else end,
            bit_width=width,
        )
    if temp_storage is not None:
        _validate_common_temp_storage(operation, temp_storage)


@_common_group_operation("radix_sort_keys", group_kinds=("block",))
def radix_sort_keys(
    group: BlockGroup,
    keys: CommonThreadDataLike[_KeyT],
    /,
    *,
    begin_bit: IntegerValue = 0,
    end_bit: IntegerValue | None = None,
    descending: bool = False,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_KeyT]:
    """Return stable, blocked radix-sorted integral keys without mutation.

    Parameters
    ----------
    group : ThreadGroup
        The complete physical block returned by ``this_block()``. All block
        threads must participate with identical options and payload extents.
    keys : ThreadDataLike
        Fixed-size per-thread keys with int32, uint32, int64, or uint64 dtype.
        The input sequence is the flattened blocked arrangement.
    begin_bit, end_bit : int or compiler integer
        Half-open interval in CUB's ordered key representation. The default
        begin is zero; omitted end selects the full key width, even when begin
        is nonzero. Bounds may be runtime values but must be block-uniform and
        satisfy ``0 <= begin_bit < end_bit <= key_width``. Known bounds are
        checked during compilation. Invalid runtime bounds trap before
        conversion to CUB's integer arguments.
    descending : bool
        Compile-time selector for descending instead of ascending digit order.
    temp_storage : TempStorageLike, optional
        Caller-owned block scratch. Omit to allocate scratch automatically.
        An explicit descriptor must satisfy the specialization's size and
        alignment. With ``auto_sync=False``, the caller synchronizes before
        reusing it.

    Returns
    -------
    ThreadDataLike
        Sorted keys in blocked arrangement with the input dtype and extent.
        The input payload is preserved. Equal selected digits retain their
        original blocked order, including for descending sorts.

    Notes
    -----
    Wraps CUB ``BlockRadixSort::Sort`` or ``SortDescending``. For signed
    integers, the sign bit is inverted before selecting the bit interval,
    then restored in the returned keys. Qualified ``cuda.coop.numba_mlir``
    and ``cuda.coop.cutlass`` calls also support floating-point keys, scalar
    payloads, and striped output. Numba-CUDA-MLIR additionally accepts local
    arrays; CUTLASS accepts CuTe register tensors.

    See Also
    --------
    cuda.coop.numba_mlir.radix_sort_keys
        Numba-CUDA-MLIR payloads and qualified controls.
    cuda.coop.cutlass.radix_sort_keys
        CuTe payloads and qualified controls.

    Examples
    --------
    Sort unsigned keys by their low byte in descending order. Keys with
    equal low bytes retain their original order.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_radix_examples.py
        :language: python
        :start-after: # radix-sort-keys-example-begin
        :end-before: # radix-sort-keys-example-end
        :dedent: 4
    """
    _validate(
        "radix_sort_keys",
        group,
        keys,
        None,
        begin_bit,
        end_bit,
        descending,
        temp_storage,
    )
    return _group_primitive_marker(
        "radix_sort_keys",
        group,
        keys,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        temp_storage=temp_storage,
    )


@_common_group_operation("radix_sort_pairs", group_kinds=("block",))
def radix_sort_pairs(
    group: BlockGroup,
    keys: CommonThreadDataLike[_KeyT],
    values: CommonThreadDataLike[_ValueT],
    /,
    *,
    begin_bit: IntegerValue = 0,
    end_bit: IntegerValue | None = None,
    descending: bool = False,
    temp_storage: TempStorageLike | None = None,
) -> tuple[ThreadDataLike[_KeyT], ThreadDataLike[_ValueT]]:
    """Return stable sorted keys and associated numeric values without mutation.

    Parameters
    ----------
    group : ThreadGroup
        A complete physical block; every thread participates.
    keys, values : ThreadDataLike
        Fixed-size per-thread payloads with matching extents. Keys use int32,
        uint32, int64, or uint64. Values use the common API's numeric dtypes:
        signed or unsigned 8-, 16-, 32-, or 64-bit integers, float32, or
        float64.
    begin_bit, end_bit : int or compiler integer
        Block-uniform half-open interval in CUB's ordered key representation.
        Omitted end selects the key width. Require
        ``0 <= begin_bit < end_bit <= key_width``. Invalid static bounds fail
        compilation; invalid runtime bounds trap before conversion to CUB's
        integer arguments. Signed keys invert their sign bit before digit
        extraction. Returned keys keep their original representation.
    descending : bool
        Compile-time order selector. Equal digits retain their input order
        for both ascending and descending sorts.
    temp_storage : TempStorageLike, optional
        Explicit block scratch; omitted storage is allocated automatically.
        Its requested size and alignment must cover the specialization. The
        caller supplies reuse synchronization when ``auto_sync=False``.

    Returns
    -------
    tuple[ThreadDataLike, ThreadDataLike]
        Keys and associated values in blocked arrangement, preserving both
        input dtypes, their matching extent, and key/value association. Neither
        input payload is modified.

    Notes
    -----
    Wraps the key/value overload of CUB ``BlockRadixSort::Sort`` or
    ``SortDescending``. Both qualified backends also support floating-point
    keys, scalar payloads, and striped output. Numba-CUDA-MLIR additionally
    accepts local arrays; CUTLASS accepts CuTe register tensors.

    See Also
    --------
    cuda.coop.numba_mlir.radix_sort_pairs
        Numba-CUDA-MLIR payloads and qualified controls.
    cuda.coop.cutlass.radix_sort_pairs
        CuTe payloads and qualified controls.

    Examples
    --------
    Sort signed keys together with their original positions. Equal keys
    retain their input order, as checked by the stable host sort.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_radix_examples.py
        :language: python
        :start-after: # radix-sort-example-begin
        :end-before: # radix-sort-example-end
        :dedent: 4
    """
    _validate(
        "radix_sort_pairs",
        group,
        keys,
        values,
        begin_bit,
        end_bit,
        descending,
        temp_storage,
    )
    return _group_primitive_marker(
        "radix_sort_pairs",
        group,
        keys,
        values,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        temp_storage=temp_storage,
    )


@_common_group_operation("radix_rank_keys", group_kinds=("block",))
def radix_rank_keys(
    group: BlockGroup,
    keys: CommonThreadDataLike[_RankKeyT],
    /,
    *,
    begin_bit: int = 0,
    end_bit: int | None = None,
    radix_bits: int | None = None,
    descending: bool = False,
) -> ThreadDataLike[numpy.int32]:
    """Return stable int32 digit ranks without mutating integral keys.

    Parameters
    ----------
    group : ThreadGroup
        A complete physical block. All threads participate with identical
        compile-time controls and per-thread payload extents.
    keys : ThreadDataLike
        Fixed-size int32, uint32, int64, or uint64 per-thread keys in blocked
        arrangement.
    begin_bit, end_bit : int, optional
        Compile-time half-open interval in CUB's ordered representation.
        Begin defaults to zero; omitted end is begin plus ``radix_bits`` or
        four when that option is omitted. The interval must remain within
        the key width and contain one through eight bits.
    radix_bits : int, optional
        Compile-time digit width. When end is also supplied, it must equal
        ``end_bit - begin_bit``.
    descending : bool
        Compile-time selector that places greater digits before smaller ones.

    Returns
    -------
    ThreadDataLike
        Signed int32 ranks with the same per-thread extent as keys. Ranks
        form a permutation of the block tile's indices. Equal digits retain
        flattened blocked input order. The keys are not modified.

    Notes
    -----
    Uses CUB ``BlockRadixRank::RankKeys`` with a digit extractor. Signed keys
    invert their sign bit before digit extraction, matching radix sort's
    ordered representation. Scratch allocation and its reuse barrier are
    automatic. Both qualified backends also accept scalars and can write
    exclusive digit prefixes into a caller-provided output payload.
    Numba-CUDA-MLIR additionally accepts local arrays; CUTLASS accepts CuTe
    register tensors.

    See Also
    --------
    cuda.coop.numba_mlir.radix_rank_keys
        Numba-CUDA-MLIR payloads and qualified controls.
    cuda.coop.cutlass.radix_rank_keys
        CuTe payloads and qualified controls.

    Examples
    --------
    Find each key's position in a stable ordering by its low four bits.
    Rank returns positions without rearranging the input keys.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_radix_examples.py
        :language: python
        :start-after: # radix-rank-example-begin
        :end-before: # radix-rank-example-end
        :dedent: 4
    """
    _validate(
        "radix_rank_keys",
        group,
        keys,
        None,
        begin_bit,
        end_bit,
        descending,
        None,
        radix_bits,
    )
    return _group_primitive_marker(
        "radix_rank_keys",
        group,
        keys,
        begin_bit=begin_bit,
        end_bit=end_bit,
        radix_bits=radix_bits,
        descending=descending,
    )


__all__ = [
    "_radix_bounds",
    "radix_rank_keys",
    "radix_sort_keys",
    "radix_sort_pairs",
]
