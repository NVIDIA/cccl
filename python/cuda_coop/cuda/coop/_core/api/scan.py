# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from typing import Any

from ..thread_group import CoopCompilerContextRequiredError, ThreadGroup
from ._dispatch import (
    _portable_group_operation,
)
from ._payload import (
    TempStorageLike,
)

_PORTABLE_SCAN_GROUP_KINDS = ("block", "warp", "threads_within_warp")


@_portable_group_operation("scan", group_kinds=_PORTABLE_SCAN_GROUP_KINDS)
def scan(
    group: ThreadGroup,
    value: object,
    /,
    *,
    mode: str = "exclusive",
    scan_op: Any = None,
    initial_value: Any = None,
    algorithm: str | None = None,
    temp_storage: TempStorageLike | None = None,
) -> Any:
    """Compute a prefix for every input value in a block or warp group.

    Parameters
    ----------
    group : cuda.coop.ThreadGroup
        Participating threads; see :ref:`thread groups <coop-thread-groups>`.
        Supports blocks and physical or logical warps. Warp scans require
        an enclosing block size divisible by 32. All group members must
        execute the primitive together.
    value : numeric scalar or cuda.coop.ThreadDataLike
        Each thread's input. Blocks accept a scalar or a readable
        :ref:`per-thread payload <coop-thread-data>`; warps accept one scalar
        per lane. Payloads have the same item count and dtype in each thread.
        A block scans payloads in blocked order: all items from thread zero,
        then all items from thread one, and so on. Scalar inputs follow
        linear group rank. The input is preserved.
    mode : {"exclusive", "inclusive"}, optional
        Compile-time choice, default ``"exclusive"``. An exclusive prefix
        combines ``initial_value`` with the inputs preceding the current
        item. An inclusive prefix also includes the current item and has
        no initial value.
    scan_op : str, optional
        Compile-time operator: ``"sum"``, ``"multiplies"``, ``"min"``,
        ``"max"``, ``"bit_and"``, ``"bit_or"``, or ``"bit_xor"``.
        ``None`` selects sum. Bitwise operators require integer values.
        Use a backend-qualified API for custom operators.
    initial_value : numeric scalar, optional
        First output of an exclusive scan, combined with every subsequent
        prefix. Defaults to zero for sum; required for other exclusive
        operators. Must be ``None`` for inclusive scans. The value must be
        uniform across the group. Typed scalars must have exactly the input
        dtype. Python literals must be finite and in range; an integer
        input requires an integer literal.
    algorithm : str, optional
        Compile-time block algorithm. ``None`` selects ``"raking"``, which
        scans shared partial reductions. ``"raking_memoize"`` keeps those
        partials in registers to reduce shared-memory reads.
        ``"warp_scans"`` combines warp scans and requires a block size
        divisible by 32. Warp groups require ``None``.
    temp_storage : cuda.coop.TempStorageLike, optional
        :ref:`Scratch descriptor <coop-temp-storage>` for a block scan.
        ``None`` uses automatic scratch and reuse synchronization. Warp
        scans require ``None`` and use automatic storage for each group.

    Returns
    -------
    numeric scalar or cuda.coop.ThreadDataLike
        This thread's prefixes, with the input dtype and item count. A scalar
        input produces a scalar. A payload input produces a new payload;
        every output item has a defined value on every participating thread.

    Notes
    -----
    Prefixes restart for each group; this call does not scan across blocks.
    Scan relies on associative operators and may regroup arithmetic.
    Floating-point prefixes can differ from sequential CPU results.
    Integer scans accumulate in the input dtype and can overflow.

    See Also
    --------
    :cpp:struct:`cub::BlockScan`, :cpp:struct:`cub::WarpScan`
        C++ primitive types providing ``ExclusiveScan``, ``InclusiveScan``,
        and their sum overloads.

    Examples
    --------
    Compute inclusive bitwise XOR prefixes across a block with
    Numba-CUDA-MLIR. The qualified import activates the backend, and the
    calls use the common ``cuda.coop`` API.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_scan_examples.py
        :language: python
        :start-after: # scan-example-begin
        :end-before: # scan-example-end
        :dedent: 4
    """

    raise CoopCompilerContextRequiredError(
        "cuda.coop.scan must be called from a supported GPU kernel."
    )


@_portable_group_operation(
    "exclusive_sum",
    group_kinds=_PORTABLE_SCAN_GROUP_KINDS,
)
def exclusive_sum(
    group: ThreadGroup,
    value: object,
    /,
    *,
    algorithm: str | None = None,
    temp_storage: TempStorageLike | None = None,
) -> Any:
    """Sum the inputs preceding each item, starting with zero.

    Parameters
    ----------
    group : cuda.coop.ThreadGroup
        Block or physical/logical warp whose members execute the primitive
        together; see :ref:`thread groups <coop-thread-groups>`. Warp scans
        require an enclosing block size divisible by 32.
    value : numeric scalar or cuda.coop.ThreadDataLike
        Each thread's input. Blocks accept a scalar or a readable
        :ref:`per-thread payload <coop-thread-data>`; warps accept one scalar
        per lane. All threads use the same dtype and item count. Payload
        items follow blocked order, with all items from each thread placed
        consecutively in linear group-rank order. The input is preserved.
    algorithm : str, optional
        Compile-time block algorithm: ``"raking"``, ``"raking_memoize"``,
        or ``"warp_scans"``; ``None`` selects ``"raking"``. The
        ``"warp_scans"`` algorithm requires a block size divisible by 32.
        Warp groups require ``None``. See :func:`cuda.coop.scan` for the
        algorithm choices.
    temp_storage : cuda.coop.TempStorageLike, optional
        :ref:`Scratch descriptor <coop-temp-storage>` for blocks. ``None``
        uses automatic scratch and reuse synchronization. Warp groups
        require ``None``.

    Returns
    -------
    numeric scalar or cuda.coop.ThreadDataLike
        This thread's exclusive prefixes, in the input dtype. The first
        item in each group is zero. A scalar input produces a scalar;
        a payload input produces a new payload with the same item count.

    Notes
    -----
    Prefixes restart for each group. Integer overflow and floating-point
    regrouping follow the rules in :func:`cuda.coop.scan`. Use
    :func:`cuda.coop.exclusive_scan` to supply a nonzero initial value or
    another operator.

    See Also
    --------
    :cpp:struct:`cub::BlockScan`, :cpp:struct:`cub::WarpScan`
        C++ primitive types providing ``ExclusiveSum``.

    Examples
    --------
    Turn per-thread item counts into offsets within one block using
    Numba-CUDA-MLIR. Each offset is the sum of earlier threads' counts.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_scan_examples.py
        :language: python
        :start-after: # exclusive-sum-example-begin
        :end-before: # exclusive-sum-example-end
        :dedent: 4
    """

    raise CoopCompilerContextRequiredError(
        "cuda.coop.exclusive_sum must be called from a supported GPU kernel."
    )


@_portable_group_operation(
    "inclusive_sum",
    group_kinds=_PORTABLE_SCAN_GROUP_KINDS,
)
def inclusive_sum(
    group: ThreadGroup,
    value: object,
    /,
    *,
    algorithm: str | None = None,
    temp_storage: TempStorageLike | None = None,
) -> Any:
    """Sum the inputs up to and including each item.

    Parameters
    ----------
    group : cuda.coop.ThreadGroup
        Block or physical/logical warp whose members execute the primitive
        together; see :ref:`thread groups <coop-thread-groups>`. Warp scans
        require an enclosing block size divisible by 32.
    value : numeric scalar or cuda.coop.ThreadDataLike
        Each thread's input. Blocks accept a scalar or a readable
        :ref:`per-thread payload <coop-thread-data>`; warps accept one scalar
        per lane. All threads use the same dtype and item count. Payload
        items follow blocked order, with all items from each thread placed
        consecutively in linear group-rank order. The input is preserved.
    algorithm : str, optional
        Compile-time block algorithm: ``"raking"``, ``"raking_memoize"``,
        or ``"warp_scans"``; ``None`` selects ``"raking"``. The
        ``"warp_scans"`` algorithm requires a block size divisible by 32.
        Warp groups require ``None``. See :func:`cuda.coop.scan` for the
        algorithm choices.
    temp_storage : cuda.coop.TempStorageLike, optional
        :ref:`Scratch descriptor <coop-temp-storage>` for blocks. ``None``
        uses automatic scratch and reuse synchronization. Warp groups
        require ``None``.

    Returns
    -------
    numeric scalar or cuda.coop.ThreadDataLike
        This thread's inclusive prefixes, in the input dtype. The first
        item in each group equals its input. A scalar input produces a
        scalar; a payload input produces a new payload with the same
        item count.

    Notes
    -----
    Prefixes restart for each group. Integer overflow and floating-point
    regrouping follow the rules in :func:`cuda.coop.scan`. Use
    :func:`cuda.coop.inclusive_scan` for another operator.

    See Also
    --------
    :cpp:struct:`cub::BlockScan`, :cpp:struct:`cub::WarpScan`
        C++ primitive types providing ``InclusiveSum``.

    Examples
    --------
    Scan two values per thread with Numba-CUDA-MLIR. The Load, Scan, and
    Store use the same blocked order, so the output is the prefix sum of
    the source array. The original per-thread values remain available.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_scan_examples.py
        :language: python
        :start-after: # inclusive-sum-example-begin
        :end-before: # inclusive-sum-example-end
        :dedent: 4
    """

    raise CoopCompilerContextRequiredError(
        "cuda.coop.inclusive_sum must be called from a supported GPU kernel."
    )


@_portable_group_operation(
    "exclusive_scan",
    group_kinds=_PORTABLE_SCAN_GROUP_KINDS,
)
def exclusive_scan(
    group: ThreadGroup,
    value: object,
    /,
    *,
    scan_op: Any = None,
    initial_value: Any = None,
    algorithm: str | None = None,
    temp_storage: TempStorageLike | None = None,
) -> Any:
    """Combine an initial value with the inputs preceding each item.

    Parameters
    ----------
    group : cuda.coop.ThreadGroup
        Block or physical/logical warp whose members execute the primitive
        together; see :ref:`thread groups <coop-thread-groups>`. Warp scans
        require an enclosing block size divisible by 32.
    value : numeric scalar or cuda.coop.ThreadDataLike
        Each thread's input. Blocks accept a scalar or a readable
        :ref:`per-thread payload <coop-thread-data>`; warps accept one scalar
        per lane. All threads use the same dtype and item count. Payload
        items follow blocked order, with all items from each thread placed
        consecutively in linear group-rank order. The input is preserved.
    scan_op : str, optional
        Compile-time operator: ``"sum"``, ``"multiplies"``, ``"min"``,
        ``"max"``, ``"bit_and"``, ``"bit_or"``, or ``"bit_xor"``.
        ``None`` selects sum. Bitwise operators require integer values.
        Use a backend-qualified API for custom operators.
    initial_value : numeric scalar, optional
        First output of each group, combined with every subsequent prefix.
        Defaults to zero for sum; required for all other operators. Must
        have the same value across the group. Typed scalars must have
        exactly the input dtype. Python literals must be finite and in
        range; an integer input requires an integer literal.
    algorithm : str, optional
        Compile-time block algorithm: ``"raking"``, ``"raking_memoize"``,
        or ``"warp_scans"``; ``None`` selects ``"raking"``. The
        ``"warp_scans"`` algorithm requires a block size divisible by 32.
        Warp groups require ``None``. See :func:`cuda.coop.scan` for the
        algorithm choices.
    temp_storage : cuda.coop.TempStorageLike, optional
        :ref:`Scratch descriptor <coop-temp-storage>` for blocks. ``None``
        uses automatic scratch and reuse synchronization. Warp groups
        require ``None``.

    Returns
    -------
    numeric scalar or cuda.coop.ThreadDataLike
        This thread's exclusive prefixes, in the input dtype. A scalar
        input produces a scalar; a payload input produces a new payload
        with the same item count. Every item has a defined output,
        including the first item, whose output is ``initial_value``.

    Notes
    -----
    Equivalent to :func:`cuda.coop.scan` with ``mode="exclusive"``.
    Prefixes restart for each group. See that function for arithmetic
    assumptions and algorithm descriptions.

    See Also
    --------
    :cpp:struct:`cub::BlockScan`, :cpp:struct:`cub::WarpScan`
        C++ primitive types providing ``ExclusiveScan``.

    Examples
    --------
    Compute exclusive minimum prefixes with Numba-CUDA-MLIR. The runtime
    initial value has the same dtype as the input and seeds the first
    output. Each later output is the minimum of that seed and earlier
    input values.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_scan_examples.py
        :language: python
        :start-after: # exclusive-scan-example-begin
        :end-before: # exclusive-scan-example-end
        :dedent: 4
    """

    raise CoopCompilerContextRequiredError(
        "cuda.coop.exclusive_scan must be called from a supported GPU kernel."
    )


@_portable_group_operation(
    "inclusive_scan",
    group_kinds=_PORTABLE_SCAN_GROUP_KINDS,
)
def inclusive_scan(
    group: ThreadGroup,
    value: object,
    /,
    *,
    scan_op: Any = None,
    algorithm: str | None = None,
    temp_storage: TempStorageLike | None = None,
) -> Any:
    """Combine the inputs up to and including each item.

    Parameters
    ----------
    group : cuda.coop.ThreadGroup
        Block or physical/logical warp whose members execute the primitive
        together; see :ref:`thread groups <coop-thread-groups>`. Warp scans
        require an enclosing block size divisible by 32.
    value : numeric scalar or cuda.coop.ThreadDataLike
        Each thread's input. Blocks accept a scalar or a readable
        :ref:`per-thread payload <coop-thread-data>`; warps accept one scalar
        per lane. All threads use the same dtype and item count. Payload
        items follow blocked order, with all items from each thread placed
        consecutively in linear group-rank order. The input is preserved.
    scan_op : str, optional
        Compile-time operator: ``"sum"``, ``"multiplies"``, ``"min"``,
        ``"max"``, ``"bit_and"``, ``"bit_or"``, or ``"bit_xor"``.
        ``None`` selects sum. Bitwise operators require integer values.
        Use a backend-qualified API for custom operators.
    algorithm : str, optional
        Compile-time block algorithm: ``"raking"``, ``"raking_memoize"``,
        or ``"warp_scans"``; ``None`` selects ``"raking"``. The
        ``"warp_scans"`` algorithm requires a block size divisible by 32.
        Warp groups require ``None``. See :func:`cuda.coop.scan` for the
        algorithm choices.
    temp_storage : cuda.coop.TempStorageLike, optional
        :ref:`Scratch descriptor <coop-temp-storage>` for blocks. ``None``
        uses automatic scratch and reuse synchronization. Warp groups
        require ``None``.

    Returns
    -------
    numeric scalar or cuda.coop.ThreadDataLike
        This thread's inclusive prefixes, in the input dtype. A scalar
        input produces a scalar; a payload input produces a new payload
        with the same item count. The first item in each group equals
        its input.

    Notes
    -----
    Equivalent to :func:`cuda.coop.scan` with ``mode="inclusive"``; there
    is no initial value. Prefixes restart for each group. See that
    function for arithmetic assumptions and algorithm descriptions.

    See Also
    --------
    :cpp:struct:`cub::BlockScan`, :cpp:struct:`cub::WarpScan`
        C++ primitive types providing ``InclusiveScan``.

    Examples
    --------
    Compute running maxima independently in each group of eight lanes
    with Numba-CUDA-MLIR. The 64-thread block contains eight logical warps.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_scan_examples.py
        :language: python
        :start-after: # inclusive-scan-example-begin
        :end-before: # inclusive-scan-example-end
        :dedent: 4
    """

    raise CoopCompilerContextRequiredError(
        "cuda.coop.inclusive_scan must be called from a supported GPU kernel."
    )


__all__ = [
    "exclusive_scan",
    "exclusive_sum",
    "inclusive_scan",
    "inclusive_sum",
    "scan",
]
