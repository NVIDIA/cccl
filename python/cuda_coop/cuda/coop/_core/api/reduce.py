# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable cooperative reduction entry points."""

from __future__ import annotations

from typing import Any

from ..thread_group import CoopCompilerContextRequiredError, ThreadGroup
from ._dispatch import (
    _portable_group_operation,
)

_PORTABLE_REDUCTION_GROUP_KINDS = (
    "thread",
    "warp",
    "threads_within_warp",
    "block",
    "warps_within_block",
    "cluster",
)


@_portable_group_operation(
    "reduce",
    group_kinds=_PORTABLE_REDUCTION_GROUP_KINDS,
)
def reduce(
    group: ThreadGroup,
    value: object,
    /,
    *,
    binary_op: Any = None,
    broadcast: bool = True,
    valid_items: object = None,
    algorithm: str | None = None,
) -> Any:
    """Combine a group's values into one scalar.

    Parameters
    ----------
    group : cuda.coop.ThreadGroup
        Participating :ref:`thread group <coop-thread-groups>`. Supports a
        single thread, physical or logical warp, block, mapped group of warps,
        or cluster. Grid reductions are unsupported. Every member must call
        the collective, including members excluded by ``valid_items``.
    value : numeric scalar or cuda.coop.ThreadDataLike
        Each thread's contribution. A :ref:`per-thread payload
        <coop-thread-data>` contributes all its elements to the same scalar
        reduction; its dtype and fixed extent must agree across the group.
        Input values are preserved. Supported dtypes are signed and unsigned
        8-, 16-, 32-, and 64-bit integers, ``float32``, and ``float64``.
    binary_op : str, optional
        Compile-time operator: ``"sum"`` (the default), ``"multiplies"``,
        ``"min"``, ``"max"``, ``"bit_and"``, ``"bit_or"``, or ``"bit_xor"``.
        ``None`` selects sum. Bitwise operators require integer values.
        Operator aliases include ``"+"``, ``"*"``, ``"&"``, ``"|"``, and
        ``"^"``. Use the qualified ``cuda.coop.<backend>`` API for custom
        operators where supported.
    broadcast : bool, optional
        Compile-time flag, default ``True``. Return the result to every group
        member. With ``False``, only group rank zero has a defined result;
        other members must not use their return value.
    valid_items : int or integer scalar, optional
        Reduce only the first ``valid_items`` members by linear group rank.
        Requires scalar ``value``, ``broadcast=False``, and a block or physical
        or logical warp. The count must be uniform across the group and lie
        between one and the group size, inclusive. ``None`` includes all
        members. An empty reduction is unsupported.
    algorithm : str, optional
        Compile-time block algorithm: ``"raking_commutative_only"``,
        ``"raking"``, or ``"warp_reductions"``. An explicit choice requires
        a block and ``broadcast=False``. All portable operators support these
        choices. ``None`` lets the implementation select an algorithm.

    Returns
    -------
    numeric scalar
        Reduced value with the input dtype. ``ThreadData`` input also produces
        one scalar. Result visibility is controlled by ``broadcast``.

    Notes
    -----
    The reduction can regroup operations, so floating-point results can differ
    from a sequential fold. This call manages any required
    :ref:`temporary storage <coop-temp-storage>` automatically.

    See Also
    --------
    :cpp:struct:`cub::BlockReduce`, :cpp:struct:`cub::WarpReduce`
        C++ counterparts for the block algorithm and valid-prefix variants.
        Full-group built-in reductions use the CUDAX cooperative group API.

    Examples
    --------
    Find the maximum of a block and the minimum of its first 93 values.
    All 128 threads participate in both calls; only thread zero writes the
    partial reduction's result.

    .. literalinclude:: ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_reduce_examples.py
        :language: python
        :start-after: # reduce-example-begin
        :end-before: # reduce-example-end
        :dedent: 4
    """

    raise CoopCompilerContextRequiredError(
        "cuda.coop.reduce must be called from a supported GPU kernel."
    )


@_portable_group_operation(
    "sum",
    group_kinds=_PORTABLE_REDUCTION_GROUP_KINDS,
)
def sum(
    group: ThreadGroup,
    value: object,
    /,
    *,
    broadcast: bool = True,
    valid_items: object = None,
    algorithm: str | None = None,
) -> Any:
    """Add a group's values and return one scalar.

    This is equivalent to :func:`cuda.coop.reduce` with ``binary_op="sum"``.

    Parameters
    ----------
    group : cuda.coop.ThreadGroup
        Participating :ref:`thread group <coop-thread-groups>`. Supports a
        single thread, physical or logical warp, block, mapped group of warps,
        or cluster. Grid reductions are unsupported. Every member must call
        the collective.
    value : numeric scalar or cuda.coop.ThreadDataLike
        Each thread's contribution. A :ref:`per-thread payload
        <coop-thread-data>` contributes all its elements; its dtype and fixed
        extent must agree across the group. Input values are preserved.
        Supports signed and unsigned 8-, 16-, 32-, and 64-bit integers,
        ``float32``, and ``float64``.
    broadcast : bool, optional
        Compile-time flag, default ``True``. Return the sum to every member.
        With ``False``, only group rank zero has a defined result; all other
        members must still participate but must not use their return value.
    valid_items : int or integer scalar, optional
        Include only the first ``valid_items`` members by linear group rank.
        Requires scalar ``value``, ``broadcast=False``, and a block or physical
        or logical warp. The count must be uniform across the group and lie
        between one and the group size, inclusive. ``None`` includes all
        members. For a partial ``ThreadData`` tile, pad unused elements with
        zero before calling ``sum``.
    algorithm : str, optional
        Compile-time block algorithm: ``"raking_commutative_only"``,
        ``"raking"``, or ``"warp_reductions"``. An explicit choice requires
        a block and ``broadcast=False``. ``None`` lets the implementation
        select an algorithm.

    Returns
    -------
    numeric scalar
        Sum with the input dtype, including when the input is ``ThreadData``.
        The operation does not promote narrow integer types. Result visibility
        is controlled by ``broadcast``.

    Notes
    -----
    Floating-point addition can be regrouped, so the result can differ from a
    sequential sum. The implementation manages any required
    :ref:`temporary storage <coop-temp-storage>` automatically.

    See Also
    --------
    :cpp:struct:`cub::BlockReduce`, :cpp:struct:`cub::WarpReduce`
        C++ counterparts for the block algorithm and valid-prefix variants.
        Full-group built-in reductions use the CUDAX cooperative group API.

    Examples
    --------
    Sum an array in tiles of 256 elements using two values per thread. The last
    tile is padded with zero; each block writes one partial sum.

    .. literalinclude:: ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_reduce_examples.py
        :language: python
        :start-after: # sum-example-begin
        :end-before: # sum-example-end
        :dedent: 4
    """

    raise CoopCompilerContextRequiredError(
        "cuda.coop.sum must be called from a supported GPU kernel."
    )


__all__ = ["reduce", "sum"]
