# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Define CUB block and warp reductions with leader-owned results."""

from __future__ import annotations

from typing import TypeVar

from cuda.coop._typing import (
    CommonNumericScalar,
    CommonThreadDataLike,
    ReduceAlgorithm,
    ReduceOperator,
    TempStorageLike,
    ValidItems,
)

from ..thread_group import CoopCompilerContextRequiredError
from ._dispatch import _common_group_operation
from .thread_group import BlockGroup, WarpGroup

_ItemT = TypeVar("_ItemT", bound=CommonNumericScalar)
_COMMON_REDUCTION_GROUP_KINDS = ("warp", "threads_within_warp", "block")


@_common_group_operation("reduce", group_kinds=_COMMON_REDUCTION_GROUP_KINDS)
def reduce(
    group: BlockGroup | WarpGroup,
    value: CommonThreadDataLike[_ItemT] | _ItemT,
    /,
    *,
    binary_op: ReduceOperator | None = None,
    valid_items: ValidItems | None = None,
    algorithm: ReduceAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ItemT:
    """Reduce a block or warp to a scalar defined at group rank zero.

    Parameters
    ----------
    group : cuda.coop.ThreadGroup
        A block, physical warp, or logical warp with a power-of-two width
        from 1 through 32 or a width from 17 through 31. CUB supports only one
        non-power-of-two group per physical warp. For those widths, use
        ``group_by(width, exhaustive=False)`` and guard the call with
        ``group.is_member()``. Every group member must participate, including
        members excluded by ``valid_items``. Warp groups require complete
        physical warps in the enclosing block.
    value : numeric scalar or cuda.coop.ThreadDataLike
        Each thread's contribution. Reductions accept a scalar or a
        :ref:`per-thread payload <coop-thread-data>` whose elements all
        contribute to the result.
        Input values are preserved. Supported dtypes are signed and unsigned
        8-, 16-, 32-, and 64-bit integers, ``float32``, and ``float64``.
    binary_op : str, optional
        Compile-time operator: ``"sum"`` (the default), ``"multiplies"``,
        ``"min"``, ``"max"``, ``"bit_and"``, ``"bit_or"``, or ``"bit_xor"``.
        ``None`` selects sum. Bitwise operators require integer values.
        Operator aliases include ``"+"``, ``"*"``, ``"&"``, ``"|"``, and
        ``"^"``. Qualified backend APIs also support custom device operators
        where documented.
    valid_items : int or integer scalar, optional
        Reduce the first ``valid_items`` members by linear group rank.
        Requires scalar ``value``. The count must be uniform across the group
        and between one and the group size, inclusive. ``None`` includes all
        members. Empty reductions are unsupported.
    algorithm : str, optional
        Compile-time block algorithm: ``"raking_commutative_only"``,
        ``"raking"``, or ``"warp_reductions"``. ``None`` selects
        ``"warp_reductions"`` for blocks. Warp reductions require ``None``.
    temp_storage : cuda.coop.TempStorageLike, optional
        Explicit block scratch descriptor. Size and alignment are inferred
        from all uses unless requested explicitly. Descriptor options govern
        scratch sharing and reuse synchronization. When omitted, the compiler
        allocates scratch and inserts reuse barriers. Warp reductions require
        omission so the compiler can allocate a separate slice per warp.

    Returns
    -------
    numeric scalar
        Reduced value with the input dtype, defined only at group rank zero.
        Other members must not use their return value.

    Notes
    -----
    Floating-point results can differ from a sequential fold because reduction
    regroups operations. See :ref:`temporary storage <coop-temp-storage>` for
    scratch lifetime and synchronization.

    See Also
    --------
    :cpp:struct:`cub::BlockReduce`, :cpp:struct:`cub::WarpReduce`
        Native primitives and their result-ownership contracts.

    Examples
    --------
    .. literalinclude::
        ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_reduce_examples.py
        :language: python
        :start-after: # reduce-example-begin
        :end-before: # reduce-example-end
        :dedent: 4
    """
    raise CoopCompilerContextRequiredError(
        "cuda.coop.reduce must be called from a supported GPU kernel."
    )


@_common_group_operation("sum", group_kinds=_COMMON_REDUCTION_GROUP_KINDS)
def sum(
    group: BlockGroup | WarpGroup,
    value: CommonThreadDataLike[_ItemT] | _ItemT,
    /,
    *,
    valid_items: ValidItems | None = None,
    algorithm: ReduceAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ItemT:
    """Sum a block or warp to a scalar defined at group rank zero.

    Equivalent to :func:`cuda.coop.reduce` with ``binary_op="sum"``. Its group,
    operand, valid-prefix, algorithm, and temporary-storage contracts apply.
    Every group member participates; only rank zero may use the result.

    Parameters
    ----------
    group : cuda.coop.ThreadGroup
        Block, physical warp, or supported logical warp.
        See :func:`cuda.coop.reduce` for participation requirements.
    value : numeric scalar or cuda.coop.ThreadDataLike
        Each thread supplies a scalar or a fixed per-thread payload. Every
        payload item contributes to the result.
    valid_items : int or integer scalar, optional
        Uniform count of contributing group members, for scalar inputs only.
    algorithm : str, optional
        Block algorithm; see :func:`cuda.coop.reduce`. Omit for warp groups.
    temp_storage : cuda.coop.TempStorageLike, optional
        Explicit block scratch descriptor. Omission enables compiler-managed
        scratch and reuse synchronization. Omit for warp groups.

    Returns
    -------
    numeric scalar
        Sum with the input dtype, defined only at group rank zero.

    See Also
    --------
    cuda.coop.reduce
        Reduction contracts and available algorithms.
    """
    raise CoopCompilerContextRequiredError(
        "cuda.coop.sum must be called from a supported GPU kernel."
    )


__all__ = ["reduce", "sum"]
