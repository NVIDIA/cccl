# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing contract for portable scalar block and warp reduction."""

from typing import TypeVar, overload

from cuda.coop._typing import (
    PortableNumericScalar,
    ReduceAlgorithm,
    ReduceOperator,
    ValidItems,
)

from .thread_group import BlockGroup, WarpGroup

_ScalarT = TypeVar("_ScalarT", bound=PortableNumericScalar)

@overload
def reduce(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    binary_op: ReduceOperator | None = None,
    valid_items: ValidItems | None = None,
    algorithm: ReduceAlgorithm | None = None,
) -> _ScalarT:
    """Reduce one scalar per group member and return the root result.

    Every member of ``group`` must participate in converged control flow. The
    return value is defined only for group rank zero; other members must not
    consume it. ``valid_items`` selects a prefix of participating group ranks.

    Args:
        group: The current CUDA thread block.
        value: One numeric scalar owned by the calling thread.
        binary_op: Built-in reduction selector. The default is ``"sum"``.
        valid_items: Optional number of valid group ranks, starting at rank zero.
        algorithm: Optional deterministic CUB BlockReduce algorithm selector.

    Returns:
        The reduced scalar, defined only for group rank zero.

    Raises:
        TypeError: If ``group`` or a static ``valid_items`` is invalid.
        ValueError: If a selector or static ``valid_items`` is invalid.
        CoopCompilerContextRequiredError: If no compatible backend is active.

    Example:
        >>> total = coop.reduce(block, value, binary_op="sum")
    """

@overload
def reduce(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    binary_op: ReduceOperator | None = None,
    valid_items: ValidItems | None = None,
    algorithm: None = None,
) -> _ScalarT:
    """Reduce one scalar per group member and return the root result.

    Every lane in the physical warp must participate in converged control flow.
    The return value is defined only for lane zero. ``valid_items`` selects the
    lane prefix ``[0, valid_items)`` and must be uniform and between 1 and 32.

    An explicit BlockReduce ``algorithm`` is not accepted for warp groups.

    Raises:
        TypeError: If ``group`` or a static ``valid_items`` is invalid.
        ValueError: If a selector or static ``valid_items`` is invalid.
        CoopCompilerContextRequiredError: If no compatible backend is active.

    Example:
        >>> total = coop.reduce(warp, value, binary_op="max")
    """

@overload
def sum(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    valid_items: ValidItems | None = None,
    algorithm: ReduceAlgorithm | None = None,
) -> _ScalarT:
    """Sum one scalar per group member and return the root result.

    Every member of ``group`` must participate in converged control flow. The
    return value is defined only for group rank zero; other members must not
    consume it. ``valid_items`` selects a prefix of participating group ranks.

    Args:
        group: The current CUDA thread block.
        value: One numeric scalar owned by the calling thread.
        valid_items: Optional number of valid group ranks, starting at rank zero.
        algorithm: Optional deterministic CUB BlockReduce algorithm selector.

    Returns:
        The sum, defined only for group rank zero.

    Raises:
        TypeError: If ``group`` or a static ``valid_items`` is invalid.
        ValueError: If an algorithm or static ``valid_items`` is invalid.
        CoopCompilerContextRequiredError: If no compatible backend is active.

    Example:
        >>> total = coop.sum(block, value)
    """

@overload
def sum(
    group: WarpGroup,
    value: _ScalarT,
    /,
    *,
    valid_items: ValidItems | None = None,
    algorithm: None = None,
) -> _ScalarT:
    """Sum one scalar per group member and return the root result.

    Every lane in the physical warp must participate in converged control flow.
    The return value is defined only for lane zero. ``valid_items`` selects the
    lane prefix ``[0, valid_items)`` and must be uniform and between 1 and 32.

    An explicit BlockReduce ``algorithm`` is not accepted for warp groups.

    Raises:
        TypeError: If ``group`` or a static ``valid_items`` is invalid.
        ValueError: If an algorithm or static ``valid_items`` is invalid.
        CoopCompilerContextRequiredError: If no compatible backend is active.

    Example:
        >>> total = coop.sum(warp, value)
    """

__all__ = ["reduce", "sum"]
