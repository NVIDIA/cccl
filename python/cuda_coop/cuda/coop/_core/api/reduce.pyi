# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing contract for the portable scalar BlockReduce family."""

from typing import TypeVar

from cuda.coop._typing import (
    PortableNumericScalar,
    ReduceAlgorithm,
    ReduceOperator,
    ValidItems,
)

from .thread_group import ThreadGroup

_ScalarT = TypeVar("_ScalarT", bound=PortableNumericScalar)

def reduce(
    group: ThreadGroup,
    value: _ScalarT,
    /,
    *,
    binary_op: ReduceOperator | None = None,
    valid_items: ValidItems | None = None,
    algorithm: ReduceAlgorithm | None = None,
) -> _ScalarT:
    """Reduce one scalar per block thread and return the root result.

    Every thread in ``group`` must participate in converged control flow. The
    return value is defined only for block rank zero; other threads must not
    consume it. ``valid_items`` selects a prefix of participating block ranks.

    Args:
        group: The current CUDA thread block.
        value: One numeric scalar owned by the calling thread.
        binary_op: Built-in reduction selector. The default is ``"sum"``.
        valid_items: Optional number of valid block ranks, starting at rank zero.
        algorithm: Optional deterministic CUB BlockReduce algorithm selector.

    Returns:
        The reduced scalar, defined only for block rank zero.

    Raises:
        TypeError: If ``group`` or a static ``valid_items`` is invalid.
        ValueError: If a selector or static ``valid_items`` is invalid.
        CoopCompilerContextRequiredError: If no compatible backend is active.

    Example:
        >>> total = coop.reduce(block, value, binary_op="sum")
    """

def sum(
    group: ThreadGroup,
    value: _ScalarT,
    /,
    *,
    valid_items: ValidItems | None = None,
    algorithm: ReduceAlgorithm | None = None,
) -> _ScalarT:
    """Sum one scalar per block thread and return the root result.

    Every thread in ``group`` must participate in converged control flow. The
    return value is defined only for block rank zero; other threads must not
    consume it. ``valid_items`` selects a prefix of participating block ranks.

    Args:
        group: The current CUDA thread block.
        value: One numeric scalar owned by the calling thread.
        valid_items: Optional number of valid block ranks, starting at rank zero.
        algorithm: Optional deterministic CUB BlockReduce algorithm selector.

    Returns:
        The sum, defined only for block rank zero.

    Raises:
        TypeError: If ``group`` or a static ``valid_items`` is invalid.
        ValueError: If an algorithm or static ``valid_items`` is invalid.
        CoopCompilerContextRequiredError: If no compatible backend is active.

    Example:
        >>> total = coop.sum(block, value)
    """

__all__ = ["reduce", "sum"]
