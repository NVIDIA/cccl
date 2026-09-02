# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable cooperative reduction entry points.

The root functions validate the conservative group-reduction profile before
delegating to the active backend. Semantic planning and provider selection live
in the portable group family and backend lowering layers.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any, TypeVar

from ..block.reduce import (
    normalize_block_reduce_algorithm,
    normalize_block_reduce_operator,
)
from ..thread_group import ThreadGroup
from ._dispatch import _group_primitive_marker

_ScalarT = TypeVar("_ScalarT")


def _validate_group(group: ThreadGroup, *, operation: str) -> None:
    if not isinstance(group, ThreadGroup):
        raise TypeError(f"cuda.coop.{operation} group must be a ThreadGroup")
    if group.kind not in {"block", "warp"}:
        raise NotImplementedError(
            f"cuda.coop.{operation} currently supports block and warp groups only"
        )


def _normalize_valid_items(
    operation: str,
    valid_items: Any,
    *,
    group_size: int | None = None,
) -> Any:
    if valid_items is None:
        return None
    if isinstance(valid_items, bool):
        raise TypeError(f"cuda.coop.{operation} valid_items must be an integer")
    if isinstance(valid_items, Integral):
        normalized = int(valid_items)
        if normalized < 1:
            raise ValueError(f"cuda.coop.{operation} valid_items must be at least 1")
        if group_size is not None and normalized > group_size:
            raise ValueError(
                f"cuda.coop.{operation} valid_items must be at most {group_size}"
            )
        return normalized
    try:
        width = valid_items.width
        signed = valid_items.signed
        dtype = valid_items.dtype
        ir_value = valid_items.ir_value
    except AttributeError:
        pass
    else:
        if (
            isinstance(width, int)
            and not isinstance(width, bool)
            and width > 0
            and isinstance(signed, bool)
            and dtype is not None
            and callable(ir_value)
        ):
            return valid_items
    raise TypeError(f"cuda.coop.{operation} valid_items must be an integer")


def reduce(
    group: ThreadGroup,
    value: _ScalarT,
    /,
    *,
    binary_op: Any = None,
    valid_items: Any = None,
    algorithm: Any = None,
) -> _ScalarT:
    """Reduce one scalar per group member and return the root result.

    Every member of ``group`` must participate in converged control flow. The
    return value is defined only for group rank zero; other members must not
    consume it. ``valid_items`` selects a prefix of participating group ranks.

    Args:
        group: The current CUDA thread block or physical warp.
        value: One numeric scalar owned by the calling thread.
        binary_op: Built-in reduction selector. The default is ``"sum"``.
        valid_items: Optional number of valid group ranks, starting at rank zero.
        algorithm: Optional deterministic CUB BlockReduce algorithm selector;
            accepted for block groups only.

    Returns:
        The reduced scalar, defined only for group rank zero.

    Raises:
        TypeError: If ``group`` or a static ``valid_items`` is invalid.
        ValueError: If a selector or static ``valid_items`` is invalid.
        CoopCompilerContextRequiredError: If no compatible backend is active.

    Example:
        >>> total = coop.reduce(block, value, binary_op="sum")
    """

    _validate_group(group, operation="reduce")
    operator = normalize_block_reduce_operator(binary_op)
    if group.kind == "warp" and algorithm is not None:
        raise ValueError(
            "cuda.coop.reduce algorithm selection applies to block groups, "
            "not physical warps"
        )
    selected_algorithm = (
        normalize_block_reduce_algorithm(algorithm).value
        if group.kind == "block"
        else None
    )
    valid_items = _normalize_valid_items(
        "reduce", valid_items, group_size=group.static_size
    )
    return _group_primitive_marker(
        "reduce",
        group,
        value,
        binary_op=operator.value,
        valid_items=valid_items,
        algorithm=selected_algorithm,
    )


def sum(
    group: ThreadGroup,
    value: _ScalarT,
    /,
    *,
    valid_items: Any = None,
    algorithm: Any = None,
) -> _ScalarT:
    """Sum one scalar per group member and return the root result.

    Every member of ``group`` must participate in converged control flow. The
    return value is defined only for group rank zero; other members must not
    consume it. ``valid_items`` selects a prefix of participating group ranks.

    Args:
        group: The current CUDA thread block or physical warp.
        value: One numeric scalar owned by the calling thread.
        valid_items: Optional number of valid group ranks, starting at rank zero.
        algorithm: Optional deterministic CUB BlockReduce algorithm selector;
            accepted for block groups only.

    Returns:
        The sum, defined only for group rank zero.

    Raises:
        TypeError: If ``group`` or a static ``valid_items`` is invalid.
        ValueError: If an algorithm or static ``valid_items`` is invalid.
        CoopCompilerContextRequiredError: If no compatible backend is active.

    Example:
        >>> total = coop.sum(block, value)
    """

    _validate_group(group, operation="sum")
    if group.kind == "warp" and algorithm is not None:
        raise ValueError(
            "cuda.coop.sum algorithm selection applies to block groups, "
            "not physical warps"
        )
    selected_algorithm = (
        normalize_block_reduce_algorithm(algorithm).value
        if group.kind == "block"
        else None
    )
    valid_items = _normalize_valid_items(
        "sum", valid_items, group_size=group.static_size
    )
    return _group_primitive_marker(
        "sum",
        group,
        value,
        valid_items=valid_items,
        algorithm=selected_algorithm,
    )


__all__ = ["reduce", "sum"]
