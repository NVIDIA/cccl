# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from typing import Literal, TypeVar

from ._typing import (
    PortableNumericScalar,
    ReduceAlgorithm,
    ReduceOperator,
    ValidItems,
)

_ScalarT = TypeVar("_ScalarT", bound=PortableNumericScalar)

class _ThreadGroupConstructionToken: ...

class ThreadGroup:
    """Descriptor for the current CUDA thread block.

    The descriptor is compiler-free. A backend resolves its exact dimensions
    from verified launch facts while tracing a cooperative reduction.

    Raises:
        TypeError: If user code attempts to construct the opaque descriptor
            directly instead of calling ``this_block``.

    Example:
        >>> from cuda import coop
        >>> block = coop.this_block()
        >>> block.kind
        'block'
    """

    def __init__(self, _token: _ThreadGroupConstructionToken, /) -> None: ...
    @property
    def kind(self) -> Literal["block"]: ...

def this_block() -> ThreadGroup:
    """Return a descriptor for the current CUDA thread block.

    The descriptor carries no user-supplied dimensions. A compiler integration
    resolves exact block dimensions from verified launch facts when lowering a
    reduction.

    Returns:
        A compiler-free block descriptor accepted by ``reduce`` and ``sum``.

    Raises:
        RuntimeError: If a compiler later cannot resolve exact block dimensions.

    Example:
        >>> from cuda import coop
        >>> block = coop.this_block()
        >>> block.kind
        'block'
    """

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

__version__: str
