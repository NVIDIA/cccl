# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group-first Reduce and Sum markers for Numba-CUDA-MLIR."""

from __future__ import annotations

from typing import Any

from ._compiler._operations import group_operation
from ._group_marker import group_primitive_marker
from ._thread_group import ThreadGroup


@group_operation(
    "reduce",
    family_module="cuda.coop.numba_mlir._compiler._group_reduce",
)
def reduce(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    binary_op: Any = None,
    broadcast: bool = True,
    valid_items: Any = None,
    algorithm: Any = None,
) -> Any:
    """Reduce values with Numba-CUDA-MLIR, including custom operators.

    See :func:`cuda.coop.reduce` for the shared parameters, defaults, supported
    groups, result visibility, and examples. The qualified API adds these
    operand forms:

    Parameters
    ----------
    value : numeric scalar, cuda.coop.ThreadDataLike, or local array
        Also accepts a one-dimensional ``cuda.local.array`` with a fixed
        extent. Its elements contribute to one scalar result, just like
        :ref:`ThreadData <coop-thread-data>`. The input is preserved.
    binary_op : str or callable, optional
        In addition to the built-in strings, accepts a device-compilable
        binary callable returning the input dtype. Custom operators require
        ``broadcast=False`` and a complete block or physical or logical warp.
        Warp custom reductions accept scalar inputs only. For a block, use
        ``algorithm=None``, ``"raking"``, or ``"warp_reductions"``;
        ``"raking_commutative_only"`` requires proven commutativity and is
        unavailable for Python callables. ``None`` selects sum.

    Returns
    -------
    numeric scalar
        Reduced value with the input dtype, defined at every member when
        ``broadcast=True`` and only at group rank zero otherwise.

    See Also
    --------
    cuda.coop.reduce
        Shared reduction contract and executable examples.
    :cpp:struct:`cub::BlockReduce`, :cpp:struct:`cub::WarpReduce`
        C++ primitives used for custom operators, valid prefixes, and explicit
        block algorithms. Full-group built-in reductions use CUDAX.
    """

    return group_primitive_marker(
        "reduce",
        group,
        value,
        binary_op=binary_op,
        broadcast=broadcast,
        valid_items=valid_items,
        algorithm=algorithm,
    )


@group_operation(
    "sum",
    family_module="cuda.coop.numba_mlir._compiler._group_reduce",
)
def sum(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    broadcast: bool = True,
    valid_items: Any = None,
    algorithm: Any = None,
) -> Any:
    """Sum values with Numba-CUDA-MLIR.

    See :func:`cuda.coop.sum` for all parameters, defaults, supported groups,
    and examples. The qualified API also accepts a one-dimensional
    ``cuda.local.array`` with a fixed extent as ``value``. Its elements
    contribute to one scalar result, and the input is preserved.

    Returns
    -------
    numeric scalar
        Sum with the input dtype, defined at every member when
        ``broadcast=True`` and only at group rank zero otherwise.

    See Also
    --------
    cuda.coop.sum
        Shared sum contract and executable example.
    :cpp:struct:`cub::BlockReduce`, :cpp:struct:`cub::WarpReduce`
        C++ primitives used for valid prefixes and explicit block algorithms.
        Full-group built-in reductions use CUDAX.
    """

    return group_primitive_marker(
        "sum",
        group,
        value,
        broadcast=broadcast,
        valid_items=valid_items,
        algorithm=algorithm,
    )


__all__ = ["reduce", "sum"]
