# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Declare qualified Reduce and Sum calls for Numba-CUDA-MLIR kernels.

These markers share the common reduction contract and add native local-array
operands and supported device-operator forms. The whole-function planner
resolves them to CUB providers before ordinary type inference.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypeVar

import numpy

from ..._typing import (
    CommonNumericScalar,
    CommonThreadDataLike,
    ReduceAlgorithm,
    ReduceOperator,
    TempStorageLike,
    ValidItems,
)
from .._compiler._operations import group_operation
from .._thread_group import BlockGroup, WarpGroup
from ._marker import group_primitive_marker

_ItemT = TypeVar("_ItemT", bound=CommonNumericScalar)


@group_operation(
    "reduce",
    family_module="cuda.coop.numba_mlir._compiler._group_reduce",
)
def reduce(
    group: BlockGroup | WarpGroup,
    value: CommonThreadDataLike[_ItemT] | _ItemT | numpy.ndarray,
    /,
    *,
    binary_op: ReduceOperator
    | Callable[[_ItemT, _ItemT], _ItemT]
    | None = None,
    valid_items: ValidItems | None = None,
    algorithm: ReduceAlgorithm
    | Literal["raking", "warp_reductions"]
    | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ItemT:
    """Reduce values with a built-in alias or custom device operator.

    See :func:`cuda.coop.reduce` for the shared parameters, defaults, supported
    groups, and examples. Qualified operand forms are described below.

    Parameters
    ----------
    value : numeric scalar, cuda.coop.ThreadDataLike, or local array
        Also accepts a one-dimensional ``cuda.local.array`` with a fixed
        extent. Its elements contribute to one scalar result, just like
        :ref:`ThreadData <coop-thread-data>`. The input is preserved.
    binary_op : str or callable, optional
        Also accepts ``operator``/NumPy aliases such as ``operator.add`` and
        ``numpy.add``, which retain the built-in behavior, or a stateless
        device callable ``op(left, right)`` returning the input dtype.
        Custom operators must be associative and require a complete block
        or physical or logical warp.
        Custom reductions accept scalar or fixed-size array inputs.
        For a block, use
        ``algorithm=None``, ``"raking"``, or ``"warp_reductions"``;
        ``"raking_commutative_only"`` requires proven commutativity and is
        unavailable for Python callables. ``None`` selects sum.

    Returns
    -------
    numeric scalar
        Reduced value with the input dtype, defined only at group rank zero.

    See Also
    --------
    cuda.coop.reduce
        Shared reduction contract and executable examples.
    :cpp:struct:`cub::BlockReduce`, :cpp:struct:`cub::WarpReduce`
        C++ primitives used for block and warp reductions.

    Examples
    --------
    Reduce a tile with a custom device maximum and read the leader-owned
    result.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_qualified_reduce_examples.py
        :language: python
        :start-after: # qualified-reduce-example-begin
        :end-before: # qualified-reduce-example-end
        :dedent: 4
    """

    return group_primitive_marker(
        "reduce",
        group,
        value,
        binary_op=binary_op,
        valid_items=valid_items,
        algorithm=algorithm,
        temp_storage=temp_storage,
    )


@group_operation(
    "sum",
    family_module="cuda.coop.numba_mlir._compiler._group_reduce",
)
def sum(
    group: BlockGroup | WarpGroup,
    value: CommonThreadDataLike[_ItemT] | _ItemT | numpy.ndarray,
    /,
    *,
    valid_items: ValidItems | None = None,
    algorithm: ReduceAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ItemT:
    """Sum values with Numba-CUDA-MLIR.

    See :func:`cuda.coop.sum` for all parameters, defaults, supported groups,
    and examples. The qualified API also accepts a one-dimensional
    ``cuda.local.array`` with a fixed extent as ``value``. Its elements
    contribute to one scalar result, and the input is preserved.

    Returns
    -------
    numeric scalar
        Sum with the input dtype, defined only at group rank zero.

    See Also
    --------
    cuda.coop.sum
        Shared sum contract and executable example.
    :cpp:struct:`cub::BlockReduce`, :cpp:struct:`cub::WarpReduce`
        C++ primitives used for block and warp reductions.

    Examples
    --------
    Write the sum from the leader of each eight-lane logical warp.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_qualified_reduce_examples.py
        :language: python
        :start-after: # qualified-sum-example-begin
        :end-before: # qualified-sum-example-end
        :dedent: 4
    """

    return group_primitive_marker(
        "sum",
        group,
        value,
        valid_items=valid_items,
        algorithm=algorithm,
        temp_storage=temp_storage,
    )


__all__ = ["reduce", "sum"]
