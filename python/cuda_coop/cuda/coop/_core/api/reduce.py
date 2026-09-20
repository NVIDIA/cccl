# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Common cooperative reduction entry points."""

from __future__ import annotations

from enum import Enum
from typing import Any

from ..dtype_policy import validate_common_integer_value_dtype_name
from ..thread_group import ThreadGroup
from ._dispatch import (
    _backend_module_name,
    _common_group_operation,
    _group_primitive_marker,
    _validate_common_operation_group,
)
from ._payload import (
    _ReadableThreadDataLike,
    _validate_common_integer_value,
    _validate_common_numeric_value,
)

_PARTIAL_REDUCTION_GROUP_KINDS = frozenset({"block", "warp", "threads_within_warp"})
_COMMON_REDUCTION_GROUP_KINDS = (
    "thread",
    "warp",
    "threads_within_warp",
    "block",
    "warps_within_block",
    "cluster",
)
_COMMON_REDUCE_ALGORITHMS = frozenset(
    {"raking_commutative_only", "raking", "warp_reductions"}
)
_COMMON_OPERATOR_ALIASES = {
    "+": "sum",
    "sum": "sum",
    "add": "sum",
    "plus": "sum",
    "*": "multiplies",
    "mul": "multiplies",
    "multiply": "multiplies",
    "multiplies": "multiplies",
    "min": "min",
    "minimum": "min",
    "max": "max",
    "maximum": "max",
    "&": "bit_and",
    "bit_and": "bit_and",
    "|": "bit_or",
    "bit_or": "bit_or",
    "^": "bit_xor",
    "bit_xor": "bit_xor",
}
_BITWISE_OPERATORS = frozenset({"bit_and", "bit_or", "bit_xor"})


def _is_plain_string(value: Any) -> bool:
    return isinstance(value, str) and not isinstance(value, Enum)


def _common_reduce_algorithm(operation: str, value: Any) -> Any:
    if _backend_module_name() is None or value is None:
        return value
    if not _is_plain_string(value):
        raise TypeError(f"cuda.coop.{operation} algorithm must be a string")
    token = value.strip().lower().replace("-", "_")
    if token not in _COMMON_REDUCE_ALGORITHMS:
        choices = ", ".join(sorted(_COMMON_REDUCE_ALGORITHMS))
        raise ValueError(
            f"cuda.coop.{operation} algorithm must be one of: {choices}; "
            "use a backend-qualified import for backend-only controls"
        )
    return token


def _common_reduce_operator(value: Any) -> Any:
    if _backend_module_name() is None or value is None:
        return value
    if not _is_plain_string(value):
        raise TypeError("cuda.coop.reduce binary_op must be a string")
    token = value.strip().lower().replace("-", "_")
    try:
        return _COMMON_OPERATOR_ALIASES[token]
    except KeyError:
        choices = ", ".join(sorted(set(_COMMON_OPERATOR_ALIASES.values())))
        raise ValueError(
            "cuda.coop.reduce binary_op must be one of: "
            f"{choices}; use a backend-qualified import for custom operators"
        ) from None


def _validate_common_reduce_options(
    operation: str,
    group: ThreadGroup,
    value: Any,
    *,
    broadcast: bool,
    valid_items: Any,
    algorithm: Any,
) -> None:
    if _backend_module_name() is None:
        return
    if isinstance(group, ThreadGroup) and group.kind == "grid":
        raise NotImplementedError(
            f"cuda.coop.{operation} does not support grid groups because grid "
            "reduction requires hidden per-launch workspace"
        )
    _validate_common_operation_group(operation, group)
    if not isinstance(broadcast, bool):
        raise TypeError(f"cuda.coop.{operation} broadcast must be a bool")
    if valid_items is not None:
        static_valid_items = _validate_common_integer_value(
            operation,
            "valid_items",
            valid_items,
        )
        if group.kind not in _PARTIAL_REDUCTION_GROUP_KINDS:
            raise ValueError(
                f"cuda.coop.{operation} valid_items requires a block or warp group"
            )
        if broadcast is not False:
            raise ValueError(
                f"cuda.coop.{operation} valid_items requires broadcast=False"
            )
        if isinstance(value, _ReadableThreadDataLike):
            raise ValueError(
                f"cuda.coop.{operation} valid_items supports scalar values only"
            )
        if static_valid_items is not None:
            if static_valid_items < 1:
                raise ValueError(
                    f"cuda.coop.{operation} valid_items must be at least 1"
                )
            if group.static_size is not None and static_valid_items > group.static_size:
                raise ValueError(
                    f"cuda.coop.{operation} valid_items {static_valid_items} "
                    f"exceeds group size {group.static_size}"
                )
    if algorithm is not None:
        if group.kind != "block":
            raise ValueError(
                f"cuda.coop.{operation} algorithm selection requires a block group"
            )
        if broadcast is not False:
            raise ValueError(
                f"cuda.coop.{operation} algorithm selection requires broadcast=False"
            )


def _validate_common_reduce_value(
    operation: str,
    value: Any,
    operator: Any,
) -> None:
    dtype_name = _validate_common_numeric_value(
        operation,
        "value",
        value,
        allow_readonly_thread_data=True,
    )
    assert dtype_name is not None
    if operator in _BITWISE_OPERATORS:
        validate_common_integer_value_dtype_name(
            dtype_name,
            operation=operation,
            parameter="value",
        )


@_common_group_operation(
    "reduce",
    group_kinds=_COMMON_REDUCTION_GROUP_KINDS,
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
    """Combine a group's values into one scalar.

    Parameters
    ----------
    group : cuda.coop.ThreadGroup
        Participating :ref:`thread group <coop-common-groups>`. Supports a
        single thread, physical or logical warp, block, mapped group of warps,
        or cluster. Grid reductions are unsupported. Every member must call
        the primitive, including members excluded by ``valid_items``.
    value : numeric scalar or cuda.coop.ThreadDataLike
        Each thread's contribution. A :ref:`per-thread payload
        <coop-common-payloads>` contributes all its elements to the same scalar
        reduction; its dtype and fixed extent must agree across the group.
        Input values are preserved. Supported dtypes are signed and unsigned
        8-, 16-, 32-, and 64-bit integers, ``float32``, and ``float64``.
    binary_op : str, optional
        Compile-time operator: ``"sum"`` (the default), ``"multiplies"``,
        ``"min"``, ``"max"``, ``"bit_and"``, ``"bit_or"``, or ``"bit_xor"``.
        ``None`` selects sum. Bitwise operators require integer values.
        Operator aliases include ``"+"``, ``"*"``, ``"&"``, ``"|"``, and
        ``"^"``. :func:`cuda.coop.numba_mlir.reduce` also accepts custom
        operators. :func:`cuda.coop.cutlass.reduce` accepts recognized Python
        and NumPy aliases for built-in operators, but no custom callbacks.
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
        a block and ``broadcast=False``. All common operators support these
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
    :ref:`temporary storage <coop-common-storage>` automatically.

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

    The :ref:`CUTLASS reduction example <coop-cutlass-reduce>` covers CuTe
    block and logical-warp reductions, valid prefixes, and result ownership.
    """

    algorithm = _common_reduce_algorithm("reduce", algorithm)
    binary_op = _common_reduce_operator(binary_op)
    if _backend_module_name() is not None:
        _validate_common_reduce_value("reduce", value, binary_op)
    _validate_common_reduce_options(
        "reduce",
        group,
        value,
        broadcast=broadcast,
        valid_items=valid_items,
        algorithm=algorithm,
    )
    return _group_primitive_marker(
        "reduce",
        group,
        value,
        binary_op=binary_op,
        broadcast=broadcast,
        valid_items=valid_items,
        algorithm=algorithm,
    )


@_common_group_operation(
    "sum",
    group_kinds=_COMMON_REDUCTION_GROUP_KINDS,
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
    """Add a group's values and return one scalar.

    This is equivalent to :func:`cuda.coop.reduce` with ``binary_op="sum"``.

    Parameters
    ----------
    group : cuda.coop.ThreadGroup
        Participating :ref:`thread group <coop-common-groups>`. Supports a
        single thread, physical or logical warp, block, mapped group of warps,
        or cluster. Grid reductions are unsupported. Every member must call
        the primitive.
    value : numeric scalar or cuda.coop.ThreadDataLike
        Each thread's contribution. A :ref:`per-thread payload
        <coop-common-payloads>` contributes all its elements; its dtype and fixed
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
    :ref:`temporary storage <coop-common-storage>` automatically.

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

    For CuTe payload sums and a scalar valid-prefix sum, see
    :ref:`CUTLASS Reduce and Sum <coop-cutlass-reduce>`.
    """

    algorithm = _common_reduce_algorithm("sum", algorithm)
    if _backend_module_name() is not None:
        _validate_common_reduce_value("sum", value, None)
    _validate_common_reduce_options(
        "sum",
        group,
        value,
        broadcast=broadcast,
        valid_items=valid_items,
        algorithm=algorithm,
    )
    return _group_primitive_marker(
        "sum",
        group,
        value,
        broadcast=broadcast,
        valid_items=valid_items,
        algorithm=algorithm,
    )


__all__ = ["reduce", "sum"]
