# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Validate CUB reductions before delegating to an active compiler."""

from __future__ import annotations

from enum import Enum
from typing import Any, TypeVar

from cuda.coop._typing import (
    CommonNumericScalar,
    CommonThreadDataLike,
    ReduceAlgorithm,
    ReduceOperator,
    TempStorageLike,
    ValidItems,
)

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
from .thread_group import BlockGroup, WarpGroup

_ItemT = TypeVar("_ItemT", bound=CommonNumericScalar)


_PARTIAL_REDUCTION_GROUP_KINDS = frozenset(
    {"block", "warp", "threads_within_warp"}
)
_COMMON_REDUCTION_GROUP_KINDS = ("warp", "threads_within_warp", "block")
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
    """Normalize the common algorithm selector during active dispatch.

    Accept only the shared block choices. Outside an active backend, leave the
    value intact so the operation can report the missing compiler context.
    """

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
    """Normalize shared operator aliases during active dispatch.

    Custom operators need a backend-qualified import. Outside a compiler
    environment, return the value unchanged; the operation marker reports
    the missing context.
    """

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
    value: object,
    *,
    valid_items: Any,
    algorithm: Any,
    temp_storage: Any,
) -> None:
    """Validate supported CUB groups, scalar prefixes, and block scratch."""

    if _backend_module_name() is None:
        return
    if isinstance(group, ThreadGroup) and group.kind == "grid":
        raise NotImplementedError(
            f"cuda.coop.{operation} does not support grid groups because grid "
            "reduction requires hidden per-launch workspace"
        )
    _validate_common_operation_group(operation, group)
    if group.kind != "block" and temp_storage is not None:
        raise ValueError(
            f"cuda.coop.{operation} temp_storage requires a block group"
        )
    if valid_items is not None:
        static_valid_items = _validate_common_integer_value(
            operation,
            "valid_items",
            valid_items,
        )
        if group.kind not in _PARTIAL_REDUCTION_GROUP_KINDS:
            raise ValueError(
                f"cuda.coop.{operation} valid_items requires "
                "a block or warp group"
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
            if (
                group.static_size is not None
                and static_valid_items > group.static_size
            ):
                raise ValueError(
                    f"cuda.coop.{operation} valid_items {static_valid_items} "
                    f"exceeds group size {group.static_size}"
                )
    if algorithm is not None and group.kind != "block":
        raise ValueError(
            f"cuda.coop.{operation} algorithm selection requires a block group"
        )


def _validate_common_reduce_value(
    operation: str,
    value: object,
    operator: Any,
) -> None:
    """Require common numeric payloads and integer bitwise operands.

    Readable payloads contribute their elements without mutation. Check
    dtypes here so delegation retains the common input contract.
    """

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
    regroups operations. See :ref:`temporary storage <coop-common-storage>` for
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

    See :ref:`CUTLASS Reduce and Sum <coop-cutlass-reduce>` for CuTe examples.
    """

    algorithm = _common_reduce_algorithm("reduce", algorithm)
    binary_op = _common_reduce_operator(binary_op)
    if _backend_module_name() is not None:
        _validate_common_reduce_value("reduce", value, binary_op)
    _validate_common_reduce_options(
        "reduce",
        group,
        value,
        temp_storage=temp_storage,
        valid_items=valid_items,
        algorithm=algorithm,
    )
    return _group_primitive_marker(
        "reduce",
        group,
        value,
        binary_op=binary_op,
        temp_storage=temp_storage,
        valid_items=valid_items,
        algorithm=algorithm,
    )


@_common_group_operation(
    "sum",
    group_kinds=_COMMON_REDUCTION_GROUP_KINDS,
)
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

    See :ref:`CUTLASS Reduce and Sum <coop-cutlass-reduce>` for CuTe examples.
    """

    algorithm = _common_reduce_algorithm("sum", algorithm)
    if _backend_module_name() is not None:
        _validate_common_reduce_value("sum", value, None)
    _validate_common_reduce_options(
        "sum",
        group,
        value,
        temp_storage=temp_storage,
        valid_items=valid_items,
        algorithm=algorithm,
    )
    return _group_primitive_marker(
        "sum",
        group,
        value,
        temp_storage=temp_storage,
        valid_items=valid_items,
        algorithm=algorithm,
    )


__all__ = [
    "_common_reduce_operator",
    "_validate_common_reduce_value",
    "reduce",
    "sum",
]
