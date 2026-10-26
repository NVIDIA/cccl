# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Describe Scan behavior before selecting a block or warp implementation.

The shared record separates input shape, operator, and initial value from
group topology and CUB algorithm selection. Factories and group planners use
it to choose an overload without importing a compiler backend. Parameter
descriptors identify static expressions or runtime operands; this module
does not read device values or compile callbacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from ._symbols import semantic_token
from ._types import (
    CxxFunction,
    CxxOperator,
    Dependency,
    PythonOperator,
    Reference,
    StatefulOperator,
)


class ScanMode(str, Enum):
    """Whether each output includes its corresponding input."""

    EXCLUSIVE = "exclusive"
    INCLUSIVE = "inclusive"


class ScanValueKind(str, Enum):
    """Distinguish one scalar per thread from a fixed per-thread array.

    Array scans use blocked order: each thread's items are consecutive,
    followed by the next thread's items. A one-item array remains an array
    operand and uses a different signature from a scalar.
    """

    SCALAR = "scalar"
    ARRAY = "array"


_SCAN_OPERATORS = (CxxOperator, PythonOperator)
_INITIAL_VALUES = (CxxFunction, Reference)
_SCAN_OPERATOR_ALIASES = {
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


def normalize_scan_operator_alias(value: object) -> str | None:
    """Resolve a built-in operator spelling before choosing its descriptor.

    Strip surrounding whitespace, ignore case, and treat hyphens as
    underscores. Return a canonical operator name, or ``None`` for an unknown
    string so the caller can issue its own diagnostic. Non-string values raise
    ``TypeError``. This helper does not validate Python callbacks or
    backend-specific selectors.
    """

    if not isinstance(value, str):
        raise TypeError("scan_op must be a string")
    token = value.strip().lower().replace("-", "_")
    return _SCAN_OPERATOR_ALIASES.get(token)


_PREFIX_CALLBACKS = (PythonOperator, StatefulOperator)


def _initial_dtype_matches(
    dtype: Any, initial_value: CxxFunction | Reference
) -> bool:
    """Require the initial value to use the payload's type identity.

    ``Dependency("T")`` refers directly to the payload template parameter.
    Concrete dtypes compare by semantic token, so this check neither performs
    numeric promotion nor relies on a backend's display name for the type.
    """

    initial_dtype = initial_value.dtype
    if isinstance(initial_dtype, Dependency):
        return initial_dtype.name == "T"
    return semantic_token(initial_dtype) == semantic_token(dtype)


@dataclass(frozen=True, eq=False)
class ScanSemantics:
    """Describe the input, operator, and outputs of one scan.

    Use ``make_scan_semantics`` to validate this record. The record has no
    group size or CUB algorithm choice. Its identity includes the operator,
    seed, and prefix-callback descriptors, so different call forms stay
    distinct.

    Attributes
    ----------
    dtype : object
        Input and output element dtype, interpreted by the backend.
    mode : ScanMode
        Whether each prefix excludes or includes its current input item.
    value_kind : ScanValueKind
        Scalar or fixed per-thread array form.
    items_per_thread : int
        Positive item count; scalar form requires one item.
    scan_operator : CxxOperator or PythonOperator or None
        Static operator description. ``None`` requests the sum overload.
    initial_value : CxxFunction or Reference or None
        Static C++ expression or runtime scalar that seeds an exclusive scan.
        Inclusive scans have no initial value. Its dtype must match ``dtype``
        or refer to the payload through ``Dependency("T")``.
    aggregate : bool
        Whether to request a separate scalar reduction of the inputs. This
        aggregate excludes the initial value and is available to every member.
    prefix_callback : PythonOperator or StatefulOperator or None
        Block callback that receives the input aggregate and returns a seed
        for the scan. A stateful descriptor also identifies mutable state.
        This form excludes both initial_value and a separate aggregate.
    """

    dtype: Any
    mode: ScanMode
    value_kind: ScanValueKind
    items_per_thread: int
    scan_operator: CxxOperator | PythonOperator | None = None
    initial_value: CxxFunction | Reference | None = None
    aggregate: bool = False
    prefix_callback: PythonOperator | StatefulOperator | None = None

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            "scan",
            semantic_token(self.dtype),
            self.mode.value,
            self.value_kind.value,
            self.items_per_thread,
            semantic_token(self.scan_operator),
            semantic_token(self.initial_value),
            self.aggregate,
            semantic_token(self.prefix_callback),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScanSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


def make_scan_semantics(
    *,
    dtype: Any,
    mode: str | ScanMode,
    value_kind: str | ScanValueKind,
    items_per_thread: int,
    scan_operator: CxxOperator | PythonOperator | None = None,
    initial_value: CxxFunction | Reference | None = None,
    aggregate: bool = False,
    prefix_callback: PythonOperator | StatefulOperator | None = None,
) -> ScanSemantics:
    """Validate Scan shape and value descriptors independently of a group.

    This checks the operation's intrinsic constraints. Group planning later
    checks supported groups and CUB call variants. In particular, this builder
    allows a custom exclusive operator without an initial value; group
    planning requires a seed or prefix callback to define the first output.

    Parameters
    ----------
    dtype : object
        Non-``None`` payload dtype. The backend determines which dtypes it can
        compile; this function does not translate or promote the dtype.
    mode : str or ScanMode
        ``"exclusive"`` or ``"inclusive"``.
    value_kind : str or ScanValueKind
        ``"scalar"`` or ``"array"``; a one-item array keeps the array form.
    items_per_thread : int
        Positive Python integer, excluding booleans. Must be one for scalars.
    scan_operator : CxxOperator or PythonOperator, optional
        Operator descriptor. ``None`` selects the built-in sum form.
    initial_value : CxxFunction or Reference, optional
        Static expression or runtime value for an exclusive scan. Its dtype
        must match the payload or be ``Dependency("T")``. The frontend handles
        literal conversion before constructing this descriptor.
    aggregate : bool, optional
        Request a separate scalar aggregate that excludes the seed.
    prefix_callback : PythonOperator or StatefulOperator, optional
        Block callback that supplies the seed from the input aggregate.
        Mutually exclusive with initial_value and aggregate output. Group
        planning checks the block-only restriction; the backend compiles
        the callback and handles any mutable state.

    Returns
    -------
    ScanSemantics
        Validated record with canonical mode and operand-form enums.

    Raises
    ------
    TypeError
        An operator, prefix, or initial-value descriptor is unsupported.
        The initial dtype differs from the payload, or ``aggregate`` is
        not a boolean.
    ValueError
        The dtype is missing, an enum value or item count is invalid, scalar
        form has multiple items, an inclusive scan has an initial value, or a
        prefix callback is combined with an initial value or aggregate.
    """

    if dtype is None:
        raise ValueError("dtype must be provided")
    mode = ScanMode(mode)
    value_kind = ScanValueKind(value_kind)
    if (
        not isinstance(items_per_thread, int)
        or isinstance(items_per_thread, bool)
        or items_per_thread < 1
    ):
        raise ValueError("items_per_thread must be a positive integer")
    if value_kind is ScanValueKind.SCALAR and items_per_thread != 1:
        raise ValueError("scalar scan requires items_per_thread == 1")
    if scan_operator is not None and not isinstance(
        scan_operator, _SCAN_OPERATORS
    ):
        raise TypeError(f"unsupported scan operator {scan_operator!r}")
    if initial_value is not None:
        if not isinstance(initial_value, _INITIAL_VALUES):
            raise TypeError(f"unsupported scan initial value {initial_value!r}")
        if not _initial_dtype_matches(dtype, initial_value):
            raise TypeError(
                "scan initial_value dtype must exactly match the payload dtype"
            )
        if mode is ScanMode.INCLUSIVE:
            raise ValueError("inclusive scans do not accept an initial value")
    if not isinstance(aggregate, bool):
        raise TypeError("aggregate must be a bool")
    if prefix_callback is not None and not isinstance(
        prefix_callback, _PREFIX_CALLBACKS
    ):
        raise TypeError(f"unsupported scan prefix callback {prefix_callback!r}")
    if initial_value is not None and prefix_callback is not None:
        raise ValueError(
            "scan initial value and prefix callback are mutually exclusive"
        )
    if aggregate and prefix_callback is not None:
        raise ValueError(
            "scan aggregate and prefix callback are mutually exclusive"
        )

    return ScanSemantics(
        dtype=dtype,
        mode=mode,
        value_kind=value_kind,
        items_per_thread=items_per_thread,
        scan_operator=scan_operator,
        initial_value=initial_value,
        aggregate=aggregate,
        prefix_callback=prefix_callback,
    )


__all__ = [
    "ScanMode",
    "ScanSemantics",
    "ScanValueKind",
    "make_scan_semantics",
    "normalize_scan_operator_alias",
]
