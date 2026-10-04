# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Carry scalar option choices from a frontend into a primitive factory.

A value alone cannot tell a factory which call to build. An omitted count can
select an unguarded overload; a known count can be embedded in C++ source; a
runtime count needs a device-call argument. ``ArgumentBinding`` keeps that
choice separate from the backend expression that supplies a runtime value.

The helpers here validate scalar representations and build parameter records.
Operation factories apply further rules, such as keeping a count within one
tile. In these helpers, i32 and i64 mean signed 32-bit and 64-bit integers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from typing import Any

from ._types import INT32, ArgumentKind, CxxFunction, RuntimeValue, Value

_I32_MIN = -(1 << 31)
_I32_MAX = (1 << 31) - 1
_I64_MIN = -(1 << 63)
_I64_MAX = (1 << 63) - 1
_U64_MAX = (1 << 64) - 1


class BindingKind(str, Enum):
    """How a scalar option is supplied to a generated primitive call.

    A factory needs more than the option's value: it must know whether to
    select an overload without that option, embed a compile-time constant,
    or accept an argument on each device call. These choices affect the
    generated wrapper signature and whether one specialization can be reused.

    For example, an omitted ``valid_items`` can select an unguarded load;
    a static count of 12 produces a guarded load with 12 embedded in its
    wrapper; a runtime count produces a guarded load whose count is supplied
    on each invocation. A static zero is therefore different from omission.

    Attributes
    ----------
    OMITTED
        The caller did not supply the option. The consuming factory chooses
        an overload or supplies its own default; no payload is stored.
    STATIC
        The value is known while building the specialization and can be
        embedded in generated code. The binding retains that value.
    RUNTIME
        The value is supplied when the device code executes. The binding
        records that requirement without retaining a value or compiler IR.

    Notes
    -----
    ``ArgumentKind`` classifies parameters after a signature is selected and
    has only static and runtime cases. ``BindingKind`` also represents the
    earlier decision to omit an option entirely.
    """

    OMITTED = "omitted"
    STATIC = "static"
    RUNTIME = "runtime"


@dataclass(frozen=True, eq=False)
class ArgumentBinding:
    """Record how a scalar option is supplied and retain its static value.

    Frontends use this record to pass argument decisions into shared
    planning code without carrying backend-specific runtime expressions.
    Factories then select overloads and turn the bindings into parameter
    descriptors: for example, ``i32_parameter`` creates an embedded
    ``CxxFunction`` for a static count or a runtime ``Value`` descriptor.
    Lowering connects runtime descriptors to the actual device operands.

    Prefer ``omitted()``, ``static(value)``, and ``runtime()`` when building
    a record explicitly, or ``binding`` to classify a frontend value.
    A static value contributes to specialization identity; runtime bindings
    carry no payload, allowing calls with different runtime values to share
    the same description.

    Attributes
    ----------
    kind : BindingKind
        Whether the option is omitted, known statically, or runtime-provided.
    value : Any, optional
        Payload for ``STATIC``; must be ``None`` for the other two modes.
        Consuming helpers and factories validate payload types, integer
        widths, and operation-specific bounds.

    Notes
    -----
    Equality and hashing use ``semantic_key`` rather than Python numeric
    equality, so static ``True`` and ``1``, or ``0.0`` and ``-0.0``, remain
    distinct requests. Integer normalization can give equivalent integral
    types the same representation before they enter a specialization key.
    """

    kind: BindingKind
    value: Any = None

    def __post_init__(self) -> None:
        if self.kind is not BindingKind.STATIC and self.value is not None:
            raise ValueError("only static argument bindings may carry a value")

    @classmethod
    def omitted(cls) -> ArgumentBinding:
        """Leave the option's overload or default selection to the factory."""

        return cls(BindingKind.OMITTED)

    @classmethod
    def static(cls, value: Any) -> ArgumentBinding:
        """Retain a compile-time value for later validation and embedding."""

        return cls(BindingKind.STATIC, value)

    @classmethod
    def runtime(cls) -> ArgumentBinding:
        """Request an argument without storing its runtime value."""

        return cls(BindingKind.RUNTIME)

    @property
    def semantic_key(self) -> tuple[str, ...]:
        """Return the binding identity used by equality and hashing.

        The kind alone identifies omitted and runtime bindings.
        Static bindings also include the payload type's module and qualified
        name and the payload's ``repr``. This distinguishes values that
        Python considers numerically equal but may generate different code.
        The key reflects the stored representation; consumers normalize
        values first when equivalent input types should share a key.
        """

        if self.kind is not BindingKind.STATIC:
            return (self.kind.value,)
        value_type = type(self.value)
        return (
            self.kind.value,
            value_type.__module__,
            value_type.__qualname__,
            repr(self.value),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ArgumentBinding):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)

    @property
    def argument_kind(self) -> ArgumentKind | None:
        """Classify a supplied parameter, or return ``None`` for omission.

        Planners use this when describing static and runtime call arguments.
        A factory may separately replace an omitted option with a default.
        """

        if self.kind is BindingKind.OMITTED:
            return None
        if self.kind is BindingKind.STATIC:
            return ArgumentKind.STATIC
        return ArgumentKind.RUNTIME


def binding(value: Any, *, omitted: Any = None) -> ArgumentBinding:
    """Classify an option as omitted, static, or runtime-provided.

    Frontends mark runtime expressions with ``RuntimeValue`` before calling
    this helper. Every other supplied object is treated as a static payload;
    its type alone does not imply that it is a device-time expression.
    In particular, booleans are static values here. The boolean flags accepted
    by some factories to select overloads are a separate convention.

    Parameters
    ----------
    value : Any
        Frontend option value or a ``RuntimeValue`` marker. An existing
        ``ArgumentBinding`` is not passed through; use it directly instead
        of classifying it again.
    omitted : Any, optional
        Sentinel for an absent option, defaulting to ``None``. Compared by
        object identity before checking for ``RuntimeValue``.

    Returns
    -------
    ArgumentBinding
        Omitted when ``value is omitted``, runtime for a ``RuntimeValue``,
        and static with the original payload otherwise. A runtime marker's
        name is discarded; parameter builders supply their own names.
    """

    if value is omitted:
        return ArgumentBinding.omitted()
    if isinstance(value, RuntimeValue):
        return ArgumentBinding.runtime()
    return ArgumentBinding.static(value)


def i32_parameter(
    option: ArgumentBinding,
    *,
    name: str,
    omitted_value: int | None = None,
) -> Value | CxxFunction | None:
    """Turn a count-like option into a signed-i32 parameter descriptor.

    A runtime ``Value`` becomes an operand of the generated wrapper. A
    ``CxxFunction`` embeds the constant expression in the underlying CUB
    call without adding a runtime operand. An omitted binding can remove
    the parameter or substitute a factory-provided constant default.

    Parameters
    ----------
    option : ArgumentBinding
        Binding that determines whether and how the parameter is supplied.
    name : str
        Parameter name, also used to identify the option in validation errors.
    omitted_value : int or None, optional
        Constant to embed when ``option`` is omitted. With the default
        ``None``, omission produces no descriptor. Ignored for static and
        runtime bindings.

    Returns
    -------
    Value or CxxFunction or None
        Runtime i32 descriptor, embedded i32 constant, or no parameter.
        This constructs metadata; wrapper code generation happens later.

    Raises
    ------
    TypeError
        A used static value or omitted default is a boolean or non-integer.
    ValueError
        A used static value or omitted default does not fit signed i32.
        The consuming factory or planner checks operation-specific bounds,
        such as a tile's item count.
    """

    if option.kind is BindingKind.OMITTED:
        if omitted_value is None:
            return None
        value = _normalize_i32(omitted_value, name=name, source="omitted")
        return CxxFunction(str(value), INT32, name=name)
    if option.kind is BindingKind.RUNTIME:
        return Value(INT32, name=name)
    value = _normalize_i32(option.value, name=name, source="static")
    return CxxFunction(str(value), INT32, name=name)


def _normalize_i32(value: Any, *, name: str, source: str) -> int:
    """Convert an integral value to Python ``int`` within signed-i32 bounds.

    Reject booleans and non-integral values with ``TypeError`` and overflow
    with ``ValueError``. ``source`` and ``name`` identify the offending
    binding in diagnostics, such as ``static valid_items``.
    """

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{source} {name} must be an integer")
    normalized = int(value)
    if not _I32_MIN <= normalized <= _I32_MAX:
        raise ValueError(f"{source} {name} must fit a signed 32-bit integer")
    return normalized


def normalize_i32_binding(
    option: ArgumentBinding,
    *,
    name: str,
) -> ArgumentBinding:
    """Validate a static i32 binding and normalize its identity.

    Equivalent integral values, such as Python ``1`` and NumPy ``int32(1)``,
    become the same Python ``int`` payload. This avoids separate semantic
    keys for constants that generate the same i32 argument.

    Parameters
    ----------
    option : ArgumentBinding
        Binding to normalize; omitted and runtime bindings pass through.
    name : str
        Option name included in validation errors.

    Returns
    -------
    ArgumentBinding
        A new static binding with a Python ``int`` payload, or the original
        non-static binding. This helper accepts negative values within the
        signed-i32 range. Factories check operation-specific bounds.

    Raises
    ------
    TypeError
        The static payload is not integral or is a boolean.
    ValueError
        The static payload is outside ``[-2**31, 2**31 - 1]``.
    """

    if option.kind is not BindingKind.STATIC:
        return option
    return ArgumentBinding.static(
        _normalize_i32(option.value, name=name, source="static")
    )


def _normalize_i64(value: Any, *, name: str, source: str) -> int:
    """Convert an integral value to Python ``int`` within signed-i64 bounds.

    Apply the same type checks and diagnostic labels as ``_normalize_i32``.
    Accept negative values here; pointer-offset factories apply their own
    nonnegative bounds after this representation check.
    """

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{source} {name} must be an integer")
    normalized = int(value)
    if not _I64_MIN <= normalized <= _I64_MAX:
        raise ValueError(f"{source} {name} must fit a signed 64-bit integer")
    return normalized


def normalize_i64_binding(
    option: ArgumentBinding,
    *,
    name: str,
) -> ArgumentBinding:
    """Validate a static i64 binding and normalize its identity.

    Pointer-offset planning uses this to give equal integral offsets the
    same Python ``int`` payload and semantic key. Omitted and runtime
    bindings already describe their supply mode without a value to normalize.

    Parameters
    ----------
    option : ArgumentBinding
        Binding to normalize; omitted and runtime bindings pass through.
    name : str
        Option name included in validation errors.

    Returns
    -------
    ArgumentBinding
        A new static binding with a Python ``int`` payload, or the original
        non-static binding. Factories and group planning check nonnegative
        offsets and combined tile-origin bounds separately.

    Raises
    ------
    TypeError
        The static payload is not integral or is a boolean.
    ValueError
        The static payload is outside ``[-2**63, 2**63 - 1]``.
    """

    if option.kind is not BindingKind.STATIC:
        return option
    return ArgumentBinding.static(
        _normalize_i64(option.value, name=name, source="static")
    )


def cxx_scalar_literal(value: Any, *, name: str) -> str:
    """Render a static scalar as an expression for a generated CUB call.

    Factories use this for embedded values such as a load's ``oob_default``.
    The result is source text for a ``CxxFunction`` descriptor, whose dtype
    is supplied separately by the factory.

    Parameters
    ----------
    value : Any
        Boolean, integral, or real scalar, or an object whose ``value``
        attribute contains one. Integral values may span signed-i64 minimum
        through unsigned-i64 maximum. Real values must remain finite after
        conversion to Python ``float``.
    name : str
        Option name included in validation errors.

    Returns
    -------
    str
        C++ boolean literal, integer expression, or floating-point literal.
        Integers above signed-i64 maximum receive a ``ULL`` suffix; signed
        i64 minimum uses an expression that avoids an overflowing positive
        literal. Conversion into the eventual CUB element type is left to
        later compilation.

    Raises
    ------
    TypeError
        The unwrapped value is not a supported numeric scalar.
    ValueError
        An integer is outside ``[-2**63, 2**64 - 1]`` or a real is nonfinite.
    """

    scalar = getattr(value, "value", value)
    if isinstance(scalar, bool):
        return "true" if scalar else "false"
    if isinstance(scalar, Integral):
        normalized = int(scalar)
        if not _I64_MIN <= normalized <= _U64_MAX:
            raise ValueError(f"static {name} must fit a 64-bit integer")
        if normalized == _I64_MIN:
            return "(-9223372036854775807LL - 1LL)"
        if normalized > _I64_MAX:
            return f"{normalized}ULL"
        return str(normalized)
    if isinstance(scalar, Real):
        normalized = float(scalar)
        if not math.isfinite(normalized):
            raise ValueError(f"static {name} must be finite")
        return repr(normalized)
    raise TypeError(f"static {name} must be a numeric scalar")


__all__ = [
    "ArgumentBinding",
    "BindingKind",
    "binding",
    "cxx_scalar_literal",
    "i32_parameter",
    "normalize_i32_binding",
    "normalize_i64_binding",
]
