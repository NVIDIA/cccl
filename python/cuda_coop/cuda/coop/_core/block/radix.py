# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared backend-neutral radix primitive vocabulary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Any

from .._bindings import ArgumentBinding, BindingKind, binding
from .._symbols import semantic_token
from ._common import normalize_boolean_option


class RadixOrder(str, Enum):
    """Ordering selected by a radix rank or sort operation."""

    ASCENDING = "ascending"
    DESCENDING = "descending"

    @property
    def descending(self) -> bool:
        return self is RadixOrder.DESCENDING

    @property
    def cpp_bool(self) -> str:
        return "true" if self.descending else "false"


def normalize_radix_order(descending: bool | RadixOrder) -> RadixOrder:
    """Normalize the public descending flag without truthiness coercion."""

    if isinstance(descending, RadixOrder):
        return descending
    value = normalize_boolean_option("descending", descending)
    return RadixOrder.DESCENDING if value else RadixOrder.ASCENDING


def _bit_binding(name: str, value: Any) -> ArgumentBinding:
    option = value if isinstance(value, ArgumentBinding) else binding(value)
    if option.kind is BindingKind.OMITTED:
        raise ValueError(f"{name} must be provided")
    if option.kind is BindingKind.STATIC:
        static_value = option.value
        if not isinstance(static_value, Integral) or isinstance(static_value, bool):
            raise ValueError(f"{name} must be an integer")
        return ArgumentBinding.static(int(static_value))
    return option


def _optional_positive_int(name: str, value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class RadixBitRange:
    """Static or runtime-classified half-open radix bit interval."""

    begin_bit: ArgumentBinding
    end_bit: ArgumentBinding
    bit_width: int | None = None

    @property
    def is_static(self) -> bool:
        return (
            self.begin_bit.kind is BindingKind.STATIC
            and self.end_bit.kind is BindingKind.STATIC
        )

    @property
    def static_begin_bit(self) -> int | None:
        if self.begin_bit.kind is BindingKind.STATIC:
            return int(self.begin_bit.value)
        return None

    @property
    def static_end_bit(self) -> int | None:
        if self.end_bit.kind is BindingKind.STATIC:
            return int(self.end_bit.value)
        return None

    @property
    def radix_bits(self) -> int | None:
        begin_bit = self.static_begin_bit
        end_bit = self.static_end_bit
        if begin_bit is None or end_bit is None:
            return None
        return end_bit - begin_bit

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            semantic_token(self.begin_bit),
            semantic_token(self.end_bit),
            self.bit_width,
        )


def make_radix_bit_range(
    *,
    begin_bit: Any,
    end_bit: Any,
    bit_width: int | None = None,
) -> RadixBitRange:
    """Validate and classify a resolved radix bit interval.

    Runtime payloads are represented by :class:`ArgumentBinding` rather than
    retained in the core record. Static bounds receive the full validation
    available from the key bit width.
    """

    bit_width = _optional_positive_int("bit_width", bit_width)
    begin = _bit_binding("begin_bit", begin_bit)
    end = _bit_binding("end_bit", end_bit)

    static_begin = begin.value if begin.kind is BindingKind.STATIC else None
    static_end = end.value if end.kind is BindingKind.STATIC else None
    if static_begin is not None and static_begin < 0:
        raise ValueError("begin_bit must be non-negative")
    if bit_width is not None and static_begin is not None and static_begin >= bit_width:
        raise ValueError("begin_bit must be less than the dtype bit width")
    if (
        static_begin is not None
        and static_end is not None
        and static_end <= static_begin
    ):
        raise ValueError("end_bit must be greater than begin_bit")
    if static_end is not None and static_end < 1:
        raise ValueError("end_bit must be positive")
    if bit_width is not None and static_end is not None and static_end > bit_width:
        raise ValueError("end_bit must not exceed the dtype bit width")

    return RadixBitRange(begin, end, bit_width)


def resolve_static_radix_end_bit(
    *,
    begin_bit: Any,
    end_bit: Any | None,
    bit_width: int | None,
    default_radix_bits: int | None = None,
    default_to_bit_width: bool = False,
    clamp_default: bool = False,
) -> int:
    """Resolve a frontend's static default and validate the resulting interval."""

    begin = _bit_binding("begin_bit", begin_bit)
    if begin.kind is not BindingKind.STATIC:
        raise ValueError("begin_bit must be a static integer")
    bit_width = _optional_positive_int("bit_width", bit_width)
    if end_bit is None:
        if default_radix_bits is not None:
            if clamp_default and bit_width is None:
                raise ValueError(
                    "end_bit must be provided when bit_width is unavailable"
                )
            default_radix_bits = _optional_positive_int(
                "default_radix_bits", default_radix_bits
            )
            assert default_radix_bits is not None
            end_bit = int(begin.value) + default_radix_bits
            if clamp_default and bit_width is not None:
                end_bit = min(end_bit, bit_width)
        elif default_to_bit_width and bit_width is not None:
            end_bit = bit_width
        else:
            raise ValueError("end_bit must be provided")

    interval = make_radix_bit_range(
        begin_bit=begin,
        end_bit=end_bit,
        bit_width=bit_width,
    )
    if not interval.is_static:
        raise ValueError("begin_bit and end_bit must be static integers")
    assert interval.static_end_bit is not None
    return interval.static_end_bit


__all__ = [
    "RadixBitRange",
    "RadixOrder",
    "make_radix_bit_range",
    "normalize_radix_order",
    "resolve_static_radix_end_bit",
]
