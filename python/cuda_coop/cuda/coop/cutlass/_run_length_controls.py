# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared CUTLASS run-length-decode control validation."""

from __future__ import annotations

from numbers import Integral, Number
from typing import Any


def validate_decoded_window_offset(
    value: Any,
    *,
    scope: str,
) -> Any:
    """Return one static or compiler-produced nonnegative integer offset.

    Compiler integer values stay dynamic. Their nonnegative and resolved-dtype
    representability requirements are collective caller preconditions when the
    values are not known while tracing.
    """

    if isinstance(value, bool):
        raise TypeError(
            f"{scope}.run_length_decode decoded_window_offset must be an integer, "
            "not bool"
        )
    if isinstance(value, Integral):
        normalized = int(value)
        if normalized < 0:
            raise ValueError(
                f"{scope}.run_length_decode decoded_window_offset must be nonnegative"
            )
        return normalized

    if not isinstance(value, type):
        ir_value = getattr(value, "ir_value", None)
        dtype = getattr(value, "dtype", None)
        candidates = (value, dtype, type(value))
        if callable(ir_value) and any(
            isinstance(getattr(candidate, "width", None), Integral)
            and isinstance(getattr(candidate, "signed", None), bool)
            for candidate in candidates
            if candidate is not None
        ):
            literal = getattr(value, "value", None)
            if isinstance(literal, bool):
                raise TypeError(
                    f"{scope}.run_length_decode decoded_window_offset must be "
                    "an integer, not bool"
                )
            if isinstance(literal, Integral) and int(literal) < 0:
                raise ValueError(
                    f"{scope}.run_length_decode decoded_window_offset must be "
                    "nonnegative"
                )
            if (
                isinstance(literal, Number) and not isinstance(literal, Integral)
            ) or isinstance(literal, (str, bytes, bytearray)):
                raise TypeError(
                    f"{scope}.run_length_decode decoded_window_offset must be "
                    "an integer"
                )
            return value

    raise TypeError(
        f"{scope}.run_length_decode decoded_window_offset must be an integer"
    )


__all__ = ["validate_decoded_window_offset"]
