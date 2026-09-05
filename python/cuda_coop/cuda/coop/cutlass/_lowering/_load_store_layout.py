# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Static layout proofs shared by CUTLASS load and store providers."""

from __future__ import annotations

from math import prod
from typing import Any


def _optional_attr(value: Any, name: str) -> Any:
    try:
        return getattr(value, name, None)
    except Exception:
        return None


def _static_layout_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        normalized = value.__index__()
    except Exception:
        return None
    if isinstance(normalized, bool):
        return None
    return int(normalized)


def _layout_leaf_pairs(
    shape: Any,
    strides: Any,
) -> tuple[tuple[Any, Any], ...] | None:
    """Flatten congruent shape/stride trees into scalar leaf pairs."""

    shape_is_tree = isinstance(shape, (tuple, list))
    strides_is_tree = isinstance(strides, (tuple, list))
    if shape_is_tree != strides_is_tree:
        return None
    if not shape_is_tree:
        return ((shape, strides),)
    if not shape or len(shape) != len(strides):
        return None

    leaves: list[tuple[Any, Any]] = []
    for shape_child, strides_child in zip(shape, strides):
        child_leaves = _layout_leaf_pairs(shape_child, strides_child)
        if child_leaves is None:
            return None
        leaves.extend(child_leaves)
    return tuple(leaves)


def _layout_leaves(value: Any) -> tuple[tuple[Any, Any], ...] | None:
    shape = _optional_attr(value, "shape")
    strides = _optional_attr(value, "strides")
    if strides is None:
        strides = _optional_attr(value, "stride")
    if shape is None or strides is None:
        return None
    return _layout_leaf_pairs(shape, strides)


def static_layout_elements(value: Any) -> int | None:
    """Return a statically known layout capacity, when metadata proves one."""

    leaves = _layout_leaves(value)
    if leaves is None:
        return None
    extents = tuple(_static_layout_int(extent) for extent, _ in leaves)
    if any(extent is None or extent <= 0 for extent in extents):
        return None
    return prod(int(extent) for extent in extents)


def contiguous_layout_reason(value: Any) -> str | None:
    """Return why an operand is not statically compact, or ``None``."""

    shape = _optional_attr(value, "shape")
    strides_value = _optional_attr(value, "strides")
    if strides_value is None:
        strides_value = _optional_attr(value, "stride")
    if shape is None and strides_value is None:
        if callable(_optional_attr(value, "to_llvm_ptr")):
            return None
        return "has no inspectable shape/stride contract"
    if shape is None or strides_value is None:
        return "does not expose both shape and stride metadata"
    leaves = _layout_leaf_pairs(shape, strides_value)
    if leaves is None:
        return "has incongruent shape and stride layouts"

    static_shape = tuple(_static_layout_int(extent) for extent, _ in leaves)
    static_strides = tuple(_static_layout_int(stride) for _, stride in leaves)
    if any(value is None for value in (*static_shape, *static_strides)):
        return "is not statically provable as compact"
    normalized_shape = tuple(int(extent) for extent in static_shape)
    normalized_strides = tuple(int(stride) for stride in static_strides)
    if any(extent <= 0 for extent in normalized_shape):
        return "has a non-positive static extent"
    if any(stride <= 0 for stride in normalized_strides):
        return "has a non-positive static stride"

    expected_stride = 1
    for stride, extent in sorted(
        zip(normalized_strides, normalized_shape),
        key=lambda entry: entry[0],
    ):
        if extent == 1:
            continue
        if stride != expected_stride:
            return "is not a compact contiguous layout"
        expected_stride *= extent
    return None
