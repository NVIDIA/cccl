# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared load/store argument handling for CUTLASS tensor scopes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import prod
from typing import Any

from .._prims import is_cutlass_array_operand
from ._thread_data import (
    ThreadData,
    _coerce_thread_payload,
    _validate_items_per_thread,
)

LOAD_STORE_ALGORITHM_ALIASES = {
    "direct": "direct",
    "striped": "striped",
    "vectorize": "direct",
}
BLOCK_UNSUPPORTED_LOAD_STORE_ALGORITHMS = frozenset(
    ("transpose", "warp_transpose", "warp_transpose_timesliced")
)
WARP_UNSUPPORTED_LOAD_STORE_ALGORITHMS = frozenset(("transpose",))


class ScopedLoadStoreRoute(str, Enum):
    """Lowering selected before a scoped load/store call mutates provider state."""

    CANONICAL_CUB = "canonical_cub"
    CUTE_INDEXING_PAYLOAD_ADAPTER = "cute_indexing_payload_adapter"


@dataclass(frozen=True)
class ScopedLoadStoreRouteDecision:
    route: ScopedLoadStoreRoute
    reason: str
    exact_block_dim: tuple[int, int, int] | None


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


def _has_raw_pointer_contract(value: Any) -> bool:
    candidates = [value]
    for name in ("iterator", "pointer", "ptr", "_pointer", "_ptr"):
        candidate = _optional_attr(value, name)
        if candidate is not None:
            candidates.append(candidate)
    if callable(_optional_attr(value, "data_ptr")):
        return True
    return any(
        callable(_optional_attr(candidate, "to_llvm_ptr"))
        or _optional_attr(candidate, "llvm_ptr") is not None
        for candidate in candidates
    )


def classify_scoped_load_store_route(
    memory_operand: Any,
    *,
    scope: str,
    primitive_name: str,
    launch_kwargs: dict[str, Any],
    dtype: Any,
    items_per_thread: int,
    threads_in_warp: int | None = None,
) -> ScopedLoadStoreRouteDecision:
    """Choose canonical CUB or the explicit CuTe indexing payload adapter."""

    if is_cutlass_array_operand(memory_operand):
        raise TypeError(
            f"{scope}.{primitive_name} cutlass.Array values must use the Prims "
            "payload adapter"
        )
    if items_per_thread < 1:
        raise ValueError(f"{scope}.{primitive_name} items_per_thread must be positive")

    from ._launch import infer_launch_facts

    launch = infer_launch_facts(
        launch_kwargs,
        scope=scope,
        primitive_name=primitive_name,
    )
    exact_block_dim = launch.exact_block_dim
    if exact_block_dim is None:
        return ScopedLoadStoreRouteDecision(
            ScopedLoadStoreRoute.CUTE_INDEXING_PAYLOAD_ADAPTER,
            "exact block dimensions are unproven",
            None,
        )

    if scope.endswith((".warp", "._warp")):
        if threads_in_warp != 32:
            return ScopedLoadStoreRouteDecision(
                ScopedLoadStoreRoute.CUTE_INDEXING_PAYLOAD_ADAPTER,
                "logical warp width is not the physical width 32",
                exact_block_dim,
            )
        block_threads = prod(exact_block_dim)
        if block_threads < 32 or block_threads % 32 != 0:
            return ScopedLoadStoreRouteDecision(
                ScopedLoadStoreRoute.CUTE_INDEXING_PAYLOAD_ADAPTER,
                "the exact block shape does not contain complete physical warps",
                exact_block_dim,
            )

    layout_reason = contiguous_layout_reason(memory_operand)
    if layout_reason is not None:
        return ScopedLoadStoreRouteDecision(
            ScopedLoadStoreRoute.CUTE_INDEXING_PAYLOAD_ADAPTER,
            layout_reason,
            exact_block_dim,
        )
    if not _has_raw_pointer_contract(memory_operand):
        return ScopedLoadStoreRouteDecision(
            ScopedLoadStoreRoute.CUTE_INDEXING_PAYLOAD_ADAPTER,
            "does not expose a raw iterator/pointer contract",
            exact_block_dim,
        )

    from . import _cub_load_store_provider as provider

    proof, reason = provider._contiguous_memory_proof(
        memory_operand,
        primitive_name=primitive_name,
    )
    if proof is None:
        return ScopedLoadStoreRouteDecision(
            ScopedLoadStoreRoute.CUTE_INDEXING_PAYLOAD_ADAPTER,
            reason,
            exact_block_dim,
        )

    memory_dtype = provider._memory_dtype(memory_operand)
    if memory_dtype is None:
        return ScopedLoadStoreRouteDecision(
            ScopedLoadStoreRoute.CUTE_INDEXING_PAYLOAD_ADAPTER,
            "memory dtype is unproven",
            exact_block_dim,
        )
    try:
        memory_type = provider._resolve_type(
            memory_dtype,
            allowed=provider._ALL_PROVIDER_TYPES,
            feature=primitive_name,
        )
    except (TypeError, ValueError, NotImplementedError):
        return ScopedLoadStoreRouteDecision(
            ScopedLoadStoreRoute.CUTE_INDEXING_PAYLOAD_ADAPTER,
            "memory dtype is not supported by the canonical CUB provider",
            exact_block_dim,
        )
    if dtype is not None:
        try:
            requested_type = provider._resolve_type(
                dtype,
                allowed=provider._ALL_PROVIDER_TYPES,
                feature=primitive_name,
            )
        except (TypeError, ValueError, NotImplementedError):
            return ScopedLoadStoreRouteDecision(
                ScopedLoadStoreRoute.CUTE_INDEXING_PAYLOAD_ADAPTER,
                "payload dtype is not supported by the canonical CUB provider",
                exact_block_dim,
            )
        if requested_type is not memory_type:
            raise TypeError(
                f"{scope}.{primitive_name} memory dtype does not match payload dtype"
            )

    return ScopedLoadStoreRouteDecision(
        ScopedLoadStoreRoute.CANONICAL_CUB,
        "raw contiguous pointer, compatible dtype, and exact group shape proven",
        exact_block_dim,
    )


def validate_payload_selector(
    payload: Any,
    *,
    scope: str,
    primitive_name: str,
) -> None:
    """Reject payload selection on the private tensor adapter."""
    if payload is not None:
        raise ValueError(
            f"{scope}.{primitive_name} payload must be prims when an explicit "
            "payload selector is required"
        )


def normalize_algorithm(
    value: Any,
    *,
    scope: str,
    primitive_name: str,
    unsupported_algorithms: frozenset[str],
    error_algorithm_names: frozenset[str] | None = None,
) -> str:
    if value is None:
        return "direct"

    token = getattr(value, "name", value)
    token = str(token).split(".")[-1].lower().replace("-", "_")
    if token in unsupported_algorithms:
        raise NotImplementedError(
            f"{scope}.{primitive_name} algorithm {token!r} is not implemented yet"
        )
    if token not in LOAD_STORE_ALGORITHM_ALIASES:
        choices = (
            error_algorithm_names
            if error_algorithm_names is not None
            else frozenset(LOAD_STORE_ALGORITHM_ALIASES)
        )
        raise ValueError(
            f"{scope}.{primitive_name} algorithm must be one of "
            + ", ".join(sorted(choices))
        )
    return LOAD_STORE_ALGORITHM_ALIASES[token]


def merge_valid_items(
    *,
    scope: str,
    valid_items: Any,
    num_valid_items: Any,
    primitive_name: str,
) -> Any:
    if num_valid_items is not None:
        if valid_items is not None:
            raise TypeError(
                f"{scope}.{primitive_name} accepts only one of "
                "valid_items or num_valid_items"
            )
        return num_valid_items
    return valid_items


def infer_source_dtype(source: Any) -> Any:
    for attr_name in ("element_type", "dtype"):
        dtype = getattr(source, attr_name, None)
        if dtype is not None:
            return dtype
    return None


def reject_cutlass_array_operand(
    value: Any,
    *,
    scope: str,
    primitive_name: str,
    operand_name: str,
) -> None:
    if not is_cutlass_array_operand(value):
        return

    scope_name = scope.rsplit(".", 1)[-1]
    raise TypeError(
        f"{scope}.{primitive_name} expects a CuTe tensor {operand_name}; "
        "cutlass.Array values must use the public "
        f"cuda.coop.cutlass.{scope_name}.{primitive_name} path directly, "
        "where cutlass.Array operands are detected automatically or can be selected "
        "explicitly with payload=Payload.PRIMS"
    )


def resolve_items_per_thread(
    *,
    scope: str,
    output: ThreadData | None,
    items_per_thread: Any,
    primitive_name: str,
) -> int:
    if items_per_thread is None:
        if output is None:
            raise TypeError(
                f"{scope}.{primitive_name} requires items_per_thread when "
                "no ThreadData output is provided"
            )
        return output.items_per_thread
    try:
        items_per_thread = _validate_items_per_thread(items_per_thread)
    except TypeError as exc:
        raise TypeError(
            f"{scope}.{primitive_name} items_per_thread must be an int"
        ) from exc
    except ValueError as exc:
        raise ValueError(
            f"{scope}.{primitive_name} items_per_thread must be positive"
        ) from exc
    if output is not None and output.items_per_thread != items_per_thread:
        raise ValueError(
            f"{scope}.{primitive_name} ThreadData.items_per_thread does "
            "not match items_per_thread"
        )
    return items_per_thread


def parse_load_args(
    args: tuple[Any, ...],
    *,
    scope: str,
    items_per_thread: Any,
    valid_items: Any,
    oob_default: Any,
) -> tuple[Any, Any, Any]:
    if len(args) == 0:
        return items_per_thread, valid_items, oob_default
    if len(args) == 1:
        if items_per_thread is None:
            return args[0], valid_items, oob_default
        if valid_items is not None:
            raise TypeError(f"{scope}.load got duplicate valid_items")
        return items_per_thread, args[0], oob_default
    if len(args) == 2:
        if items_per_thread is None:
            raise TypeError(
                f"{scope}.load got ambiguous positional args; pass "
                "items_per_thread, valid_items, and oob_default by keyword"
            )
        if valid_items is not None or oob_default is not None:
            raise TypeError(f"{scope}.load got duplicate valid_items/oob_default")
        return items_per_thread, args[0], args[1]
    raise TypeError(f"{scope}.load accepts at most two extra positional args")


def parse_store_args(
    args: tuple[Any, ...],
    *,
    scope: str,
    items_per_thread: Any,
    valid_items: Any,
) -> tuple[Any, Any]:
    if len(args) == 0:
        return items_per_thread, valid_items
    if len(args) == 1:
        if items_per_thread is None:
            return args[0], valid_items
        if valid_items is not None:
            raise TypeError(f"{scope}.store got duplicate valid_items")
        return items_per_thread, args[0]
    if len(args) == 2:
        if items_per_thread is not None or valid_items is not None:
            raise TypeError(f"{scope}.store got duplicate positional arguments")
        return args[0], args[1]
    raise TypeError(f"{scope}.store accepts at most two extra positional args")


def resolve_load_dtype(
    *,
    scope: str,
    output: ThreadData | None,
    dtype: Any,
    source: Any,
    validate_output_dtype: bool = True,
) -> Any:
    if output is None:
        return dtype if dtype is not None else infer_source_dtype(source)
    if dtype is None:
        return output.dtype if output.dtype is not None else infer_source_dtype(source)
    if output.dtype is None:
        return dtype
    if validate_output_dtype and output.dtype != dtype:
        raise TypeError(f"{scope}.load dtype does not match output.dtype")
    return output.dtype if validate_output_dtype else dtype


def resolve_store_dtype(
    *,
    scope: str,
    value: ThreadData,
    dtype: Any,
) -> Any:
    if dtype is None:
        return value.dtype
    if value.dtype is None:
        return dtype
    if value.dtype != dtype:
        raise TypeError(f"{scope}.store dtype does not match value.dtype")
    return value.dtype


def coerce_store_value(
    *,
    scope: str,
    value: Any,
    dtype: Any = None,
) -> ThreadData:
    value = _coerce_thread_payload(
        value,
        scope=scope,
        primitive_name="store",
        arg_name="value",
    )
    if not isinstance(value, ThreadData):
        raise TypeError(f"{scope}.store value must be ThreadData")
    if dtype is not None and value.dtype is None:
        value = ThreadData.from_payload(value, dtype=dtype)
    return value
