# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Trace-time launch-fact inference for CUTLASS family lowerers."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, TypedDict

from ..._core.launch import LaunchFactOrigin, LaunchFacts, merge_launch_facts
from .._thread_data import _normalize_index_int

LAUNCH_METADATA_KEYS = ("launch_metadata", "launch_meta", "launch", "launch_config")
LAUNCH_THREAD_COUNT_KEYS = ("threads_per_block", "block", "block_dim")

LaunchDimension = int | tuple[int, ...] | list[int]


class CutlassLaunchMetadata(TypedDict, total=False):
    """Static launch facts accepted by CUTLASS group-first primitives."""

    threads_per_block: LaunchDimension
    block: LaunchDimension
    block_dim: LaunchDimension
    block_x: int
    block_y: int
    block_z: int
    block_dim_x: int
    block_dim_y: int
    block_dim_z: int
    grid: LaunchDimension
    grid_dim: LaunchDimension
    grid_x: int
    grid_y: int
    grid_z: int
    grid_dim_x: int
    grid_dim_y: int
    grid_dim_z: int
    cluster: LaunchDimension
    cluster_dim: LaunchDimension
    cluster_x: int
    cluster_y: int
    cluster_z: int
    cluster_dim_x: int
    cluster_dim_y: int
    cluster_dim_z: int
    cooperative_launch: bool
    cluster_launch: bool


def _normalize_static_int(value: Any) -> int | None:
    try:
        return _normalize_index_int(value)
    except Exception:
        return None


def resolve_block_threads(
    scope: str,
    primitive_name: str,
    *,
    threads_per_block: Any,
    dim: Any,
) -> Any:
    if threads_per_block is not None:
        threads_per_block = require_static_block_dim(
            scope,
            primitive_name,
            "threads_per_block",
            threads_per_block,
        )
    if dim is not None:
        dim = require_static_block_dim(scope, primitive_name, "dim", dim)
    if threads_per_block is not None and dim is not None:
        if normalize_block_dim(threads_per_block) != normalize_block_dim(dim):
            raise TypeError(
                f"{scope}.{primitive_name} got conflicting threads_per_block and dim"
            )
    return threads_per_block if threads_per_block is not None else dim


def is_block_dim_sequence(value: Any) -> bool:
    return isinstance(value, (tuple, list)) and len(value) > 0


def block_dim_product(value: Any) -> int | None:
    if is_block_dim_sequence(value):
        if len(value) > 3:
            return None
        threads = 1
        for dim in value:
            dim = _normalize_static_int(dim)
            if dim is None or dim <= 0:
                return None
            threads *= dim
        return threads
    value = _normalize_static_int(value)
    if value is not None and value > 0:
        return value
    return None


def normalize_block_dim(value: Any) -> tuple[int, int, int] | None:
    """Try to parse a block shape, returning ``None`` instead of raising.

    This tolerant helper is for optional backend metadata. Core descriptor
    construction uses strict ``normalize_thread_dim`` validation instead.
    """

    if isinstance(value, bool):
        return None
    if isinstance(value, (tuple, list)):
        if not value or len(value) > 3:
            return None
        dims = tuple(_normalize_static_int(dim) for dim in value)
        if any(dim is None or dim <= 0 for dim in dims):
            return None
        return (*dims, *((1,) * (3 - len(dims))))  # type: ignore[return-value]
    dim = _normalize_static_int(value)
    if dim is None or dim <= 0:
        return None
    return dim, 1, 1


def block_dim_from_launch_metadata(metadata: Any) -> tuple[int, int, int] | None:
    """Extract an exact static block shape from normalized launch metadata."""

    if not isinstance(metadata, Mapping):
        return None

    if "threads_per_block" in metadata:
        return normalize_block_dim(metadata["threads_per_block"])

    for key in ("block", "block_dim"):
        if key in metadata:
            return normalize_block_dim(metadata[key])

    for prefix in ("block", "block_dim"):
        keys = tuple(f"{prefix}_{suffix}" for suffix in ("x", "y", "z"))
        present = tuple(key in metadata for key in keys)
        if not any(present):
            continue
        if not all(present):
            return None
        return normalize_block_dim(tuple(metadata[key] for key in keys))

    return None


def _launch_metadata_dim_candidates(
    metadata: Mapping[str, Any],
    aliases: tuple[str, ...],
) -> tuple[tuple[str, tuple[int, int, int]], ...]:
    candidates: list[tuple[str, tuple[int, int, int]]] = []
    for alias in aliases:
        if alias in metadata:
            dimension = normalize_block_dim(metadata[alias])
            if dimension is None:
                raise ValueError(
                    f"launch metadata {alias!r} must be a complete static shape"
                )
            candidates.append((alias, dimension))

        axis_keys = tuple(f"{alias}_{axis}" for axis in ("x", "y", "z"))
        present = tuple(key in metadata for key in axis_keys)
        if not any(present):
            continue
        if not all(present):
            raise ValueError(
                f"launch metadata {alias!r} axis form requires x, y, and z"
            )
        dimension = normalize_block_dim(tuple(metadata[key] for key in axis_keys))
        if dimension is None:
            raise ValueError(
                f"launch metadata {alias!r} axis form must be a static shape"
            )
        candidates.append(("/".join(axis_keys), dimension))
    return tuple(candidates)


def launch_facts_from_launch_metadata(
    metadata: Mapping[str, Any],
    *,
    source: str = "launch_metadata",
) -> LaunchFacts:
    """Extract exact launch facts while reconciling all recognized aliases."""

    if not isinstance(metadata, Mapping):
        raise TypeError("launch metadata must be a mapping")

    facts: list[LaunchFacts] = []
    for field_name, aliases in (
        ("exact_block_dim", ("threads_per_block", "block", "block_dim")),
        ("exact_grid_dim", ("grid", "grid_dim")),
        ("exact_cluster_dim", ("cluster", "cluster_dim")),
    ):
        for alias, dimension in _launch_metadata_dim_candidates(metadata, aliases):
            facts.append(
                LaunchFacts(
                    **{field_name: dimension},
                    provenance=LaunchFactOrigin(
                        fact=field_name,
                        source=source,
                        detail=alias,
                    ),
                )
            )

    for field_name in ("cooperative_launch", "cluster_launch"):
        if field_name not in metadata:
            continue
        value = metadata[field_name]
        if not isinstance(value, bool):
            raise TypeError(f"launch metadata {field_name!r} must be a bool")
        facts.append(
            LaunchFacts(
                **{field_name: value},
                provenance=LaunchFactOrigin(
                    fact=field_name,
                    source=source,
                    detail=field_name,
                ),
            )
        )

    return merge_launch_facts(*facts)


def block_dim_from_nvvm_thread_attr(attr: Any) -> tuple[int, int, int] | None:
    """Parse a static block shape from an NVVM reqntid/maxntid attribute."""

    attr_text = str(attr)
    match = re.search(r":\s*([0-9,\s-]+)\s*>", attr_text)
    if not match:
        return None
    parts = tuple(
        int(part.strip()) for part in match.group(1).split(",") if part.strip()
    )
    return normalize_block_dim(parts)


def launch_facts_from_cutlass_api(
    facts: Any,
    *,
    detail: str = "cute._get_launch_facts()",
) -> LaunchFacts:
    """Normalize launch facts returned by CUTLASS's private provider hook."""

    normalized: list[LaunchFacts] = []
    for field_name in (
        "exact_block_dim",
        "exact_grid_dim",
        "exact_cluster_dim",
    ):
        raw_value = getattr(facts, field_name, None)
        if raw_value is None:
            continue
        dimension = normalize_block_dim(raw_value)
        if dimension is None:
            raise ValueError(
                f"{detail} field {field_name!r} does not contain a valid static shape"
            )
        normalized.append(
            LaunchFacts(
                **{field_name: dimension},
                provenance=LaunchFactOrigin(
                    fact=field_name,
                    source="cutlass_provider_api",
                    detail=detail,
                    verified=True,
                ),
            )
        )

    for field_name in ("cooperative_launch", "cluster_launch"):
        value = getattr(facts, field_name, None)
        if value is None:
            continue
        if not isinstance(value, bool):
            raise ValueError(f"{detail} field {field_name!r} must be a bool")
        normalized.append(
            LaunchFacts(
                **{field_name: value},
                provenance=LaunchFactOrigin(
                    fact=field_name,
                    source="cutlass_provider_api",
                    detail=detail,
                    verified=True,
                ),
            )
        )
    return merge_launch_facts(*normalized)


def block_dim_from_kernel_op(
    current_op: Any,
    *,
    allow_maxntid: bool,
) -> tuple[int, int, int] | None:
    """Read block dimensions from a kernel op and its ancestors.

    ``reqntid`` is an exact launch requirement and always takes precedence.
    ``maxntid`` is only an upper bound, so callers that require an exact shape
    must leave ``allow_maxntid`` disabled.
    """

    ops = []
    op = current_op
    while op is not None:
        ops.append(op)
        op = getattr(op, "parent_op", None) or getattr(op, "parent", None)

    attribute_groups = [("nvvm.reqntid", "reqntid")]
    if allow_maxntid:
        attribute_groups.append(("nvvm.maxntid", "maxntid"))

    for keys in attribute_groups:
        for op in ops:
            attrs = getattr(op, "attributes", None)
            if attrs is None:
                continue
            for key in keys:
                try:
                    attr = attrs[key]
                except Exception:
                    continue
                block_dim = block_dim_from_nvvm_thread_attr(attr)
                if block_dim is not None:
                    return block_dim
    return None


def launch_facts_from_kernel_op(current_op: Any) -> LaunchFacts:
    """Extract ``reqntid`` exact and ``maxntid`` upper-bound fallback facts."""

    facts: list[LaunchFacts] = []
    op = current_op
    depth = 0
    while op is not None:
        attrs = getattr(op, "attributes", None)
        if attrs is not None:
            for field_name, keys in (
                ("exact_block_dim", ("nvvm.reqntid", "reqntid")),
                ("max_block_dim", ("nvvm.maxntid", "maxntid")),
            ):
                for key in keys:
                    try:
                        attr = attrs[key]
                    except Exception:
                        continue
                    block_dim = block_dim_from_nvvm_thread_attr(attr)
                    if block_dim is None:
                        if field_name == "max_block_dim":
                            continue
                        raise ValueError(
                            f"kernel attribute {key!r} does not contain a valid "
                            "static thread shape"
                        )
                    facts.append(
                        LaunchFacts(
                            **{field_name: block_dim},
                            provenance=LaunchFactOrigin(
                                fact=field_name,
                                source="kernel_attribute",
                                detail=f"{key}@ancestor[{depth}]",
                                verified=field_name == "exact_block_dim",
                            ),
                        )
                    )
        op = getattr(op, "parent_op", None) or getattr(op, "parent", None)
        depth += 1
    return merge_launch_facts(*facts)


def current_kernel_block_dim(
    *,
    allow_maxntid: bool = False,
) -> tuple[int, int, int] | None:
    """Read exact block dimensions, or an allowed upper bound, from the kernel."""

    facts = current_kernel_launch_facts()
    if facts.exact_block_dim is not None:
        return facts.exact_block_dim
    if allow_maxntid:
        return facts.max_block_dim
    return None


def current_kernel_launch_facts() -> LaunchFacts:
    """Read launch facts from the active CUTLASS MLIR insertion point."""

    try:
        from cutlass._mlir import ir

        current_ip = ir.InsertionPoint.current
    except Exception:
        return LaunchFacts()
    if current_ip is None or current_ip.block is None:
        return LaunchFacts()
    kernel_facts = launch_facts_from_kernel_op(current_ip.block.owner)
    try:
        from cutlass import cute

        get_launch_facts = getattr(cute, "_get_launch_facts", None)
        if callable(get_launch_facts):
            try:
                provider_facts = launch_facts_from_cutlass_api(get_launch_facts())
            except Exception as exc:
                if "launch facts are unavailable" not in str(exc):
                    raise
            else:
                return merge_launch_facts(provider_facts, kernel_facts)
    except ImportError:
        pass
    return kernel_facts


def infer_launch_facts(
    kwargs: Mapping[str, Any],
    *,
    scope: str,
    primitive_name: str,
) -> LaunchFacts:
    """Merge exact call metadata with current kernel attributes."""

    present = tuple(
        key for key in LAUNCH_METADATA_KEYS if key in kwargs and kwargs[key] is not None
    )
    if len(present) > 1:
        names = ", ".join(present)
        raise TypeError(
            f"{scope}.{primitive_name} got multiple launch metadata aliases: {names}"
        )
    facts = [current_kernel_launch_facts()]
    for key in present:
        facts.append(
            launch_facts_from_launch_metadata(
                kwargs[key],
                source=f"call_argument:{key}",
            )
        )
    return merge_launch_facts(*facts)


def infer_block_dim(
    kwargs: Mapping[str, Any],
    *,
    scope: str,
    primitive_name: str,
) -> tuple[int, int, int]:
    """Infer the current static block shape from call or kernel metadata."""

    present = tuple(
        key for key in LAUNCH_METADATA_KEYS if key in kwargs and kwargs[key] is not None
    )
    facts = infer_launch_facts(
        kwargs,
        scope=scope,
        primitive_name=primitive_name,
    )
    if facts.exact_block_dim is not None:
        return facts.exact_block_dim
    if present:
        raise ValueError(
            f"{scope}.{primitive_name} requires launch metadata with a complete "
            "static block shape"
        )

    raise ValueError(
        f"{scope}.{primitive_name} requires launch metadata or kernel "
        "reqntid attributes to infer the exact block dimensions"
    )


def require_static_block_dim(
    scope: str,
    primitive_name: str,
    arg_name: str,
    value: Any,
) -> Any:
    if is_block_dim_sequence(value):
        dims = tuple(_normalize_static_int(dim) for dim in value)
        if len(dims) > 3:
            raise TypeError(
                f"{scope}.{primitive_name} {arg_name} must have at most 3 dimensions"
            )
        if not dims or any(dim is None or dim <= 0 for dim in dims):
            raise TypeError(
                f"{scope}.{primitive_name} {arg_name} must be a compile-time "
                "positive int or tuple/list of positive ints"
            )
        return dims
    value = _normalize_static_int(value)
    if value is None or value <= 0:
        raise TypeError(
            f"{scope}.{primitive_name} {arg_name} must be a compile-time "
            "positive int or tuple/list of positive ints"
        )
    return value


def resolve_threads_in_warp(
    scope: str,
    primitive_name: str,
    value: Any,
) -> int:
    value = _normalize_static_int(value)
    if value is None:
        raise TypeError(f"{scope}.{primitive_name} threads_in_warp must be an int")
    if value <= 0:
        raise ValueError(f"{scope}.{primitive_name} threads_in_warp must be positive")
    if value > 32:
        raise ValueError(f"{scope}.{primitive_name} threads_in_warp must be <= 32")
    if value & (value - 1):
        raise ValueError(
            f"{scope}.{primitive_name} threads_in_warp must be a power of two"
        )
    return value


def metadata_thread_count(metadata: Mapping[str, Any]) -> int | None:
    if "threads_per_block" in metadata:
        return block_dim_product(metadata["threads_per_block"])
    for key in LAUNCH_THREAD_COUNT_KEYS[1:]:
        if key in metadata:
            return block_dim_product(metadata[key])
    return None


def validate_metadata_thread_count(
    scope: str,
    primitive_name: str,
    metadata: Mapping[str, Any],
) -> tuple[int | None, list[str]]:
    existing_thread_keys = [
        name for name in LAUNCH_THREAD_COUNT_KEYS if name in metadata
    ]
    if len(existing_thread_keys) > 1:
        names = ", ".join(existing_thread_keys)
        raise TypeError(
            f"{scope}.{primitive_name} launch metadata has multiple "
            f"thread-count keys: {names}"
        )
    existing_count = metadata_thread_count(metadata)
    if existing_thread_keys and existing_count is None:
        names = ", ".join(existing_thread_keys)
        raise TypeError(
            f"{scope}.{primitive_name} launch metadata has "
            f"invalid thread-count key(s): {names}"
        )
    return existing_count, existing_thread_keys


def pop_launch_metadata(kwargs: dict[str, Any]) -> dict[str, Any]:
    launch_kwargs: dict[str, Any] = {}
    for key in LAUNCH_METADATA_KEYS:
        if key in kwargs:
            launch_kwargs[key] = kwargs.pop(key)
    return launch_kwargs


def bind_block_launch_metadata(
    scope: str,
    primitive_name: str,
    kwargs: dict[str, Any],
    *,
    threads_per_block: Any,
) -> None:
    present = [
        name
        for name in LAUNCH_METADATA_KEYS
        if name in kwargs and kwargs[name] is not None
    ]
    if len(present) > 1:
        names = ", ".join(present)
        raise TypeError(
            f"{scope}.{primitive_name} got multiple launch metadata aliases: {names}"
        )

    if not present:
        if threads_per_block is None:
            return
        requested_count = block_dim_product(threads_per_block)
        metadata_key = (
            "block" if is_block_dim_sequence(threads_per_block) else "threads_per_block"
        )
        metadata_value = (
            tuple(threads_per_block)
            if is_block_dim_sequence(threads_per_block)
            else threads_per_block
        )
        kwargs["launch_metadata"] = {metadata_key: metadata_value}
        return

    key = present[0]
    metadata = kwargs[key]
    if threads_per_block is None:
        if isinstance(metadata, Mapping):
            validate_metadata_thread_count(scope, primitive_name, metadata)
        return

    if not isinstance(metadata, Mapping):
        raise TypeError(
            f"{scope}.{primitive_name} launch metadata must "
            "be a mapping when threads_per_block or dim is also provided"
        )

    requested_count = block_dim_product(threads_per_block)
    metadata_key = (
        "block" if is_block_dim_sequence(threads_per_block) else "threads_per_block"
    )
    metadata_value = (
        tuple(threads_per_block)
        if is_block_dim_sequence(threads_per_block)
        else threads_per_block
    )
    existing_count, existing_thread_keys = validate_metadata_thread_count(
        scope,
        primitive_name,
        metadata,
    )
    if (
        requested_count is not None
        and existing_count is not None
        and existing_count != requested_count
    ):
        raise TypeError(
            f"{scope}.{primitive_name} got conflicting "
            "launch metadata and threads_per_block"
        )
    merged = dict(metadata)
    block_candidates = _launch_metadata_dim_candidates(
        metadata,
        ("threads_per_block", "block", "block_dim"),
    )
    exact_dimensions = tuple(
        dimension
        for candidate, dimension in block_candidates
        if "/" in candidate or is_block_dim_sequence(metadata.get(candidate))
    )
    if exact_dimensions and any(
        dimension != exact_dimensions[0] for dimension in exact_dimensions[1:]
    ):
        raise TypeError(
            f"{scope}.{primitive_name} got conflicting launch metadata block shapes"
        )
    if exact_dimensions and block_dim_product(exact_dimensions[0]) != requested_count:
        raise TypeError(
            f"{scope}.{primitive_name} got conflicting "
            "launch metadata and threads_per_block"
        )
    requested_dimension = normalize_block_dim(threads_per_block)
    assert requested_dimension is not None
    if (
        is_block_dim_sequence(threads_per_block)
        and exact_dimensions
        and exact_dimensions[0] != requested_dimension
    ):
        raise TypeError(
            f"{scope}.{primitive_name} got conflicting "
            "launch metadata and threads_per_block shapes"
        )

    precise_dimension = (
        exact_dimensions[0]
        if exact_dimensions
        else requested_dimension
        if is_block_dim_sequence(threads_per_block)
        else None
    )
    if existing_thread_keys and precise_dimension is not None:
        existing_key = existing_thread_keys[0]
        if not is_block_dim_sequence(metadata[existing_key]):
            merged[existing_key] = precise_dimension
    if block_candidates:
        kwargs[key] = merged
        return
    merged[metadata_key] = metadata_value
    kwargs[key] = merged


def bind_block_launch_kwargs(
    scope: str,
    primitive_name: str,
    kwargs: dict[str, Any],
    *,
    threads_per_block: Any = None,
    dim: Any = None,
) -> dict[str, Any]:
    bound = dict(kwargs)
    resolved_threads = resolve_block_threads(
        scope,
        primitive_name,
        threads_per_block=threads_per_block,
        dim=dim,
    )
    bind_block_launch_metadata(
        scope,
        primitive_name,
        bound,
        threads_per_block=resolved_threads,
    )
    return bound


def normalize_block_launch_kwargs(
    scope: str,
    primitive_name: str,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    bound = dict(kwargs)
    threads_per_block = bound.pop("threads_per_block", None)
    dim = bound.pop("dim", None)
    return bind_block_launch_kwargs(
        scope,
        primitive_name,
        bound,
        threads_per_block=threads_per_block,
        dim=dim,
    )
