# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Private Prims array-path adapter for CUTLASS cooperative APIs."""

from __future__ import annotations

from collections.abc import Mapping
from operator import index as _index
from typing import Any

from ._dsl._launch import LAUNCH_METADATA_KEYS as _LAUNCH_METADATA_KEYS
from ._dsl._launch import (
    _reject_launch_metadata_kwargs as _reject_launch_metadata_kwargs,
)
from ._dsl._launch import bind_block_launch_kwargs as _bind_block_launch_kwargs
from ._dsl._launch import block_dim_product as _block_dim_product
from ._dsl._launch import metadata_thread_count as _metadata_thread_count
from ._dsl._launch import resolve_block_threads as _resolve_block_threads
from ._dsl._launch import resolve_threads_in_warp as _resolve_threads_in_warp
from ._dsl._load_store import (
    BLOCK_UNSUPPORTED_LOAD_STORE_ALGORITHMS as _UNSUPPORTED_LOAD_STORE_ALGORITHMS,
)
from ._dsl._load_store import coerce_store_value as _coerce_store_value
from ._dsl._load_store import merge_valid_items as _merge_valid_items
from ._dsl._load_store import normalize_algorithm as _normalize_algorithm
from ._dsl._load_store import parse_store_args as _parse_store_args
from ._dsl._load_store import (
    resolve_items_per_thread as _resolve_thread_data_items_per_thread,
)
from ._dsl._load_store import resolve_store_dtype as _resolve_store_dtype
from ._internal._thread_data import ThreadData, _validate_items_per_thread
from ._prims import is_cutlass_array_operand as _is_cutlass_array


def _resolve_items_per_thread(
    args: tuple[Any, ...],
    *,
    scope: str,
    items_per_thread: Any,
    primitive_name: str = "load",
) -> int:
    if len(args) > 1:
        raise TypeError(
            f"{scope}.{primitive_name} accepts at most one positional argument"
        )
    if args:
        if items_per_thread is not None:
            raise TypeError(f"{scope}.{primitive_name} got duplicate items_per_thread")
        items_per_thread = args[0]
    if items_per_thread is None:
        raise TypeError(f"{scope}.{primitive_name} requires items_per_thread")
    try:
        return _validate_items_per_thread(items_per_thread)
    except TypeError as exc:
        raise TypeError(
            f"{scope}.{primitive_name} items_per_thread must be an int"
        ) from exc
    except ValueError as exc:
        raise ValueError(
            f"{scope}.{primitive_name} items_per_thread must be positive"
        ) from exc


def _infer_offset_alignment(
    items_per_thread: int,
    dtype: Any,
    offset: Any,
    *,
    base_alignment: Any = None,
) -> int | None:
    """Cap vector alignment by the array base and element offset."""
    # Public cutlass.Array.align is a compile-time byte count, unlike a staged
    # element offset, so it is safe to normalize through operator.index here.
    try:
        resolved_base_alignment = _index(base_alignment)
    except TypeError:
        resolved_base_alignment = None
    base_power_of_two = (
        resolved_base_alignment & -resolved_base_alignment
        if resolved_base_alignment is not None and resolved_base_alignment > 0
        else None
    )
    zero_offset = isinstance(offset, int) and offset == 0
    unknown_dtype_alignment = base_power_of_two if zero_offset else 1
    try:
        bytes_per_item = _index(getattr(dtype, "bytes", None))
    except TypeError:
        return unknown_dtype_alignment
    if bytes_per_item <= 0:
        return unknown_dtype_alignment

    element_alignment = bytes_per_item & -bytes_per_item
    vector_bytes = items_per_thread * bytes_per_item
    vector_alignment = (
        vector_bytes
        if vector_bytes > 0 and vector_bytes & (vector_bytes - 1) == 0
        else element_alignment
    )
    if base_power_of_two is not None:
        vector_alignment = min(vector_alignment, base_power_of_two)

    # CUTLASS staged integers are intentionally not coerced with operator.index.
    if not isinstance(offset, int):
        return min(vector_alignment, element_alignment)

    byte_offset = offset * bytes_per_item
    while vector_alignment > element_alignment and byte_offset % vector_alignment:
        vector_alignment //= 2
    return vector_alignment


def _infer_array_dtype(array: Any) -> Any:
    for attr_name in ("dtype", "element_type", "_dtype"):
        dtype = getattr(array, attr_name, None)
        if dtype is not None:
            return dtype
    return None


def _array_view(
    source: Any,
    *,
    scope: str,
    primitive_name: str,
    dtype: Any,
    bounds_check: bool,
    loc: Any,
    ip: Any,
) -> Any:
    if _is_cutlass_array(source):
        if not _is_cutlass_array(source, method=primitive_name):
            raise TypeError(
                f"{scope}.{primitive_name} cutlass.Array operand must support "
                f"{primitive_name}"
            )
        if dtype is not None:
            source_dtype = _infer_array_dtype(source)
            if source_dtype is not None and dtype != source_dtype:
                if primitive_name == "store":
                    raise TypeError(
                        f"{scope}.{primitive_name} value dtype does not match "
                        "cutlass.Array dtype"
                    )
                raise TypeError(
                    f"{scope}.{primitive_name} dtype= does not match "
                    "cutlass.Array dtype"
                )
        if bounds_check:
            raise TypeError(
                f"{scope}.{primitive_name} bounds_check= is only accepted "
                "when wrapping a source with cutlass.make_array_view"
            )
        return source

    import cutlass

    # Let CUTLASS derive the array element type from the wrapped source. The dtype
    # argument is used for coop-side alignment inference.
    return cutlass.make_array_view(
        source,
        bounds_check=bounds_check,
        loc=loc,
        ip=ip,
    )


def _linear_thread_and_block_threads() -> tuple[Any, Any]:
    import cutlass

    tx, ty, tz = cutlass.cute.arch.thread_idx()
    bx, by, bz = cutlass.cute.arch.block_dim()
    linear_tid = tx + ty * bx + tz * bx * by
    block_threads = bx * by * bz
    return linear_tid, block_threads


def _resolve_explicit_group_shape(
    *,
    scope: str,
    primitive_name: str,
    threads_per_block: Any,
    threads_in_warp: Any,
    dim: Any,
) -> tuple[Any, Any]:
    if scope.endswith((".warp", "._warp")):
        if threads_per_block is not None or dim is not None:
            raise TypeError(
                f"{scope}.{primitive_name} does not accept threads_per_block or dim"
            )
        resolved_threads_in_warp = (
            32
            if threads_in_warp is None
            else _resolve_threads_in_warp(scope, primitive_name, threads_in_warp)
        )
        return None, resolved_threads_in_warp

    if threads_in_warp is not None:
        raise TypeError(f"{scope}.{primitive_name} does not accept threads_in_warp")

    resolved_threads_per_block = _resolve_block_threads(
        scope,
        primitive_name,
        threads_per_block=threads_per_block,
        dim=dim,
    )
    if resolved_threads_per_block is not None:
        resolved_threads_per_block = _block_dim_product(resolved_threads_per_block)
    return resolved_threads_per_block, None


def _normalize_launch_aliases(
    kwargs: dict[str, Any],
    *,
    scope: str,
    primitive_name: str,
    threads_per_block: Any,
    dim: Any,
) -> tuple[dict[str, Any], Any, Any]:
    if scope.endswith((".warp", "._warp")):
        _reject_launch_metadata_kwargs(scope, primitive_name, kwargs)
        return kwargs, threads_per_block, dim

    kwargs = _bind_block_launch_kwargs(
        scope,
        primitive_name,
        kwargs,
        threads_per_block=threads_per_block,
        dim=dim,
    )
    metadata_threads = None
    for name in _LAUNCH_METADATA_KEYS:
        metadata = kwargs.pop(name, None)
        if isinstance(metadata, Mapping):
            metadata_threads = _metadata_thread_count(metadata)
    if metadata_threads is not None:
        threads_per_block = metadata_threads
        dim = None
    else:
        threads_per_block = None
        dim = None
    return kwargs, threads_per_block, dim


def _flat_indices(
    *,
    scope: str,
    items_per_thread: int,
    offset: Any,
    algorithm: str,
    threads_per_block: Any,
    threads_in_warp: Any,
) -> list[Any]:
    linear_tid, block_threads = _linear_thread_and_block_threads()
    if scope.endswith((".warp", "._warp")):
        if threads_in_warp is None:
            threads_in_warp = 32
        group_base = (linear_tid // threads_in_warp) * threads_in_warp
        lane = linear_tid - group_base
        tile_base = offset + group_base * items_per_thread
        if algorithm == "striped":
            return [
                tile_base + lane + threads_in_warp * item_idx
                for item_idx in range(items_per_thread)
            ]
        if algorithm == "direct":
            return [
                tile_base + lane * items_per_thread + item_idx
                for item_idx in range(items_per_thread)
            ]
    else:
        if algorithm == "striped":
            if threads_per_block is None:
                threads_per_block = block_threads
            return [
                offset + linear_tid + threads_per_block * item_idx
                for item_idx in range(items_per_thread)
            ]
        if algorithm == "direct":
            return [
                offset + linear_tid * items_per_thread + item_idx
                for item_idx in range(items_per_thread)
            ]
    raise AssertionError(f"unhandled load algorithm {algorithm!r}")


def _scalar_gather_load(
    array: Any,
    *,
    indices: list[Any],
    offset: Any,
    valid_items: Any,
    oob_default: Any,
    dtype: Any,
    is_volatile: bool,
    is_nontemporal: bool,
    is_invariant: bool,
    is_invariant_group: bool,
    ordering: Any,
    syncscope: Any,
    loc: Any,
    ip: Any,
) -> Any:
    cutlass = None
    if valid_items is not None:
        import cutlass as _cutlass

        cutlass = _cutlass

    values = []
    for flat_idx in indices:
        if valid_items is None:
            values.append(
                array.load(
                    flat_idx,
                    alignment=None,
                    is_volatile=is_volatile,
                    is_nontemporal=is_nontemporal,
                    is_invariant=is_invariant,
                    is_invariant_group=is_invariant_group,
                    ordering=ordering,
                    syncscope=syncscope,
                    loc=loc,
                    ip=ip,
                )
            )
            continue

        logical_idx = flat_idx - offset
        values.append(
            cutlass.if_generate(
                logical_idx < valid_items,
                lambda flat_idx=flat_idx: array.load(
                    flat_idx,
                    alignment=None,
                    is_volatile=is_volatile,
                    is_nontemporal=is_nontemporal,
                    is_invariant=is_invariant,
                    is_invariant_group=is_invariant_group,
                    ordering=ordering,
                    syncscope=syncscope,
                    loc=loc,
                    ip=ip,
                ),
                lambda oob_default=oob_default: oob_default,
                return_types=[dtype],
            )
        )
    return ThreadData.from_values(*values, dtype=dtype)


def _prims_store_value(value: Any, *, primitive_name: str) -> Any:
    values = value.values(primitive_name)
    return values[0] if len(values) == 1 else values


def _scalar_scatter_store(
    array: Any,
    value: Any,
    *,
    indices: list[Any],
    offset: Any,
    valid_items: Any,
    is_volatile: bool,
    is_nontemporal: bool,
    ordering: Any,
    syncscope: Any,
    loc: Any,
    ip: Any,
) -> None:
    cutlass = None
    if valid_items is not None:
        import cutlass as _cutlass

        cutlass = _cutlass

    for item_idx, flat_idx in enumerate(indices):
        if valid_items is None:
            array.store(
                value[item_idx],
                flat_idx,
                alignment=None,
                is_volatile=is_volatile,
                is_nontemporal=is_nontemporal,
                ordering=ordering,
                syncscope=syncscope,
                loc=loc,
                ip=ip,
            )
            continue

        logical_idx = flat_idx - offset
        cutlass.if_generate(
            logical_idx < valid_items,
            lambda flat_idx=flat_idx, item_idx=item_idx: array.store(
                value[item_idx],
                flat_idx,
                alignment=None,
                is_volatile=is_volatile,
                is_nontemporal=is_nontemporal,
                ordering=ordering,
                syncscope=syncscope,
                loc=loc,
                ip=ip,
            ),
        )


def load(
    source: Any,
    /,
    *args: Any,
    scope: str,
    items_per_thread: Any = None,
    offset: Any = 0,
    dtype: Any = None,
    alignment: Any = None,
    algorithm: Any = "direct",
    valid_items: Any = None,
    num_valid_items: Any = None,
    oob_default: Any = None,
    threads_per_block: Any = None,
    threads_in_warp: Any = None,
    dim: Any = None,
    temp_storage: Any = None,
    bounds_check: bool = False,
    is_volatile: bool = False,
    is_nontemporal: bool = False,
    is_invariant: bool = False,
    is_invariant_group: bool = False,
    ordering: Any = "not_atomic",
    syncscope: Any = None,
    loc: Any = None,
    ip: Any = None,
    **kwargs: Any,
) -> Any:
    """Load one tile through the Prims array path into per-thread registers."""
    del temp_storage
    kwargs, threads_per_block, dim = _normalize_launch_aliases(
        kwargs,
        scope=scope,
        primitive_name="load",
        threads_per_block=threads_per_block,
        dim=dim,
    )
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(
            f"{scope}.load does not accept extra keyword args; got "
            f"unexpected keyword argument(s): {unexpected}"
        )

    items_per_thread = _resolve_items_per_thread(
        args,
        scope=scope,
        items_per_thread=items_per_thread,
    )
    valid_items = _merge_valid_items(
        scope=scope,
        valid_items=valid_items,
        num_valid_items=num_valid_items,
        primitive_name="load",
    )
    if oob_default is not None and valid_items is None:
        raise TypeError(f"{scope}.load oob_default requires valid_items")
    normalized_algorithm = _normalize_algorithm(
        algorithm,
        scope=scope,
        primitive_name="load",
        unsupported_algorithms=_UNSUPPORTED_LOAD_STORE_ALGORITHMS,
    )
    threads_per_block, threads_in_warp = _resolve_explicit_group_shape(
        scope=scope,
        primitive_name="load",
        threads_per_block=threads_per_block,
        threads_in_warp=threads_in_warp,
        dim=dim,
    )

    array = _array_view(
        source,
        scope=scope,
        primitive_name="load",
        dtype=dtype,
        bounds_check=bounds_check,
        loc=loc,
        ip=ip,
    )
    resolved_dtype = dtype if dtype is not None else _infer_array_dtype(array)
    adapter_offset = 0 if offset is None else offset
    explicit_alignment = alignment is not None
    if alignment is None and valid_items is None and normalized_algorithm == "direct":
        alignment = _infer_offset_alignment(
            items_per_thread,
            resolved_dtype,
            adapter_offset,
            base_alignment=getattr(array, "align", None),
        )
    indices = _flat_indices(
        scope=scope,
        items_per_thread=items_per_thread,
        offset=adapter_offset,
        algorithm=normalized_algorithm,
        threads_per_block=threads_per_block,
        threads_in_warp=threads_in_warp,
    )
    if valid_items is not None or normalized_algorithm == "striped":
        if oob_default is None:
            if valid_items is not None:
                raise NotImplementedError(
                    f"{scope}.load currently requires oob_default when "
                    "valid_items is provided"
                )
        if explicit_alignment:
            raise NotImplementedError(
                f"{scope}.load explicit alignment is not implemented for "
                f"{normalized_algorithm} cutlass.Array loads"
            )
        if resolved_dtype is None:
            raise TypeError(
                f"{scope}.load {normalized_algorithm} cutlass.Array loads require "
                "dtype or a cutlass.Array with dtype"
            )
        return _scalar_gather_load(
            array,
            indices=indices,
            offset=adapter_offset,
            valid_items=valid_items,
            oob_default=oob_default,
            dtype=resolved_dtype,
            is_volatile=is_volatile,
            is_nontemporal=is_nontemporal,
            is_invariant=is_invariant,
            is_invariant_group=is_invariant_group,
            ordering=ordering,
            syncscope=syncscope,
            loc=loc,
            ip=ip,
        )

    return array.load(
        indices[0],
        items_per_thread,
        alignment=alignment,
        is_volatile=is_volatile,
        is_nontemporal=is_nontemporal,
        is_invariant=is_invariant,
        is_invariant_group=is_invariant_group,
        ordering=ordering,
        syncscope=syncscope,
        loc=loc,
        ip=ip,
    )


def store(
    destination: Any,
    value: Any,
    /,
    *args: Any,
    scope: str,
    items_per_thread: Any = None,
    offset: Any = 0,
    valid_items: Any = None,
    num_valid_items: Any = None,
    algorithm: Any = "direct",
    dtype: Any = None,
    alignment: Any = None,
    threads_per_block: Any = None,
    threads_in_warp: Any = None,
    dim: Any = None,
    temp_storage: Any = None,
    bounds_check: bool = False,
    is_volatile: bool = False,
    is_nontemporal: bool = False,
    ordering: Any = "not_atomic",
    syncscope: Any = None,
    loc: Any = None,
    ip: Any = None,
    **kwargs: Any,
) -> None:
    """Store a per-thread payload through the Prims array path."""
    del temp_storage
    kwargs, threads_per_block, dim = _normalize_launch_aliases(
        kwargs,
        scope=scope,
        primitive_name="store",
        threads_per_block=threads_per_block,
        dim=dim,
    )
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(
            f"{scope}.store does not accept extra keyword args; got "
            f"unexpected keyword argument(s): {unexpected}"
        )

    value = _coerce_store_value(
        scope=scope,
        value=value,
        dtype=dtype,
    )
    resolved_dtype = _resolve_store_dtype(scope=scope, value=value, dtype=dtype)
    items_per_thread, valid_items = _parse_store_args(
        args,
        scope=scope,
        items_per_thread=items_per_thread,
        valid_items=valid_items,
    )
    valid_items = _merge_valid_items(
        scope=scope,
        valid_items=valid_items,
        num_valid_items=num_valid_items,
        primitive_name="store",
    )
    resolved_items = _resolve_thread_data_items_per_thread(
        scope=scope,
        output=value,
        items_per_thread=items_per_thread,
        primitive_name="store",
    )
    normalized_algorithm = _normalize_algorithm(
        algorithm,
        scope=scope,
        primitive_name="store",
        unsupported_algorithms=_UNSUPPORTED_LOAD_STORE_ALGORITHMS,
    )
    threads_per_block, threads_in_warp = _resolve_explicit_group_shape(
        scope=scope,
        primitive_name="store",
        threads_per_block=threads_per_block,
        threads_in_warp=threads_in_warp,
        dim=dim,
    )
    if value.dtype is None:
        value.dtype = resolved_dtype

    array = _array_view(
        destination,
        scope=scope,
        primitive_name="store",
        dtype=resolved_dtype,
        bounds_check=bounds_check,
        loc=loc,
        ip=ip,
    )
    adapter_offset = 0 if offset is None else offset
    indices = _flat_indices(
        scope=scope,
        items_per_thread=resolved_items,
        offset=adapter_offset,
        algorithm=normalized_algorithm,
        threads_per_block=threads_per_block,
        threads_in_warp=threads_in_warp,
    )
    explicit_alignment = alignment is not None
    use_vector_store = (
        valid_items is None
        and normalized_algorithm == "direct"
        and not scope.endswith((".warp", "._warp"))
    )
    if alignment is None and use_vector_store:
        alignment = _infer_offset_alignment(
            resolved_items,
            value.dtype,
            adapter_offset,
            base_alignment=getattr(array, "align", None),
        )

    if use_vector_store:
        array.store(
            _prims_store_value(value, primitive_name="store"),
            indices[0],
            resolved_items,
            alignment=alignment,
            is_volatile=is_volatile,
            is_nontemporal=is_nontemporal,
            ordering=ordering,
            syncscope=syncscope,
            loc=loc,
            ip=ip,
        )
        return

    if explicit_alignment:
        raise NotImplementedError(
            f"{scope}.store explicit alignment is not implemented for "
            f"{normalized_algorithm} cutlass.Array stores"
        )
    _scalar_scatter_store(
        array,
        value,
        indices=indices,
        offset=adapter_offset,
        valid_items=valid_items,
        is_volatile=is_volatile,
        is_nontemporal=is_nontemporal,
        ordering=ordering,
        syncscope=syncscope,
        loc=loc,
        ip=ip,
    )
