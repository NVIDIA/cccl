# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from typing import Any

from cuda.coop._core.warp import make_warp_load_spec, make_warp_store_spec

from .._launch import resolve_threads_in_warp as _resolve_threads_in_warp
from .._load_store import (
    WARP_UNSUPPORTED_LOAD_STORE_ALGORITHMS,
    ScopedLoadStoreRoute,
    classify_scoped_load_store_route,
)
from .._load_store import coerce_store_value as _coerce_store_value
from .._load_store import merge_valid_items as _merge_valid_items
from .._load_store import normalize_algorithm as _normalize_algorithm
from .._load_store import parse_load_args as _parse_load_args
from .._load_store import parse_store_args as _parse_store_args
from .._load_store import reject_cutlass_array_operand as _reject_cutlass_array_operand
from .._load_store import resolve_items_per_thread as _resolve_items_per_thread
from .._load_store import resolve_load_dtype as _resolve_load_dtype
from .._load_store import resolve_store_dtype as _resolve_store_dtype
from .._load_store import validate_payload_selector as _validate_payload_selector
from .._scope import WARP_SCOPE as _SCOPE
from .._scope import validate_no_extra_warp_args as validate_no_extra_args
from .._thread_data import ThreadData


def _linear_thread_id() -> Any:
    import cutlass.cute as cute

    tx, ty, tz = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_dim()
    return tx + ty * bx + tz * bx * by


def _flat_item_index(
    *,
    item_idx: int,
    items_per_thread: int,
    threads_in_warp: int,
    algorithm: str,
    offset: Any,
) -> Any:
    linear_tid = _linear_thread_id()
    group_base = (linear_tid // threads_in_warp) * threads_in_warp
    lane = linear_tid - group_base
    tile_base = offset + group_base * items_per_thread
    if algorithm == "striped":
        return tile_base + lane + threads_in_warp * item_idx
    if algorithm == "direct":
        return tile_base + lane * items_per_thread + item_idx
    raise AssertionError(f"unhandled load/store algorithm {algorithm!r}")


def _physical_warp_valid_items(
    valid_items: Any,
    *,
    items_per_thread: int,
    exact_block_dim: tuple[int, int, int],
) -> Any:
    """Translate scoped block-tile validity to one physical-warp tile."""

    if valid_items is None:
        return None
    block_threads = exact_block_dim[0] * exact_block_dim[1] * exact_block_dim[2]
    if block_threads == 32:
        return valid_items

    import cutlass
    from cutlass.base_dsl.typing import Int32

    tile_items = 32 * items_per_thread
    warp_rank = _linear_thread_id() // 32
    remaining = Int32(valid_items) - warp_rank * tile_items
    return cutlass.if_generate(
        remaining <= 0,
        lambda: Int32(0),
        lambda: cutlass.if_generate(
            remaining < tile_items,
            lambda: remaining,
            lambda: Int32(tile_items),
            return_types=[Int32],
        ),
        return_types=[Int32],
    )


def load(
    source: Any,
    output: ThreadData | None = None,
    /,
    *args: Any,
    items_per_thread: int | None = None,
    valid_items: Any = None,
    num_valid_items: Any = None,
    oob_default: Any = None,
    offset: Any = None,
    algorithm: Any = "direct",
    dtype: Any = None,
    threads_in_warp: int = 32,
    temp_storage: Any = None,
    payload: Any = None,
    **kwargs: Any,
) -> ThreadData:
    """Load one logical warp tile from a CuTe tensor into per-lane ThreadData.

    ``direct`` layout assigns consecutive items to each lane. ``striped`` layout
    assigns item ``i`` from lane-contiguous stripe ``i``. Multiple logical warps
    in the same block read separate contiguous tiles from ``source``.
    A complete physical warp and contiguous pointer-backed tensor use the
    canonical CUB provider. Logical or unproven routes retain this explicit
    indexing adapter. ``vectorize`` aliases direct; ``transpose`` is rejected.
    """
    del temp_storage
    _validate_payload_selector(
        payload,
        scope=_SCOPE,
        primitive_name="load",
    )
    launch_kwargs = dict(kwargs)
    validate_no_extra_args(
        "load",
        args=(),
        kwargs=kwargs,
        expected="does not accept extra keyword args",
    )
    _reject_cutlass_array_operand(
        source,
        scope=_SCOPE,
        primitive_name="load",
        operand_name="source",
    )
    items_per_thread, valid_items, oob_default = _parse_load_args(
        args,
        scope=_SCOPE,
        items_per_thread=items_per_thread,
        valid_items=valid_items,
        oob_default=oob_default,
    )
    valid_items = _merge_valid_items(
        scope=_SCOPE,
        valid_items=valid_items,
        num_valid_items=num_valid_items,
        primitive_name="load",
    )
    if oob_default is not None and valid_items is None:
        raise TypeError(f"{_SCOPE}.load oob_default requires valid_items")
    if output is not None and not isinstance(output, ThreadData):
        raise TypeError(f"{_SCOPE}.load output must be ThreadData or None")
    out_dtype = _resolve_load_dtype(
        scope=_SCOPE,
        output=output,
        dtype=dtype,
        source=source,
    )
    resolved_items = _resolve_items_per_thread(
        scope=_SCOPE,
        output=output,
        items_per_thread=items_per_thread,
        primitive_name="load",
    )
    resolved_threads = _resolve_threads_in_warp(
        _SCOPE,
        "load",
        threads_in_warp,
    )
    if valid_items is not None:
        if out_dtype is None:
            raise TypeError(
                f"{_SCOPE}.load valid_items requires dtype or a source tensor "
                "with element_type"
            )
    result = (
        output if output is not None else ThreadData(resolved_items, dtype=out_dtype)
    )
    if result.dtype is None:
        result.dtype = out_dtype

    normalized_algorithm = _normalize_algorithm(
        algorithm,
        scope=_SCOPE,
        primitive_name="load",
        unsupported_algorithms=WARP_UNSUPPORTED_LOAD_STORE_ALGORITHMS,
    )
    route = classify_scoped_load_store_route(
        source,
        scope=_SCOPE,
        primitive_name="load",
        launch_kwargs=launch_kwargs,
        dtype=out_dtype,
        items_per_thread=resolved_items,
        threads_in_warp=resolved_threads,
    )
    if route.route is ScopedLoadStoreRoute.CANONICAL_CUB:
        from ... import _group_load_store as _group_frontend
        from ..._thread_group import this_warp

        assert route.exact_block_dim is not None
        group_valid_items = _physical_warp_valid_items(
            valid_items,
            items_per_thread=resolved_items,
            exact_block_dim=route.exact_block_dim,
        )
        return _group_frontend._load(
            this_warp(),
            source,
            result,
            algorithm=normalized_algorithm,
            valid_items=group_valid_items,
            oob_default=oob_default,
            offset=offset,
            _launch_kwargs=launch_kwargs,
        )
    if valid_items is not None and oob_default is None:
        raise NotImplementedError(
            f"{_SCOPE}.load CuTe indexing payload adapter requires "
            "oob_default when valid_items is provided"
        )

    adapter_offset = 0 if offset is None else offset
    # CuTe collapses vectorize to direct and rejects transpose above. The core
    # plan passed to the indexer therefore contains only direct or striped.
    core_spec = make_warp_load_spec(
        dtype=out_dtype,
        items_per_thread=resolved_items,
        threads_in_warp=resolved_threads,
        algorithm=normalized_algorithm,
        valid_items=valid_items is not None,
        oob_default=oob_default is not None,
    )
    for item_idx in range(core_spec.items_per_thread):
        flat_idx = _flat_item_index(
            item_idx=item_idx,
            items_per_thread=core_spec.items_per_thread,
            threads_in_warp=core_spec.threads_in_warp,
            algorithm=core_spec.algorithm.value,
            offset=adapter_offset,
        )
        if valid_items is None:
            result[item_idx] = source[flat_idx]
            continue
        import cutlass

        logical_idx = flat_idx - adapter_offset
        result[item_idx] = cutlass.if_generate(
            logical_idx < valid_items,
            lambda flat_idx=flat_idx: source[flat_idx],
            lambda oob_default=oob_default: oob_default,
            return_types=[out_dtype],
        )
    return result


def store(
    destination: Any,
    value: Any,
    /,
    *args: Any,
    items_per_thread: int | None = None,
    valid_items: Any = None,
    num_valid_items: Any = None,
    algorithm: Any = "direct",
    offset: Any = None,
    dtype: Any = None,
    threads_in_warp: int = 32,
    temp_storage: Any = None,
    payload: Any = None,
    **kwargs: Any,
) -> None:
    """Store per-lane ThreadData into one logical warp tile of a CuTe tensor.

    The layout and logical-warp tiling match :func:`load`: ``direct`` writes
    consecutive items from each lane, ``striped`` writes item-wise lane stripes,
    and each logical warp in the block writes a separate contiguous tile.
    """
    del temp_storage
    _validate_payload_selector(
        payload,
        scope=_SCOPE,
        primitive_name="store",
    )
    launch_kwargs = dict(kwargs)
    validate_no_extra_args(
        "store",
        args=(),
        kwargs=kwargs,
        expected="does not accept extra keyword args",
    )
    _reject_cutlass_array_operand(
        destination,
        scope=_SCOPE,
        primitive_name="store",
        operand_name="destination",
    )
    value = _coerce_store_value(
        scope=_SCOPE,
        value=value,
        dtype=dtype,
    )
    resolved_dtype = _resolve_store_dtype(scope=_SCOPE, value=value, dtype=dtype)

    items_per_thread, valid_items = _parse_store_args(
        args,
        scope=_SCOPE,
        items_per_thread=items_per_thread,
        valid_items=valid_items,
    )
    valid_items = _merge_valid_items(
        scope=_SCOPE,
        valid_items=valid_items,
        num_valid_items=num_valid_items,
        primitive_name="store",
    )
    resolved_items = _resolve_items_per_thread(
        scope=_SCOPE,
        output=value,
        items_per_thread=items_per_thread,
        primitive_name="store",
    )
    resolved_threads = _resolve_threads_in_warp(
        _SCOPE,
        "store",
        threads_in_warp,
    )
    normalized_algorithm = _normalize_algorithm(
        algorithm,
        scope=_SCOPE,
        primitive_name="store",
        unsupported_algorithms=WARP_UNSUPPORTED_LOAD_STORE_ALGORITHMS,
    )
    if value.dtype is None:
        value.dtype = resolved_dtype
    route = classify_scoped_load_store_route(
        destination,
        scope=_SCOPE,
        primitive_name="store",
        launch_kwargs=launch_kwargs,
        dtype=resolved_dtype,
        items_per_thread=resolved_items,
        threads_in_warp=resolved_threads,
    )
    if route.route is ScopedLoadStoreRoute.CANONICAL_CUB:
        from ... import _group_load_store as _group_frontend
        from ..._thread_group import this_warp

        assert route.exact_block_dim is not None
        group_valid_items = _physical_warp_valid_items(
            valid_items,
            items_per_thread=resolved_items,
            exact_block_dim=route.exact_block_dim,
        )
        _group_frontend._store(
            this_warp(),
            destination,
            value,
            algorithm=normalized_algorithm,
            valid_items=group_valid_items,
            offset=offset,
            _launch_kwargs=launch_kwargs,
        )
        return

    adapter_offset = 0 if offset is None else offset
    # CuTe collapses vectorize to direct and rejects transpose above. The core
    # plan passed to the indexer therefore contains only direct or striped.
    core_spec = make_warp_store_spec(
        dtype=resolved_dtype,
        items_per_thread=resolved_items,
        threads_in_warp=resolved_threads,
        algorithm=normalized_algorithm,
        valid_items=valid_items is not None,
    )

    for item_idx in range(core_spec.items_per_thread):
        flat_idx = _flat_item_index(
            item_idx=item_idx,
            items_per_thread=core_spec.items_per_thread,
            threads_in_warp=core_spec.threads_in_warp,
            algorithm=core_spec.algorithm.value,
            offset=adapter_offset,
        )
        if valid_items is None:
            destination[flat_idx] = value[item_idx]
            continue
        import cutlass

        logical_idx = flat_idx - adapter_offset
        cutlass.if_generate(
            logical_idx < valid_items,
            lambda flat_idx=flat_idx, item_idx=item_idx: destination.__setitem__(
                flat_idx, value[item_idx]
            ),
        )
