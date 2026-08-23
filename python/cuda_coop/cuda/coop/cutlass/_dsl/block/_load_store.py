# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from typing import Any

from cuda.coop._core.block import make_block_load_store_semantics

from .._launch import bind_block_launch_kwargs
from .._load_store import (
    BLOCK_UNSUPPORTED_LOAD_STORE_ALGORITHMS,
    LOAD_STORE_ALGORITHM_ALIASES,
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
from .._scope import BLOCK_SCOPE as _SCOPE
from .._scope import validate_no_extra_block_args as validate_no_extra_args
from .._thread_data import ThreadData


def _linear_thread_and_block_threads() -> tuple[Any, Any]:
    import cutlass.cute as cute

    tx, ty, tz = cute.arch.thread_idx()
    bx, by, bz = cute.arch.block_dim()
    linear_tid = tx + ty * bx + tz * bx * by
    block_threads = bx * by * bz
    return linear_tid, block_threads


def _flat_item_index(
    *,
    item_idx: int,
    items_per_thread: int,
    algorithm: str,
    offset: Any,
) -> Any:
    linear_tid, block_threads = _linear_thread_and_block_threads()
    if algorithm == "striped":
        return offset + linear_tid + block_threads * item_idx
    if algorithm == "direct":
        return offset + linear_tid * items_per_thread + item_idx
    raise AssertionError(f"unhandled load/store algorithm {algorithm!r}")


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
    threads_per_block: Any = None,
    dim: Any = None,
    temp_storage: Any = None,
    payload: Any = None,
    **kwargs: Any,
) -> ThreadData:
    """Load a block tile from a CuTe tensor into per-thread ThreadData.

    Contiguous pointer-backed tensors with exact launch facts use the canonical
    CUB provider. The indexing payload adapter requires ``oob_default`` for a
    partial tile so inactive items still trace a typed value.
    """
    _validate_payload_selector(
        payload,
        scope=_SCOPE,
        primitive_name="load",
    )
    kwargs = bind_block_launch_kwargs(
        _SCOPE,
        "load",
        kwargs,
        threads_per_block=threads_per_block,
        dim=dim,
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
    resolved_items = _resolve_items_per_thread(
        scope=_SCOPE,
        output=output,
        items_per_thread=items_per_thread,
        primitive_name="load",
    )
    out_dtype = _resolve_load_dtype(
        scope=_SCOPE,
        output=output,
        dtype=dtype,
        source=source,
        validate_output_dtype=False,
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
        unsupported_algorithms=BLOCK_UNSUPPORTED_LOAD_STORE_ALGORITHMS,
        error_algorithm_names=frozenset(
            set(LOAD_STORE_ALGORITHM_ALIASES) | BLOCK_UNSUPPORTED_LOAD_STORE_ALGORITHMS
        ),
    )
    route = classify_scoped_load_store_route(
        source,
        scope=_SCOPE,
        primitive_name="load",
        launch_kwargs=launch_kwargs,
        dtype=out_dtype,
        items_per_thread=resolved_items,
    )
    if route.route is ScopedLoadStoreRoute.CANONICAL_CUB:
        from ... import _group_load_store as _group_frontend
        from ..._thread_group import this_block

        return _group_frontend._load(
            this_block(),
            source,
            result,
            algorithm=normalized_algorithm,
            valid_items=valid_items,
            oob_default=oob_default,
            offset=offset,
            temp_storage=temp_storage,
            _launch_kwargs=launch_kwargs,
        )
    if getattr(temp_storage, "is_deferred", False):
        raise NotImplementedError(
            f"{_SCOPE}.load deferred TempStorage requires the canonical CUB "
            "BlockLoad route with a contiguous source"
        )
    if valid_items is not None and oob_default is None:
        raise NotImplementedError(
            f"{_SCOPE}.load CuTe indexing payload adapter requires "
            "oob_default when valid_items is provided"
        )

    adapter_offset = 0 if offset is None else offset
    # This call is a parity contract with _core. CuTe retains ownership of the
    # runtime block indexing performed below.
    core_semantics = make_block_load_store_semantics(
        kind="load",
        dtype=out_dtype,
        items_per_thread=resolved_items,
        algorithm=normalized_algorithm,
        valid_items=valid_items is not None,
        oob_default=oob_default is not None,
    )
    for item_idx in range(core_semantics.items_per_thread):
        flat_idx = _flat_item_index(
            item_idx=item_idx,
            items_per_thread=core_semantics.items_per_thread,
            algorithm=core_semantics.algorithm.value,
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
    threads_per_block: Any = None,
    dim: Any = None,
    temp_storage: Any = None,
    payload: Any = None,
    **kwargs: Any,
) -> None:
    """Store through canonical CUB when proven, otherwise through indexing."""
    _validate_payload_selector(
        payload,
        scope=_SCOPE,
        primitive_name="store",
    )
    kwargs = bind_block_launch_kwargs(
        _SCOPE,
        "store",
        kwargs,
        threads_per_block=threads_per_block,
        dim=dim,
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
    normalized_algorithm = _normalize_algorithm(
        algorithm,
        scope=_SCOPE,
        primitive_name="store",
        unsupported_algorithms=BLOCK_UNSUPPORTED_LOAD_STORE_ALGORITHMS,
        error_algorithm_names=frozenset(
            set(LOAD_STORE_ALGORITHM_ALIASES) | BLOCK_UNSUPPORTED_LOAD_STORE_ALGORITHMS
        ),
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
    )
    if route.route is ScopedLoadStoreRoute.CANONICAL_CUB:
        from ... import _group_load_store as _group_frontend
        from ..._thread_group import this_block

        _group_frontend._store(
            this_block(),
            destination,
            value,
            algorithm=normalized_algorithm,
            valid_items=valid_items,
            offset=offset,
            temp_storage=temp_storage,
            _launch_kwargs=launch_kwargs,
        )
        return
    if getattr(temp_storage, "is_deferred", False):
        raise NotImplementedError(
            f"{_SCOPE}.store deferred TempStorage requires the canonical CUB "
            "BlockStore route with a contiguous destination"
        )

    adapter_offset = 0 if offset is None else offset

    # This call is a parity contract with _core. CuTe retains ownership of the
    # runtime block indexing performed below.
    core_semantics = make_block_load_store_semantics(
        kind="store",
        dtype=resolved_dtype,
        items_per_thread=resolved_items,
        algorithm=normalized_algorithm,
        valid_items=valid_items is not None,
    )
    for item_idx in range(core_semantics.items_per_thread):
        flat_idx = _flat_item_index(
            item_idx=item_idx,
            items_per_thread=core_semantics.items_per_thread,
            algorithm=core_semantics.algorithm.value,
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
