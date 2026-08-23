# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Secondary deferred warp-primitive adapters for CuTe DSL kernels.

Private ``coop._warp.*`` calls are retained for internal compatibility. This
module preserves the scoped ``make_*`` factory shape: each function
validates and binds logical-warp configuration, then returns a lightweight
callable that invokes a single-phase primitive such as :func:`sum`,
:func:`exclusive_scan`, or :func:`merge_sort_pairs` from inside a CuTe kernel
trace. The factories stay under ``cuda.coop.cutlass._warp`` and are not part
of the public group-first API.
"""

from __future__ import annotations

from typing import Any, Callable

from .._factory import (
    _bind_if_not_none,
    _reject_prims_specific_load_store_factory_kwargs,
)
from .._factory import (
    _make_factory as _make_scoped_factory,
)
from .._factory import (
    _normalize_pair_dtype_aliases as _normalize_pair_dtype_aliases_for_scope,
)
from .._factory import (
    _reject_if_supplied as _reject_if_supplied_for_scope,
)
from .._factory import (
    _reject_methods as _reject_methods_for_scope,
)
from .._factory import (
    _resolve_static_items_per_thread as _resolve_static_items_per_thread_for_scope,
)
from .._launch import resolve_threads_in_warp
from .._load_store import validate_payload_selector as _validate_payload_selector
from .._scope import WARP_SCOPE as _SCOPE
from ._load_store import load, store
from ._reduce import max, min, reduce, sum

_VALID_ITEMS_OVERRIDABLE_KWARGS = ("valid_items",)
_SCAN_OVERRIDABLE_KWARGS = ("valid_items", "warp_aggregate")
_LOAD_STORE_VALID_OVERRIDABLE_KWARGS = (
    "valid_items",
    "num_valid_items",
    "oob_default",
)
_LOAD_STORE_VALID_OVERRIDE_ALIASES = (
    ("valid_items", ("num_valid_items",)),
    ("num_valid_items", ("valid_items",)),
)
_MERGE_SORT_PARTIAL_OVERRIDABLE_KWARGS = ("valid_items", "oob_default")


def _make_factory(
    factory_name: str,
    primitive: Callable[..., Any],
    kwargs: dict[str, Any],
    *,
    overridable_kwargs: tuple[str, ...] = (),
    override_aliases: tuple[tuple[str, tuple[str, ...]], ...] = (),
):
    return _make_scoped_factory(
        _SCOPE,
        factory_name=factory_name,
        primitive=primitive,
        kwargs=kwargs,
        overridable_kwargs=overridable_kwargs,
        override_aliases=override_aliases,
    )


def _reject_if_supplied(factory_name: str, name: str, value: Any) -> None:
    _reject_if_supplied_for_scope(_SCOPE, factory_name, name, value)


def _reject_methods(factory_name: str, kwargs: dict[str, Any]) -> None:
    _reject_methods_for_scope(_SCOPE, factory_name, kwargs)


def _normalize_valid_items_aliases(
    factory_name: str,
    *,
    valid_items: Any,
    num_valid_items: Any,
) -> Any:
    if valid_items is not None and num_valid_items is not None:
        raise TypeError(
            f"{_SCOPE}.{factory_name} got both valid_items and num_valid_items"
        )
    return valid_items if valid_items is not None else num_valid_items


def _normalize_pair_dtype_aliases(
    factory_name: str,
    keys: Any,
    values: Any,
    key_dtype: Any,
    value_dtype: Any,
) -> tuple[Any, Any]:
    return _normalize_pair_dtype_aliases_for_scope(
        _SCOPE,
        factory_name,
        keys,
        values,
        key_dtype,
        value_dtype,
    )


def _resolve_threads_in_warp(factory_name: str, threads_in_warp: Any) -> int:
    return resolve_threads_in_warp(_SCOPE, factory_name, threads_in_warp)


def make_load(
    dtype: Any = None,
    items_per_thread: int = 1,
    threads_in_warp: int = 32,
    algorithm: Any = "direct",
    num_valid_items: Any = None,
    oob_default: Any = None,
    *,
    payload: Any = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Return a deferred warp load callable.

    The callable binds layout, item-count, logical-warp options, and optional
    dtype, then forwards each kernel-trace call to :func:`load`.
    """
    _validate_payload_selector(
        payload,
        scope=_SCOPE,
        primitive_name="make_load",
    )
    _reject_methods("make_load", kwargs)
    _reject_prims_specific_load_store_factory_kwargs(_SCOPE, "make_load", kwargs)
    valid_items = kwargs.pop("valid_items", None)
    bound = dict(kwargs)
    bound.update(
        items_per_thread=_resolve_static_items_per_thread_for_scope(
            _SCOPE,
            "make_load",
            items_per_thread,
        ),
        threads_in_warp=_resolve_threads_in_warp("make_load", threads_in_warp),
        algorithm=algorithm,
    )
    _bind_if_not_none(bound, "dtype", dtype)
    valid_items = _normalize_valid_items_aliases(
        "make_load",
        valid_items=valid_items,
        num_valid_items=num_valid_items,
    )
    _bind_if_not_none(bound, "valid_items", valid_items)
    _bind_if_not_none(bound, "oob_default", oob_default)
    return _make_factory(
        "make_load",
        load,
        bound,
        overridable_kwargs=_LOAD_STORE_VALID_OVERRIDABLE_KWARGS,
        override_aliases=_LOAD_STORE_VALID_OVERRIDE_ALIASES,
    )


def make_store(
    dtype: Any = None,
    items_per_thread: int = 1,
    threads_in_warp: int = 32,
    algorithm: Any = "direct",
    num_valid_items: Any = None,
    *,
    payload: Any = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Return a deferred warp store callable.

    The callable binds layout, item-count, logical-warp options, and optional
    dtype, then forwards each kernel-trace call to :func:`store`.
    """
    _validate_payload_selector(
        payload,
        scope=_SCOPE,
        primitive_name="make_store",
    )
    _reject_methods("make_store", kwargs)
    _reject_prims_specific_load_store_factory_kwargs(_SCOPE, "make_store", kwargs)
    valid_items = kwargs.pop("valid_items", None)
    bound = dict(kwargs)
    bound.update(
        items_per_thread=_resolve_static_items_per_thread_for_scope(
            _SCOPE,
            "make_store",
            items_per_thread,
        ),
        threads_in_warp=_resolve_threads_in_warp("make_store", threads_in_warp),
        algorithm=algorithm,
    )
    _bind_if_not_none(bound, "dtype", dtype)
    valid_items = _normalize_valid_items_aliases(
        "make_store",
        valid_items=valid_items,
        num_valid_items=num_valid_items,
    )
    _bind_if_not_none(bound, "valid_items", valid_items)
    return _make_factory(
        "make_store",
        store,
        bound,
        overridable_kwargs=_LOAD_STORE_VALID_OVERRIDABLE_KWARGS,
        override_aliases=_LOAD_STORE_VALID_OVERRIDE_ALIASES,
    )


def make_reduce(
    dtype: Any,
    binary_op: Any = None,
    threads_in_warp: int = 32,
    valid_items: Any = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Return a deferred warp reduce callable.

    The callable binds the reduction operator and logical-warp width, then
    forwards each scalar value to :func:`reduce`.
    """
    del dtype
    _reject_methods("make_reduce", kwargs)
    bound = dict(kwargs)
    bound["threads_in_warp"] = _resolve_threads_in_warp(
        "make_reduce",
        threads_in_warp,
    )
    _bind_if_not_none(bound, "binary_op", binary_op)
    _bind_if_not_none(bound, "valid_items", valid_items)
    return _make_factory(
        "make_reduce",
        reduce,
        bound,
        overridable_kwargs=_VALID_ITEMS_OVERRIDABLE_KWARGS,
    )


def make_sum(
    dtype: Any,
    threads_in_warp: int = 32,
    valid_items: Any = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Return a deferred warp sum callable.

    The callable binds logical-warp width and forwards each scalar value to
    :func:`sum`.
    """
    del dtype
    _reject_methods("make_sum", kwargs)
    bound = dict(kwargs)
    bound["threads_in_warp"] = _resolve_threads_in_warp(
        "make_sum",
        threads_in_warp,
    )
    _bind_if_not_none(bound, "valid_items", valid_items)
    return _make_factory(
        "make_sum",
        sum,
        bound,
        overridable_kwargs=_VALID_ITEMS_OVERRIDABLE_KWARGS,
    )


def make_max(
    dtype: Any,
    threads_in_warp: int = 32,
    valid_items: Any = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Return a deferred warp max callable.

    The callable binds logical-warp width and forwards each scalar value to
    :func:`max`.
    """
    del dtype
    _reject_methods("make_max", kwargs)
    bound = dict(kwargs)
    bound["threads_in_warp"] = _resolve_threads_in_warp(
        "make_max",
        threads_in_warp,
    )
    _bind_if_not_none(bound, "valid_items", valid_items)
    return _make_factory(
        "make_max",
        max,
        bound,
        overridable_kwargs=_VALID_ITEMS_OVERRIDABLE_KWARGS,
    )


def make_min(
    dtype: Any,
    threads_in_warp: int = 32,
    valid_items: Any = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Return a deferred warp min callable.

    The callable binds logical-warp width and forwards each scalar value to
    :func:`min`.
    """
    del dtype
    _reject_methods("make_min", kwargs)
    bound = dict(kwargs)
    bound["threads_in_warp"] = _resolve_threads_in_warp(
        "make_min",
        threads_in_warp,
    )
    _bind_if_not_none(bound, "valid_items", valid_items)
    return _make_factory(
        "make_min",
        min,
        bound,
        overridable_kwargs=_VALID_ITEMS_OVERRIDABLE_KWARGS,
    )
