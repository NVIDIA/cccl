# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Secondary deferred block-primitive adapters for CuTe DSL kernels.

Private ``coop._block.*`` calls are retained for internal compatibility. This
module preserves the scoped ``make_*`` factory shape: each function
validates and binds compile-time configuration, then returns a lightweight
callable that invokes a single-phase primitive such as :func:`sum`,
:func:`radix_sort_pairs`, or :func:`topk_max_keys` from inside a CuTe kernel
trace. The factories stay under ``cuda.coop.cutlass._block`` and are not part
of the public group-first API.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from .._factory import (
    _DEFAULT_SELECTOR,
    _bind_if_not_none,
    _reject_prims_specific_load_store_factory_kwargs,
    _select_positional_enum_alias,
)
from .._factory import (
    _make_factory as _make_scoped_factory,
)
from .._factory import (
    _normalize_pair_dtype_aliases as _normalize_pair_dtype_aliases_for_scope,
)
from .._factory import (
    _reject_algorithm as _reject_algorithm_for_scope,
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
from .._launch import (
    LAUNCH_METADATA_KEYS,
    bind_block_launch_kwargs,
    resolve_block_threads,
)
from .._load_store import validate_payload_selector as _validate_payload_selector
from .._scope import BLOCK_SCOPE as _SCOPE
from ._exchange import BlockExchangeType, exchange
from ._load_store import load, store
from ._reduce import reduce, sum
from ._scan import exclusive_scan, exclusive_sum, inclusive_scan, inclusive_sum, scan
from ._shuffle import shuffle

_RADIX_SORT_OVERRIDABLE_KWARGS = ("begin_bit", "end_bit", "descending")
_RADIX_SORT_DESCENDING_OVERRIDABLE_KWARGS = ("begin_bit", "end_bit")
_RADIX_RANK_OVERRIDABLE_KWARGS = (
    "begin_bit",
    "end_bit",
    "radix_bits",
    "descending",
)
_RADIX_RANK_OVERRIDE_ALIASES = (
    ("end_bit", ("radix_bits",)),
    ("radix_bits", ("end_bit",)),
)
_MERGE_SORT_OVERRIDABLE_KWARGS = ("valid_items", "oob_default")
_TOPK_OVERRIDABLE_KWARGS = ("num_valid", "begin_bit", "end_bit")
_REDUCE_VALID_OVERRIDABLE_KWARGS = ("num_valid",)
_REDUCE_VALID_OVERRIDE_ALIASES = (("valid_items", ("num_valid",)),)
_LOAD_STORE_VALID_OVERRIDABLE_KWARGS = (
    "valid_items",
    "num_valid_items",
    "oob_default",
)
_LOAD_STORE_VALID_OVERRIDE_ALIASES = (
    ("valid_items", ("num_valid_items",)),
    ("num_valid_items", ("valid_items",)),
)
_HISTOGRAM_OVERRIDABLE_KWARGS = ("bins",)
_RUN_LENGTH_OVERRIDABLE_KWARGS = ("total_decoded_size",)
_SCAN_OUTPUT_OVERRIDABLE_KWARGS = ("block_aggregate",)


def _make_factory(
    factory_name: str,
    primitive: Callable[..., Any],
    kwargs: Mapping[str, Any],
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
        launch_metadata_keys=LAUNCH_METADATA_KEYS,
    )


def _reject_if_supplied(factory_name: str, name: str, value: Any) -> None:
    _reject_if_supplied_for_scope(_SCOPE, factory_name, name, value)


def _block_kwargs(
    factory_name: str,
    *,
    threads_per_block: Any,
    dim: Any,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    return bind_block_launch_kwargs(
        _SCOPE,
        factory_name,
        threads_per_block=threads_per_block,
        dim=dim,
        kwargs=kwargs,
    )


def _reject_methods(factory_name: str, kwargs: dict[str, Any]) -> None:
    _reject_methods_for_scope(_SCOPE, factory_name, kwargs)


def _reject_algorithm(
    factory_name: str,
    algorithm: Any,
    *,
    default: Any,
) -> None:
    _reject_algorithm_for_scope(
        _SCOPE,
        factory_name,
        algorithm,
        default=default,
    )


def _select_block_exchange_type(
    factory_name: str,
    dtype: Any,
    block_exchange_type: Any,
) -> tuple[Any, Any]:
    return _select_positional_enum_alias(
        _SCOPE,
        factory_name,
        dtype,
        block_exchange_type,
        enum_type=BlockExchangeType,
        keyword_name="block_exchange_type",
    )


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


def make_load(
    dtype: Any = None,
    threads_per_block: Any = None,
    items_per_thread: int = 1,
    algorithm: Any = "direct",
    *,
    dim: Any = None,
    valid_items: Any = None,
    num_valid_items: Any = None,
    oob_default: Any = None,
    payload: Any = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Return a deferred block load callable.

    The callable binds layout, item-count, CTA-size options, and optional dtype,
    then forwards each kernel-trace call to :func:`load`.
    """
    _validate_payload_selector(
        payload,
        scope=_SCOPE,
        primitive_name="make_load",
    )
    _reject_methods("make_load", kwargs)
    _reject_prims_specific_load_store_factory_kwargs(_SCOPE, "make_load", kwargs)
    bound = dict(kwargs)
    resolved_threads = resolve_block_threads(
        _SCOPE,
        "make_load",
        threads_per_block=threads_per_block,
        dim=dim,
    )
    if resolved_threads is not None:
        bound["threads_per_block"] = resolved_threads
    valid_items = _normalize_valid_items_aliases(
        "make_load",
        valid_items=valid_items,
        num_valid_items=num_valid_items,
    )
    bound.update(
        items_per_thread=_resolve_static_items_per_thread_for_scope(
            _SCOPE,
            "make_load",
            items_per_thread,
        ),
        algorithm=algorithm,
    )
    _bind_if_not_none(bound, "dtype", dtype)
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
    threads_per_block: Any = None,
    items_per_thread: int = 1,
    algorithm: Any = "direct",
    *,
    dim: Any = None,
    valid_items: Any = None,
    num_valid_items: Any = None,
    payload: Any = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Return a deferred block store callable.

    The callable binds layout, item-count, CTA-size options, and optional dtype,
    then forwards each kernel-trace call to :func:`store`.
    """
    _validate_payload_selector(
        payload,
        scope=_SCOPE,
        primitive_name="make_store",
    )
    _reject_methods("make_store", kwargs)
    _reject_prims_specific_load_store_factory_kwargs(_SCOPE, "make_store", kwargs)
    bound = dict(kwargs)
    resolved_threads = resolve_block_threads(
        _SCOPE,
        "make_store",
        threads_per_block=threads_per_block,
        dim=dim,
    )
    if resolved_threads is not None:
        bound["threads_per_block"] = resolved_threads
    valid_items = _normalize_valid_items_aliases(
        "make_store",
        valid_items=valid_items,
        num_valid_items=num_valid_items,
    )
    bound.update(
        items_per_thread=_resolve_static_items_per_thread_for_scope(
            _SCOPE,
            "make_store",
            items_per_thread,
        ),
        algorithm=algorithm,
    )
    _bind_if_not_none(bound, "dtype", dtype)
    _bind_if_not_none(bound, "valid_items", valid_items)
    return _make_factory(
        "make_store",
        store,
        bound,
        overridable_kwargs=_LOAD_STORE_VALID_OVERRIDABLE_KWARGS,
        override_aliases=_LOAD_STORE_VALID_OVERRIDE_ALIASES,
    )


def make_exchange(
    dtype: Any,
    threads_per_block: Any = None,
    items_per_thread: int = 1,
    block_exchange_type: Any = _DEFAULT_SELECTOR,
    *,
    dim: Any = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Return a deferred block exchange callable.

    The callable binds the CUB exchange mode and CTA metadata, then forwards
    per-thread scalar or ``ThreadData`` items to :func:`exchange`.
    """
    dtype, block_exchange_type = _select_block_exchange_type(
        "make_exchange",
        dtype,
        block_exchange_type,
    )
    del dtype, items_per_thread
    _reject_methods("make_exchange", kwargs)
    bound = _block_kwargs(
        "make_exchange",
        threads_per_block=threads_per_block,
        dim=dim,
        kwargs=kwargs,
    )
    if block_exchange_type is not _DEFAULT_SELECTOR:
        bound["block_exchange_type"] = block_exchange_type
    return _make_factory("make_exchange", exchange, bound)


def make_reduce(
    dtype: Any,
    threads_per_block: Any = None,
    binary_op: Any = None,
    items_per_thread: int = 1,
    algorithm: Any = None,
    *,
    dim: Any = None,
    num_valid: Any = None,
    valid_items: Any = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Return a deferred block reduce callable.

    The callable binds the reduction operator and CTA metadata, then forwards
    each scalar or ``ThreadData`` value to :func:`reduce`. An explicitly
    selected CUB ``algorithm`` is a static deferred-call argument; omission
    preserves the scoped primitive's canonical full-group CUDAX route.
    """
    del dtype, items_per_thread
    _reject_methods("make_reduce", kwargs)
    if num_valid is not None and valid_items is not None:
        raise TypeError(f"{_SCOPE}.make_reduce got both num_valid and valid_items")
    if valid_items is not None:
        num_valid = valid_items
    bound = _block_kwargs(
        "make_reduce",
        threads_per_block=threads_per_block,
        dim=dim,
        kwargs=kwargs,
    )
    _bind_if_not_none(bound, "binary_op", binary_op)
    _bind_if_not_none(bound, "num_valid", num_valid)
    _bind_if_not_none(bound, "algorithm", algorithm)
    return _make_factory(
        "make_reduce",
        reduce,
        bound,
        overridable_kwargs=_REDUCE_VALID_OVERRIDABLE_KWARGS,
        override_aliases=_REDUCE_VALID_OVERRIDE_ALIASES,
    )


def make_sum(
    dtype: Any,
    threads_per_block: Any = None,
    items_per_thread: int = 1,
    algorithm: Any = None,
    *,
    dim: Any = None,
    num_valid: Any = None,
    valid_items: Any = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Return a deferred block sum callable.

    The callable binds CTA metadata and forwards each scalar or ``ThreadData``
    value to :func:`sum`. An explicitly selected CUB ``algorithm`` is a static
    deferred-call argument; omission preserves the scoped primitive's
    canonical full-group CUDAX route.
    """
    del dtype, items_per_thread
    _reject_methods("make_sum", kwargs)
    if num_valid is not None and valid_items is not None:
        raise TypeError(f"{_SCOPE}.make_sum got both num_valid and valid_items")
    if valid_items is not None:
        num_valid = valid_items
    bound = _block_kwargs(
        "make_sum",
        threads_per_block=threads_per_block,
        dim=dim,
        kwargs=kwargs,
    )
    _bind_if_not_none(bound, "num_valid", num_valid)
    _bind_if_not_none(bound, "algorithm", algorithm)
    return _make_factory(
        "make_sum",
        sum,
        bound,
        overridable_kwargs=_REDUCE_VALID_OVERRIDABLE_KWARGS,
        override_aliases=_REDUCE_VALID_OVERRIDE_ALIASES,
    )


def make_scan(
    dtype: Any,
    threads_per_block: Any = None,
    items_per_thread: int = 1,
    initial_value: Any = None,
    mode: str = "exclusive",
    scan_op: Any = "+",
    prefix_op: Any = None,
    *,
    dim: Any = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Return a deferred block scan callable.

    The callable binds scan mode, operator, optional initial value, and CTA
    metadata before forwarding each value to :func:`scan`.
    """
    del dtype, items_per_thread
    _reject_if_supplied("make_scan", "prefix_op", prefix_op)
    _reject_methods("make_scan", kwargs)
    bound = _block_kwargs(
        "make_scan",
        threads_per_block=threads_per_block,
        dim=dim,
        kwargs=kwargs,
    )
    bound["mode"] = mode
    bound["scan_op"] = scan_op
    _bind_if_not_none(bound, "initial_value", initial_value)
    return _make_factory(
        "make_scan",
        scan,
        bound,
        overridable_kwargs=_SCAN_OUTPUT_OVERRIDABLE_KWARGS,
    )


def make_exclusive_sum(
    dtype: Any,
    threads_per_block: Any = None,
    items_per_thread: int = 1,
    prefix_op: Any = None,
    algorithm: Any = None,
    *,
    dim: Any = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Return a deferred block exclusive-sum callable.

    The callable binds CTA metadata and forwards each scalar or ``ThreadData``
    value to :func:`exclusive_sum`.
    """
    del dtype, items_per_thread
    _reject_if_supplied("make_exclusive_sum", "prefix_op", prefix_op)
    _reject_algorithm("make_exclusive_sum", algorithm, default=None)
    _reject_methods("make_exclusive_sum", kwargs)
    bound = _block_kwargs(
        "make_exclusive_sum",
        threads_per_block=threads_per_block,
        dim=dim,
        kwargs=kwargs,
    )
    return _make_factory(
        "make_exclusive_sum",
        exclusive_sum,
        bound,
        overridable_kwargs=_SCAN_OUTPUT_OVERRIDABLE_KWARGS,
    )


def make_inclusive_sum(
    dtype: Any,
    threads_per_block: Any = None,
    items_per_thread: int = 1,
    prefix_op: Any = None,
    algorithm: Any = None,
    *,
    dim: Any = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Return a deferred block inclusive-sum callable.

    The callable binds CTA metadata and forwards each scalar or ``ThreadData``
    value to :func:`inclusive_sum`.
    """
    del dtype, items_per_thread
    _reject_if_supplied("make_inclusive_sum", "prefix_op", prefix_op)
    _reject_algorithm("make_inclusive_sum", algorithm, default=None)
    _reject_methods("make_inclusive_sum", kwargs)
    bound = _block_kwargs(
        "make_inclusive_sum",
        threads_per_block=threads_per_block,
        dim=dim,
        kwargs=kwargs,
    )
    return _make_factory(
        "make_inclusive_sum",
        inclusive_sum,
        bound,
        overridable_kwargs=_SCAN_OUTPUT_OVERRIDABLE_KWARGS,
    )


def make_exclusive_scan(
    dtype: Any,
    threads_per_block: Any = None,
    scan_op: Any = "+",
    items_per_thread: int = 1,
    initial_value: Any = None,
    prefix_op: Any = None,
    algorithm: Any = None,
    *,
    dim: Any = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Return a deferred block exclusive-scan callable.

    The callable binds scan operator, optional initial value, and CTA metadata
    before forwarding each value to :func:`exclusive_scan`.
    """
    del dtype, items_per_thread
    _reject_if_supplied("make_exclusive_scan", "prefix_op", prefix_op)
    _reject_algorithm("make_exclusive_scan", algorithm, default=None)
    _reject_methods("make_exclusive_scan", kwargs)
    bound = _block_kwargs(
        "make_exclusive_scan",
        threads_per_block=threads_per_block,
        dim=dim,
        kwargs=kwargs,
    )
    bound["scan_op"] = scan_op
    _bind_if_not_none(bound, "initial_value", initial_value)
    return _make_factory(
        "make_exclusive_scan",
        exclusive_scan,
        bound,
        overridable_kwargs=_SCAN_OUTPUT_OVERRIDABLE_KWARGS,
    )


def make_inclusive_scan(
    dtype: Any,
    threads_per_block: Any = None,
    scan_op: Any = "+",
    items_per_thread: int = 1,
    initial_value: Any = None,
    prefix_op: Any = None,
    algorithm: Any = None,
    *,
    dim: Any = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Return a deferred block inclusive-scan callable.

    The callable binds scan operator, optional initial value, and CTA metadata
    before forwarding each value to :func:`inclusive_scan`.
    """
    del dtype, items_per_thread
    _reject_if_supplied("make_inclusive_scan", "prefix_op", prefix_op)
    _reject_algorithm("make_inclusive_scan", algorithm, default=None)
    _reject_methods("make_inclusive_scan", kwargs)
    bound = _block_kwargs(
        "make_inclusive_scan",
        threads_per_block=threads_per_block,
        dim=dim,
        kwargs=kwargs,
    )
    bound["scan_op"] = scan_op
    _bind_if_not_none(bound, "initial_value", initial_value)
    return _make_factory(
        "make_inclusive_scan",
        inclusive_scan,
        bound,
        overridable_kwargs=_SCAN_OUTPUT_OVERRIDABLE_KWARGS,
    )


def make_shuffle(
    dtype: Any,
    threads_per_block: Any = None,
    items_per_thread: int | None = None,
    block_shuffle_type: Any = _DEFAULT_SELECTOR,
    distance: Any = None,
    *,
    dim: Any = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Return a deferred block shuffle callable.

    The callable binds shuffle mode, distance, and CTA metadata, then forwards
    each scalar or ``ThreadData`` value to :func:`shuffle`.
    """
    del dtype, items_per_thread
    _reject_methods("make_shuffle", kwargs)
    bound = _block_kwargs(
        "make_shuffle",
        threads_per_block=threads_per_block,
        dim=dim,
        kwargs=kwargs,
    )
    overridable_kwargs: tuple[str, ...] = ()
    if block_shuffle_type is _DEFAULT_SELECTOR:
        if "mode" not in bound:
            bound["mode"] = "up"
            overridable_kwargs = ("mode",)
    else:
        bound["block_shuffle_type"] = block_shuffle_type
    _bind_if_not_none(bound, "distance", distance)
    return _make_factory(
        "make_shuffle",
        shuffle,
        bound,
        overridable_kwargs=overridable_kwargs,
        override_aliases=(("block_shuffle_type", ("mode",)),),
    )
