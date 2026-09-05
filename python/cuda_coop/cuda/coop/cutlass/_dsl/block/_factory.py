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
from ._difference import adjacent_difference
from ._discontinuity import discontinuity
from ._exchange import BlockExchangeType, exchange
from ._histogram import histogram
from ._load_store import load, store
from ._reduce import reduce, sum
from ._run_length import run_length
from ._scan import exclusive_scan, exclusive_sum, inclusive_scan, inclusive_sum, scan
from ._shuffle import shuffle
from ._sort import (
    merge_sort_keys,
    merge_sort_pairs,
    radix_rank,
    radix_sort_keys,
    radix_sort_keys_descending,
    radix_sort_pairs,
    radix_sort_pairs_descending,
    topk_max_keys,
    topk_max_pairs,
    topk_min_keys,
    topk_min_pairs,
)

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


def _bind_radix_sort_defaults(
    bound: dict[str, Any],
    *,
    begin_bit: Any,
    end_bit: Any | None,
    descending: bool | None = None,
) -> None:
    bound["begin_bit"] = begin_bit
    _bind_if_not_none(bound, "end_bit", end_bit)
    if descending is not None:
        bound["descending"] = descending


def _bind_topk_defaults(
    bound: dict[str, Any],
    *,
    num_valid: Any,
    begin_bit: Any,
    end_bit: Any | None,
) -> None:
    _bind_if_not_none(bound, "num_valid", num_valid)
    bound["begin_bit"] = begin_bit
    _bind_if_not_none(bound, "end_bit", end_bit)


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


def make_merge_sort_keys(
    dtype: Any,
    threads_per_block: Any = None,
    items_per_thread: int = 1,
    compare_op: Any = None,
    *,
    descending: bool | None = None,
    dim: Any = None,
    valid_items: Any = None,
    oob_default: Any = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Return a deferred block merge-sort-keys callable.

    The callable binds optional sort direction, partial-tile defaults, and CTA
    metadata, then forwards scalar or ``ThreadData`` keys to
    :func:`merge_sort_keys`. Deferred calls may override the value defaults.
    """
    del dtype, items_per_thread
    _reject_if_supplied("make_merge_sort_keys", "compare_op", compare_op)
    _reject_methods("make_merge_sort_keys", kwargs)
    bound = _block_kwargs(
        "make_merge_sort_keys",
        threads_per_block=threads_per_block,
        dim=dim,
        kwargs=kwargs,
    )
    _bind_if_not_none(bound, "descending", descending)
    _bind_if_not_none(bound, "valid_items", valid_items)
    _bind_if_not_none(bound, "oob_default", oob_default)
    return _make_factory(
        "make_merge_sort_keys",
        merge_sort_keys,
        bound,
        overridable_kwargs=_MERGE_SORT_OVERRIDABLE_KWARGS,
    )


def make_merge_sort_pairs(
    keys: Any = None,
    values: Any = None,
    threads_per_block: Any = None,
    items_per_thread: int = 1,
    compare_op: Any = None,
    *,
    descending: bool | None = None,
    dim: Any = None,
    valid_items: Any = None,
    oob_default: Any = None,
    key_dtype: Any = None,
    value_dtype: Any = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Return a deferred block merge-sort-pairs callable.

    The callable binds optional sort direction, partial-tile defaults, and CTA
    metadata, then forwards key/value scalar or ``ThreadData`` pairs to
    :func:`merge_sort_pairs`. Deferred calls may override the value defaults.
    """
    keys, values = _normalize_pair_dtype_aliases(
        "make_merge_sort_pairs",
        keys,
        values,
        key_dtype,
        value_dtype,
    )
    del keys, values, items_per_thread
    _reject_if_supplied("make_merge_sort_pairs", "compare_op", compare_op)
    _reject_methods("make_merge_sort_pairs", kwargs)
    bound = _block_kwargs(
        "make_merge_sort_pairs",
        threads_per_block=threads_per_block,
        dim=dim,
        kwargs=kwargs,
    )
    _bind_if_not_none(bound, "descending", descending)
    _bind_if_not_none(bound, "valid_items", valid_items)
    _bind_if_not_none(bound, "oob_default", oob_default)
    return _make_factory(
        "make_merge_sort_pairs",
        merge_sort_pairs,
        bound,
        overridable_kwargs=_MERGE_SORT_OVERRIDABLE_KWARGS,
    )


def make_radix_sort_keys(
    dtype: Any,
    threads_per_block: Any = None,
    items_per_thread: int = 1,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    descending: bool = False,
    *,
    dim: Any = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Return a deferred block radix-sort-keys callable.

    The callable binds CTA metadata and radix bit-slice defaults, then forwards
    each key tile to :func:`radix_sort_keys`. Deferred calls may override the
    bound bit slice or sort direction.
    """
    del dtype, items_per_thread
    _reject_methods("make_radix_sort_keys", kwargs)
    bound = _block_kwargs(
        "make_radix_sort_keys",
        threads_per_block=threads_per_block,
        dim=dim,
        kwargs=kwargs,
    )
    _bind_radix_sort_defaults(
        bound,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
    )
    return _make_factory(
        "make_radix_sort_keys",
        radix_sort_keys,
        bound,
        overridable_kwargs=_RADIX_SORT_OVERRIDABLE_KWARGS,
    )


def make_radix_sort_keys_descending(
    dtype: Any,
    threads_per_block: Any = None,
    items_per_thread: int = 1,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    *,
    dim: Any = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Return a deferred descending block radix-sort-keys callable.

    The callable binds CTA metadata and radix bit-slice defaults, then forwards
    each key tile to :func:`radix_sort_keys_descending`. Deferred calls may
    override the bound bit slice.
    """
    del dtype, items_per_thread
    _reject_methods("make_radix_sort_keys_descending", kwargs)
    bound = _block_kwargs(
        "make_radix_sort_keys_descending",
        threads_per_block=threads_per_block,
        dim=dim,
        kwargs=kwargs,
    )
    _bind_radix_sort_defaults(
        bound,
        begin_bit=begin_bit,
        end_bit=end_bit,
    )
    return _make_factory(
        "make_radix_sort_keys_descending",
        radix_sort_keys_descending,
        bound,
        overridable_kwargs=_RADIX_SORT_DESCENDING_OVERRIDABLE_KWARGS,
    )


def make_radix_sort_pairs(
    keys: Any = None,
    values: Any = None,
    threads_per_block: Any = None,
    items_per_thread: int = 1,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    descending: bool = False,
    *,
    dim: Any = None,
    key_dtype: Any = None,
    value_dtype: Any = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Return a deferred block radix-sort-pairs callable.

    The callable binds CTA metadata and radix bit-slice defaults, then forwards
    key/value tiles to :func:`radix_sort_pairs`. Deferred calls may override the
    bound bit slice or sort direction.
    """
    keys, values = _normalize_pair_dtype_aliases(
        "make_radix_sort_pairs",
        keys,
        values,
        key_dtype,
        value_dtype,
    )
    del keys, values, items_per_thread
    _reject_methods("make_radix_sort_pairs", kwargs)
    bound = _block_kwargs(
        "make_radix_sort_pairs",
        threads_per_block=threads_per_block,
        dim=dim,
        kwargs=kwargs,
    )
    _bind_radix_sort_defaults(
        bound,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
    )
    return _make_factory(
        "make_radix_sort_pairs",
        radix_sort_pairs,
        bound,
        overridable_kwargs=_RADIX_SORT_OVERRIDABLE_KWARGS,
    )


def make_radix_sort_pairs_descending(
    keys: Any = None,
    values: Any = None,
    threads_per_block: Any = None,
    items_per_thread: int = 1,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    *,
    dim: Any = None,
    key_dtype: Any = None,
    value_dtype: Any = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Return a deferred descending block radix-sort-pairs callable.

    The callable binds CTA metadata and radix bit-slice defaults, then forwards
    key/value tiles to :func:`radix_sort_pairs_descending`. Deferred calls may
    override the bound bit slice.
    """
    keys, values = _normalize_pair_dtype_aliases(
        "make_radix_sort_pairs_descending",
        keys,
        values,
        key_dtype,
        value_dtype,
    )
    del keys, values, items_per_thread
    _reject_methods("make_radix_sort_pairs_descending", kwargs)
    bound = _block_kwargs(
        "make_radix_sort_pairs_descending",
        threads_per_block=threads_per_block,
        dim=dim,
        kwargs=kwargs,
    )
    _bind_radix_sort_defaults(
        bound,
        begin_bit=begin_bit,
        end_bit=end_bit,
    )
    return _make_factory(
        "make_radix_sort_pairs_descending",
        radix_sort_pairs_descending,
        bound,
        overridable_kwargs=_RADIX_SORT_DESCENDING_OVERRIDABLE_KWARGS,
    )


def make_radix_rank(
    dtype: Any,
    threads_per_block: Any = None,
    items_per_thread: int = 1,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    radix_bits: Any | None = None,
    descending: bool = False,
    *,
    dim: Any = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Return a deferred block radix-rank callable.

    The callable binds CTA metadata and radix bit-slice defaults, then forwards
    each key tile to :func:`radix_rank`. Deferred calls may override the bound
    bit slice or rank direction.
    """
    del dtype, items_per_thread
    _reject_methods("make_radix_rank", kwargs)
    bound = _block_kwargs(
        "make_radix_rank",
        threads_per_block=threads_per_block,
        dim=dim,
        kwargs=kwargs,
    )
    bound["begin_bit"] = begin_bit
    _bind_if_not_none(bound, "end_bit", end_bit)
    _bind_if_not_none(bound, "radix_bits", radix_bits)
    bound["descending"] = descending
    return _make_factory(
        "make_radix_rank",
        radix_rank,
        bound,
        overridable_kwargs=_RADIX_RANK_OVERRIDABLE_KWARGS,
        override_aliases=_RADIX_RANK_OVERRIDE_ALIASES,
    )


def make_topk_max_keys(
    dtype: Any,
    threads_per_block: Any = None,
    items_per_thread: int = 1,
    *,
    dim: Any = None,
    num_valid: Any = None,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Return a deferred block TopK-largest-keys callable.

    The callable binds CTA metadata plus optional valid-count and radix
    bit-slice defaults, then forwards key tiles plus runtime ``k`` to
    :func:`topk_max_keys`. Deferred calls may override the value defaults.
    """
    del dtype, items_per_thread
    _reject_methods("make_topk_max_keys", kwargs)
    bound = _block_kwargs(
        "make_topk_max_keys",
        threads_per_block=threads_per_block,
        dim=dim,
        kwargs=kwargs,
    )
    _bind_topk_defaults(
        bound,
        num_valid=num_valid,
        begin_bit=begin_bit,
        end_bit=end_bit,
    )
    return _make_factory(
        "make_topk_max_keys",
        topk_max_keys,
        bound,
        overridable_kwargs=_TOPK_OVERRIDABLE_KWARGS,
    )


def make_topk_min_keys(
    dtype: Any,
    threads_per_block: Any = None,
    items_per_thread: int = 1,
    *,
    dim: Any = None,
    num_valid: Any = None,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Return a deferred block TopK-smallest-keys callable.

    The callable binds CTA metadata plus optional valid-count and radix
    bit-slice defaults, then forwards key tiles plus runtime ``k`` to
    :func:`topk_min_keys`. Deferred calls may override the value defaults.
    """
    del dtype, items_per_thread
    _reject_methods("make_topk_min_keys", kwargs)
    bound = _block_kwargs(
        "make_topk_min_keys",
        threads_per_block=threads_per_block,
        dim=dim,
        kwargs=kwargs,
    )
    _bind_topk_defaults(
        bound,
        num_valid=num_valid,
        begin_bit=begin_bit,
        end_bit=end_bit,
    )
    return _make_factory(
        "make_topk_min_keys",
        topk_min_keys,
        bound,
        overridable_kwargs=_TOPK_OVERRIDABLE_KWARGS,
    )


def make_topk_max_pairs(
    keys: Any = None,
    values: Any = None,
    threads_per_block: Any = None,
    items_per_thread: int = 1,
    *,
    dim: Any = None,
    key_dtype: Any = None,
    value_dtype: Any = None,
    num_valid: Any = None,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Return a deferred block TopK-largest-pairs callable.

    The callable binds CTA metadata plus optional valid-count and radix
    bit-slice defaults, then forwards key/value tiles plus runtime ``k`` to
    :func:`topk_max_pairs`. Deferred calls may override the value defaults.
    """
    keys, values = _normalize_pair_dtype_aliases(
        "make_topk_max_pairs",
        keys,
        values,
        key_dtype,
        value_dtype,
    )
    del keys, values, items_per_thread
    _reject_methods("make_topk_max_pairs", kwargs)
    bound = _block_kwargs(
        "make_topk_max_pairs",
        threads_per_block=threads_per_block,
        dim=dim,
        kwargs=kwargs,
    )
    _bind_topk_defaults(
        bound,
        num_valid=num_valid,
        begin_bit=begin_bit,
        end_bit=end_bit,
    )
    return _make_factory(
        "make_topk_max_pairs",
        topk_max_pairs,
        bound,
        overridable_kwargs=_TOPK_OVERRIDABLE_KWARGS,
    )


def make_topk_min_pairs(
    keys: Any = None,
    values: Any = None,
    threads_per_block: Any = None,
    items_per_thread: int = 1,
    *,
    dim: Any = None,
    key_dtype: Any = None,
    value_dtype: Any = None,
    num_valid: Any = None,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Return a deferred block TopK-smallest-pairs callable.

    The callable binds CTA metadata plus optional valid-count and radix
    bit-slice defaults, then forwards key/value tiles plus runtime ``k`` to
    :func:`topk_min_pairs`. Deferred calls may override the value defaults.
    """
    keys, values = _normalize_pair_dtype_aliases(
        "make_topk_min_pairs",
        keys,
        values,
        key_dtype,
        value_dtype,
    )
    del keys, values, items_per_thread
    _reject_methods("make_topk_min_pairs", kwargs)
    bound = _block_kwargs(
        "make_topk_min_pairs",
        threads_per_block=threads_per_block,
        dim=dim,
        kwargs=kwargs,
    )
    _bind_topk_defaults(
        bound,
        num_valid=num_valid,
        begin_bit=begin_bit,
        end_bit=end_bit,
    )
    return _make_factory(
        "make_topk_min_pairs",
        topk_min_pairs,
        bound,
        overridable_kwargs=_TOPK_OVERRIDABLE_KWARGS,
    )


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


def make_adjacent_difference(
    dtype: Any,
    threads_per_block: Any = None,
    items_per_thread: int = 1,
    difference_op: Any = None,
    *,
    dim: Any = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Return a deferred block adjacent-difference callable.

    The callable binds CTA metadata and forwards each value to
    :func:`adjacent_difference`.
    """
    del dtype, items_per_thread
    _reject_methods("make_adjacent_difference", kwargs)
    bound = _block_kwargs(
        "make_adjacent_difference",
        threads_per_block=threads_per_block,
        dim=dim,
        kwargs=kwargs,
    )
    _bind_if_not_none(bound, "difference_op", difference_op)
    return _make_factory("make_adjacent_difference", adjacent_difference, bound)


def make_discontinuity(
    dtype: Any,
    threads_per_block: Any = None,
    items_per_thread: int = 1,
    flag_op: Any = None,
    *,
    dim: Any = None,
    flag_dtype: Any = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Return a deferred block discontinuity callable.

    The callable binds CTA metadata and forwards each value to
    :func:`discontinuity`.
    """
    del dtype, items_per_thread
    _reject_if_supplied("make_discontinuity", "flag_dtype", flag_dtype)
    _reject_methods("make_discontinuity", kwargs)
    bound = _block_kwargs(
        "make_discontinuity",
        threads_per_block=threads_per_block,
        dim=dim,
        kwargs=kwargs,
    )
    _bind_if_not_none(bound, "flag_op", flag_op)
    return _make_factory("make_discontinuity", discontinuity, bound)


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


def make_histogram(
    item_dtype: Any,
    counter_dtype: Any,
    threads_per_block: Any = None,
    items_per_thread: int = 1,
    *,
    dim: Any = None,
    bins: Any = None,
    bins_per_thread: int | None = None,
    algorithm: Any = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Return a deferred block histogram callable.

    The callable binds counter dtype, optional histogram defaults, and CTA
    metadata before forwarding per-thread samples to :func:`histogram`.
    ``bins`` is trace-time static. Deferred calls may override a bound value
    with another static bin count.
    """
    del item_dtype, items_per_thread
    _reject_methods("make_histogram", kwargs)
    bound = _block_kwargs(
        "make_histogram",
        threads_per_block=threads_per_block,
        dim=dim,
        kwargs=kwargs,
    )
    bound["counter_dtype"] = counter_dtype
    _bind_if_not_none(bound, "bins", bins)
    _bind_if_not_none(bound, "bins_per_thread", bins_per_thread)
    _bind_if_not_none(bound, "algorithm", algorithm)
    return _make_factory(
        "make_histogram",
        histogram,
        bound,
        overridable_kwargs=_HISTOGRAM_OVERRIDABLE_KWARGS,
    )


def make_run_length(
    item_dtype: Any,
    threads_per_block: Any = None,
    runs_per_thread: int | None = None,
    decoded_items_per_thread: int = 1,
    *,
    dim: Any = None,
    total_decoded_size: Any = None,
    decoded_offset_dtype: Any = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Return a deferred block run-length callable.

    The callable binds decoded-window options, optional decoded-size outputs,
    and CTA metadata, then forwards run values and lengths to
    :func:`run_length` to create the decode parent. Deferred calls may override
    runtime ``total_decoded_size`` defaults.
    """
    del item_dtype
    _reject_methods("make_run_length", kwargs)
    bound = _block_kwargs(
        "make_run_length",
        threads_per_block=threads_per_block,
        dim=dim,
        kwargs=kwargs,
    )
    if runs_per_thread is not None:
        bound["runs_per_thread"] = runs_per_thread
    bound["decoded_items_per_thread"] = decoded_items_per_thread
    _bind_if_not_none(bound, "decoded_offset_dtype", decoded_offset_dtype)
    _bind_if_not_none(bound, "total_decoded_size", total_decoded_size)
    return _make_factory(
        "make_run_length",
        run_length,
        bound,
        overridable_kwargs=_RUN_LENGTH_OVERRIDABLE_KWARGS,
    )
