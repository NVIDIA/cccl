# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from numbers import Integral
from typing import Any

from .._launch import pop_launch_metadata
from .._scope import BLOCK_SCOPE as _SCOPE
from .._scope import merge_block_payload as merge_payload
from .._scope import validate_no_extra_block_args as validate_no_extra_args
from ._dispatch import dispatch_primitive, register_primitive_impl


def _validate_sort_bits(
    primitive_name: str,
    *,
    begin_bit: Any,
    end_bit: Any | None = None,
) -> Any | None:
    del primitive_name
    static_begin_bit = _static_int_value(begin_bit)
    static_end_bit = None if end_bit is None else _static_int_value(end_bit)
    if static_begin_bit is not None and static_begin_bit < 0:
        raise ValueError("begin_bit must be non-negative")
    if (
        static_begin_bit is not None
        and static_end_bit is not None
        and static_end_bit <= static_begin_bit
    ):
        raise ValueError("end_bit must be greater than begin_bit")
    return end_bit


def _static_int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        raise TypeError("radix bit bounds must be int-like scalars")
    if isinstance(value, Integral):
        return int(value)
    return None


def _validate_partial_tile_args(
    primitive_name: str,
    *,
    valid_items: Any,
    oob_default: Any,
) -> None:
    if (valid_items is None) != (oob_default is None):
        raise ValueError(
            f"{_SCOPE}.{primitive_name} valid_items and oob_default "
            "must be provided together"
        )


def _radix_sort_keys_provider(
    *,
    keys: Any,
    args: tuple[Any, ...] = (),
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    descending: bool = False,
    **kwargs: Any,
) -> Any:
    from ... import _group_radix
    from ..._thread_group import this_block

    return _group_radix._radix_sort_keys(
        this_block(),
        keys,
        *args,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        source="cutlass_scoped_block",
        **kwargs,
    )


_radix_sort_keys_provider._supports_native_thread_data = True
_radix_sort_keys_provider._preserves_launch_metadata = True
_radix_sort_keys_provider._uses_planned_temp_storage = True
_radix_sort_keys_provider._supports_deferred_temp_storage = True


def _radix_sort_pairs_provider(
    *,
    keys: Any,
    values: Any,
    args: tuple[Any, ...] = (),
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    descending: bool = False,
    **kwargs: Any,
) -> tuple[Any, Any]:
    from ... import _group_radix
    from ..._thread_group import this_block

    return _group_radix._radix_sort_pairs(
        this_block(),
        keys,
        values,
        *args,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        source="cutlass_scoped_block",
        **kwargs,
    )


_radix_sort_pairs_provider._supports_native_thread_data = True
_radix_sort_pairs_provider._preserves_launch_metadata = True
_radix_sort_pairs_provider._uses_planned_temp_storage = True
_radix_sort_pairs_provider._supports_deferred_temp_storage = True


def _radix_rank_provider(
    *,
    keys: Any,
    args: tuple[Any, ...] = (),
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    radix_bits: Any | None = None,
    descending: bool = False,
    exclusive_digit_prefix: Any = None,
    **kwargs: Any,
) -> Any:
    from ... import _group_radix
    from ..._thread_group import this_block

    return _group_radix._radix_rank(
        this_block(),
        keys,
        *args,
        begin_bit=begin_bit,
        end_bit=end_bit,
        radix_bits=radix_bits,
        descending=descending,
        exclusive_digit_prefix=exclusive_digit_prefix,
        source="cutlass_scoped_block",
        **kwargs,
    )


_radix_rank_provider._supports_native_thread_data = True
_radix_rank_provider._preserves_launch_metadata = True
_radix_rank_provider._uses_planned_temp_storage = True


def _merge_sort_keys_provider(
    *,
    keys: Any,
    args: tuple[Any, ...] = (),
    descending: bool = False,
    valid_items: Any = None,
    oob_default: Any = None,
    **kwargs: Any,
) -> Any:
    if args:
        validate_no_extra_args(
            "merge_sort_keys",
            args=args,
            kwargs={},
            expected="does not accept extra positional args",
        )
    _validate_partial_tile_args(
        "merge_sort_keys",
        valid_items=valid_items,
        oob_default=oob_default,
    )

    from ... import _group_merge_sort as _group_frontend
    from ..._thread_group import this_block

    return _group_frontend._merge_sort(
        this_block(),
        keys,
        None,
        descending=descending,
        valid_items=valid_items,
        oob_default=oob_default,
        source="scoped_block",
        **kwargs,
    )


_merge_sort_keys_provider._supports_native_thread_data = True
_merge_sort_keys_provider._preserves_launch_metadata = True
_merge_sort_keys_provider._uses_planned_temp_storage = True
_merge_sort_keys_provider._supports_deferred_temp_storage = True


def _merge_sort_pairs_provider(
    *,
    keys: Any,
    values: Any,
    args: tuple[Any, ...] = (),
    descending: bool = False,
    valid_items: Any = None,
    oob_default: Any = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    if args:
        validate_no_extra_args(
            "merge_sort_pairs",
            args=args,
            kwargs={},
            expected="does not accept extra positional args",
        )
    _validate_partial_tile_args(
        "merge_sort_pairs",
        valid_items=valid_items,
        oob_default=oob_default,
    )

    from ... import _group_merge_sort as _group_frontend
    from ..._thread_group import this_block

    return _group_frontend._merge_sort(
        this_block(),
        keys,
        values,
        descending=descending,
        valid_items=valid_items,
        oob_default=oob_default,
        source="scoped_block",
        **kwargs,
    )


_merge_sort_pairs_provider._supports_native_thread_data = True
_merge_sort_pairs_provider._preserves_launch_metadata = True
_merge_sort_pairs_provider._uses_planned_temp_storage = True
_merge_sort_pairs_provider._supports_deferred_temp_storage = True


def _topk_keys_provider(
    *,
    keys: Any,
    k: Any,
    args: tuple[Any, ...] = (),
    num_valid: Any = None,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    descending: bool,
    **kwargs: Any,
) -> Any:
    launch_kwargs = pop_launch_metadata(kwargs)
    validate_no_extra_args(
        "topk_keys",
        args=args,
        kwargs=kwargs,
        expected="does not accept extra positional args",
    )

    from . import _provider

    return _provider.provider_topk_keys(
        key=keys,
        k=k,
        num_valid=num_valid,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        block_threads=_provider.infer_topk_block_threads(launch_kwargs),
        temp_storage_primitive_name="topk_max_keys" if descending else "topk_min_keys",
    )


_topk_keys_provider._supports_native_thread_data = True
_topk_keys_provider._preserves_launch_metadata = True


def _topk_pairs_provider(
    *,
    keys: Any,
    values: Any,
    k: Any,
    args: tuple[Any, ...] = (),
    num_valid: Any = None,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    descending: bool,
    **kwargs: Any,
) -> tuple[Any, Any]:
    launch_kwargs = pop_launch_metadata(kwargs)
    validate_no_extra_args(
        "topk_pairs",
        args=args,
        kwargs=kwargs,
        expected="does not accept extra positional args",
    )

    from . import _provider

    return _provider.provider_topk_pairs(
        key=keys,
        value=values,
        k=k,
        num_valid=num_valid,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        block_threads=_provider.infer_topk_block_threads(launch_kwargs),
        temp_storage_primitive_name="topk_max_pairs"
        if descending
        else "topk_min_pairs",
    )


_topk_pairs_provider._supports_native_thread_data = True
_topk_pairs_provider._preserves_launch_metadata = True


def radix_sort_keys(
    keys: Any,
    /,
    *args: Any,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    descending: bool = False,
    temp_storage: Any = None,
    **kwargs: Any,
) -> Any:
    """Return block-wide CUB radix-sorted keys.

    The input may be a scalar key or ``ThreadData`` keys. ``begin_bit`` and
    ``end_bit`` select the radix slice, and ``descending`` reverses the CUB
    ordering.
    """
    structural_payload = {
        "keys": keys,
        "args": args,
        "begin_bit": begin_bit,
        "end_bit": end_bit,
        "descending": descending,
    }
    if temp_storage is not None:
        structural_payload["temp_storage"] = temp_storage
    payload = merge_payload(
        "radix_sort_keys",
        structural_payload,
        kwargs,
    )
    return dispatch_primitive("radix_sort_keys", kwargs=payload)


def radix_sort_pairs(
    keys: Any,
    values: Any,
    /,
    *args: Any,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    descending: bool = False,
    temp_storage: Any = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Return block-wide CUB radix-sorted key/value pairs.

    Keys and values must both be scalar values or both be ``ThreadData`` with
    matching ``items_per_thread``. The output preserves the key/value
    association after sorting.
    """
    structural_payload = {
        "keys": keys,
        "values": values,
        "args": args,
        "begin_bit": begin_bit,
        "end_bit": end_bit,
        "descending": descending,
    }
    if temp_storage is not None:
        structural_payload["temp_storage"] = temp_storage
    payload = merge_payload(
        "radix_sort_pairs",
        structural_payload,
        kwargs,
    )
    return dispatch_primitive("radix_sort_pairs", kwargs=payload)


def radix_rank(
    keys: Any,
    /,
    *args: Any,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    radix_bits: Any | None = None,
    descending: bool = False,
    exclusive_digit_prefix: Any = None,
    **kwargs: Any,
) -> Any:
    """Return each item's block-wide CUB radix rank.

    ``begin_bit``, ``end_bit``, and ``radix_bits`` are trace-time static
    specialization values. ``radix_bits`` defaults to four when neither it nor
    ``end_bit`` is supplied.

    ``exclusive_digit_prefix`` may be a ``ThreadData`` output that receives the
    per-digit exclusive prefix counters used by the rank operation.
    """
    base_payload = {
        "keys": keys,
        "args": args,
        "begin_bit": begin_bit,
        "end_bit": end_bit,
        "radix_bits": radix_bits,
        "descending": descending,
    }
    if exclusive_digit_prefix is not None:
        base_payload["exclusive_digit_prefix"] = exclusive_digit_prefix
    payload = merge_payload("radix_rank", base_payload, kwargs)
    return dispatch_primitive("radix_rank", kwargs=payload)


def radix_sort_keys_descending(
    keys: Any,
    /,
    *args: Any,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    temp_storage: Any = None,
    **kwargs: Any,
) -> Any:
    """Return block-wide radix-sorted keys in descending order."""
    return radix_sort_keys(
        keys,
        *args,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=True,
        temp_storage=temp_storage,
        **kwargs,
    )


def radix_sort_pairs_descending(
    keys: Any,
    values: Any,
    /,
    *args: Any,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    temp_storage: Any = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Return block-wide radix-sorted key/value pairs in descending order."""
    return radix_sort_pairs(
        keys,
        values,
        *args,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=True,
        temp_storage=temp_storage,
        **kwargs,
    )


def merge_sort_keys(
    keys: Any,
    /,
    *args: Any,
    descending: bool = False,
    valid_items: Any = None,
    oob_default: Any = None,
    temp_storage: Any = None,
    **kwargs: Any,
) -> Any:
    """Return block-wide CUB merge-sorted keys.

    ``valid_items`` and ``oob_default`` must be provided together for partial
    tiles so inactive items trace a typed sentinel value.
    """
    _validate_partial_tile_args(
        "merge_sort_keys",
        valid_items=valid_items,
        oob_default=oob_default,
    )
    structural_payload = {
        "keys": keys,
        "args": args,
        "descending": descending,
        "valid_items": valid_items,
        "oob_default": oob_default,
    }
    if temp_storage is not None:
        structural_payload["temp_storage"] = temp_storage
    payload = merge_payload(
        "merge_sort_keys",
        structural_payload,
        kwargs,
    )
    return dispatch_primitive("merge_sort_keys", kwargs=payload)


def merge_sort_pairs(
    keys: Any,
    values: Any,
    /,
    *args: Any,
    descending: bool = False,
    valid_items: Any = None,
    oob_default: Any = None,
    temp_storage: Any = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Return block-wide CUB merge-sorted key/value pairs.

    ``valid_items`` and ``oob_default`` must be provided together for partial
    tiles. The output preserves key/value associations after sorting.
    """
    _validate_partial_tile_args(
        "merge_sort_pairs",
        valid_items=valid_items,
        oob_default=oob_default,
    )
    structural_payload = {
        "keys": keys,
        "values": values,
        "args": args,
        "descending": descending,
        "valid_items": valid_items,
        "oob_default": oob_default,
    }
    if temp_storage is not None:
        structural_payload["temp_storage"] = temp_storage
    payload = merge_payload(
        "merge_sort_pairs",
        structural_payload,
        kwargs,
    )
    return dispatch_primitive("merge_sort_pairs", kwargs=payload)


def topk_max_keys(
    keys: Any,
    k: Any,
    /,
    *args: Any,
    num_valid: Any = None,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Return the block-wide CUB TopK largest keys.

    Lowers to CUB BlockTopK. Callers should consume only the first ``k``
    positions; their order follows CUB BlockTopK and is not sorted.
    """
    resolved_end_bit = _validate_sort_bits(
        "topk_max_keys",
        begin_bit=begin_bit,
        end_bit=end_bit,
    )
    payload = merge_payload(
        "topk_max_keys",
        {
            "keys": keys,
            "k": k,
            "args": args,
            "num_valid": num_valid,
            "begin_bit": begin_bit,
            "end_bit": resolved_end_bit,
            "descending": True,
        },
        kwargs,
    )
    return dispatch_primitive("topk_max_keys", kwargs=payload)


def topk_min_keys(
    keys: Any,
    k: Any,
    /,
    *args: Any,
    num_valid: Any = None,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Return the block-wide CUB TopK smallest keys.

    Callers should consume only the first ``k`` positions; their order follows
    CUB BlockTopK and is not sorted.
    """
    resolved_end_bit = _validate_sort_bits(
        "topk_min_keys",
        begin_bit=begin_bit,
        end_bit=end_bit,
    )
    payload = merge_payload(
        "topk_min_keys",
        {
            "keys": keys,
            "k": k,
            "args": args,
            "num_valid": num_valid,
            "begin_bit": begin_bit,
            "end_bit": resolved_end_bit,
            "descending": False,
        },
        kwargs,
    )
    return dispatch_primitive("topk_min_keys", kwargs=payload)


def topk_max_pairs(
    keys: Any,
    values: Any,
    k: Any,
    /,
    *args: Any,
    num_valid: Any = None,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Return the block-wide CUB TopK largest key/value pairs.

    Callers should consume only the first ``k`` positions; their order follows
    CUB BlockTopK and is not sorted. The value output follows the selected keys.
    """
    resolved_end_bit = _validate_sort_bits(
        "topk_max_pairs",
        begin_bit=begin_bit,
        end_bit=end_bit,
    )
    payload = merge_payload(
        "topk_max_pairs",
        {
            "keys": keys,
            "values": values,
            "k": k,
            "args": args,
            "num_valid": num_valid,
            "begin_bit": begin_bit,
            "end_bit": resolved_end_bit,
            "descending": True,
        },
        kwargs,
    )
    return dispatch_primitive("topk_max_pairs", kwargs=payload)


def topk_min_pairs(
    keys: Any,
    values: Any,
    k: Any,
    /,
    *args: Any,
    num_valid: Any = None,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Return the block-wide CUB TopK smallest key/value pairs.

    Callers should consume only the first ``k`` positions; their order follows
    CUB BlockTopK and is not sorted. The value output follows the selected keys.
    """
    resolved_end_bit = _validate_sort_bits(
        "topk_min_pairs",
        begin_bit=begin_bit,
        end_bit=end_bit,
    )
    payload = merge_payload(
        "topk_min_pairs",
        {
            "keys": keys,
            "values": values,
            "k": k,
            "args": args,
            "num_valid": num_valid,
            "begin_bit": begin_bit,
            "end_bit": resolved_end_bit,
            "descending": False,
        },
        kwargs,
    )
    return dispatch_primitive("topk_min_pairs", kwargs=payload)


register_primitive_impl("radix_sort_keys", impl=_radix_sort_keys_provider)
register_primitive_impl("radix_sort_pairs", impl=_radix_sort_pairs_provider)
register_primitive_impl("radix_rank", impl=_radix_rank_provider)
register_primitive_impl("merge_sort_keys", impl=_merge_sort_keys_provider)
register_primitive_impl("merge_sort_pairs", impl=_merge_sort_pairs_provider)
register_primitive_impl("topk_max_keys", impl=_topk_keys_provider)
register_primitive_impl("topk_min_keys", impl=_topk_keys_provider)
register_primitive_impl("topk_max_pairs", impl=_topk_pairs_provider)
register_primitive_impl("topk_min_pairs", impl=_topk_pairs_provider)
