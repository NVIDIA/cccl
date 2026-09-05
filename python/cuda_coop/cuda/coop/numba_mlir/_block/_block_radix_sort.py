# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from typing import TYPE_CHECKING, Tuple, Union

from .. import _require_runtime

_require_runtime()

from cuda.coop._core.block import (
    BlockRadixSortBitPolicy,
    make_block_radix_sort_spec,
    resolve_static_radix_end_bit,
)

from .._common import (
    _validate_common_integer_key_dtype,
    _validate_common_numeric_dtype,
    dim3,
    normalize_dim_param,
    normalize_dtype_param,
    resolve_threads_per_block_alias,
)
from .._core_adapter import NumbaMlirCoreAdapter
from .._types import make_invocable_from_specialization

if TYPE_CHECKING:
    import numpy as np
    from numba_cuda_mlir import types


def _radix_sort(
    key_dtype: Union[str, type, "np.dtype", "types.Type"],
    threads_per_block: Union[int, Tuple[int, int], Tuple[int, int, int], dim3],
    items_per_thread: int,
    descending: bool,
    value_dtype: Union[str, type, "np.dtype", "types.Type", None] = None,
    begin_bit: int = None,
    end_bit: int = None,
    blocked_to_striped: bool = False,
) -> object:
    if key_dtype is None:
        raise ValueError("key dtype must be provided")
    if threads_per_block is None:
        raise ValueError("threads_per_block must be provided")
    dim = normalize_dim_param(threads_per_block)
    key_dtype = normalize_dtype_param(key_dtype)
    if value_dtype is not None:
        value_dtype = normalize_dtype_param(value_dtype)

    key_bit_width = getattr(key_dtype, "bitwidth", None)
    core_spec = make_block_radix_sort_spec(
        key_dtype=key_dtype,
        value_dtype=value_dtype,
        block_dim=tuple(dim),
        items_per_thread=items_per_thread,
        descending=descending,
        blocked_to_striped=blocked_to_striped,
        begin_bit=begin_bit,
        end_bit=end_bit,
        key_bit_width=key_bit_width,
        # The MLIR invocable historically exposes both CUB overloads, so calls
        # may either omit the bit range or pass it at runtime.
        bit_policy=BlockRadixSortBitPolicy.BOTH,
    )
    specialization = NumbaMlirCoreAdapter().materialize(core_spec.specialization)
    return make_invocable_from_specialization(specialization)


def radix_sort_keys(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    value_dtype=None,
    begin_bit=None,
    end_bit=None,
    blocked_to_striped=False,
    dim=None,
):
    """Build an ascending block-wide radix-sort invocable.

    ``value_dtype`` enables the CUB key/value overload while preserving the
    ``radix_sort_keys`` factory spelling.
    ``begin_bit`` and ``end_bit`` select the runtime bit-range overload, and
    ``blocked_to_striped`` selects CUB's blocked-to-striped output method.
    """
    return _radix_sort(
        key_dtype=dtype,
        threads_per_block=resolve_threads_per_block_alias(threads_per_block, dim),
        items_per_thread=items_per_thread,
        descending=False,
        value_dtype=value_dtype,
        begin_bit=begin_bit,
        end_bit=end_bit,
        blocked_to_striped=blocked_to_striped,
    )


def radix_sort_keys_descending(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    value_dtype=None,
    begin_bit=None,
    end_bit=None,
    blocked_to_striped=False,
    dim=None,
):
    """Build a descending block-wide radix-sort invocable."""
    return _radix_sort(
        key_dtype=dtype,
        threads_per_block=resolve_threads_per_block_alias(threads_per_block, dim),
        items_per_thread=items_per_thread,
        descending=True,
        value_dtype=value_dtype,
        begin_bit=begin_bit,
        end_bit=end_bit,
        blocked_to_striped=blocked_to_striped,
    )


def radix_sort_pairs(
    keys=None,
    values=None,
    threads_per_block=None,
    items_per_thread=1,
    begin_bit=None,
    end_bit=None,
    blocked_to_striped=False,
    key_dtype=None,
    value_dtype=None,
    dim=None,
):
    """Build an ascending block-wide key/value radix-sort invocable."""
    if key_dtype is not None:
        if keys is not None:
            raise ValueError("keys and key_dtype cannot both be provided")
        keys = key_dtype
    if value_dtype is None:
        value_dtype = values
    elif values is not None:
        raise ValueError("values and value_dtype cannot both be provided")
    if value_dtype is None:
        raise ValueError("value_dtype must be provided")
    return _radix_sort(
        key_dtype=keys,
        value_dtype=value_dtype,
        threads_per_block=resolve_threads_per_block_alias(threads_per_block, dim),
        items_per_thread=items_per_thread,
        descending=False,
        begin_bit=begin_bit,
        end_bit=end_bit,
        blocked_to_striped=blocked_to_striped,
    )


def radix_sort_pairs_descending(
    keys=None,
    values=None,
    threads_per_block=None,
    items_per_thread=1,
    begin_bit=None,
    end_bit=None,
    blocked_to_striped=False,
    key_dtype=None,
    value_dtype=None,
    dim=None,
):
    """Build a descending block-wide key/value radix-sort invocable."""
    if key_dtype is not None:
        if keys is not None:
            raise ValueError("keys and key_dtype cannot both be provided")
        keys = key_dtype
    if value_dtype is None:
        value_dtype = values
    elif values is not None:
        raise ValueError("values and value_dtype cannot both be provided")
    if value_dtype is None:
        raise ValueError("value_dtype must be provided")
    return _radix_sort(
        key_dtype=keys,
        value_dtype=value_dtype,
        threads_per_block=resolve_threads_per_block_alias(threads_per_block, dim),
        items_per_thread=items_per_thread,
        descending=True,
        begin_bit=begin_bit,
        end_bit=end_bit,
        blocked_to_striped=blocked_to_striped,
    )


def _common_radix_sort_keys(**kwargs):
    """Materialize one portable keys-only specialization after dtype inference."""

    kwargs = dict(kwargs)
    dtype = _validate_common_integer_key_dtype(
        kwargs["dtype"], operation="radix_sort_keys"
    )
    kwargs["dtype"] = dtype
    descending = kwargs.pop("descending", False)

    begin_bit = kwargs.get("begin_bit")
    end_bit = kwargs.get("end_bit")
    if begin_bit is not None or end_bit is not None:
        begin_bit = 0 if begin_bit is None else begin_bit
        try:
            end_bit = resolve_static_radix_end_bit(
                begin_bit=begin_bit,
                end_bit=end_bit,
                bit_width=int(dtype.bitwidth),
                default_to_bit_width=True,
            )
        except ValueError as exc:
            raise ValueError(f"cuda.coop.radix_sort_keys {exc}") from exc
        if begin_bit == 0 and end_bit == int(dtype.bitwidth):
            kwargs.pop("begin_bit", None)
            kwargs.pop("end_bit", None)
        else:
            kwargs["begin_bit"] = begin_bit
            kwargs["end_bit"] = end_bit

    factory = radix_sort_keys_descending if descending else radix_sort_keys
    return factory(**kwargs)


def _common_radix_sort_pairs(**kwargs):
    """Materialize one portable key/value specialization after dtype inference."""

    kwargs = dict(kwargs)
    key_name = "key_dtype" if "key_dtype" in kwargs else "keys"
    value_name = "value_dtype" if "value_dtype" in kwargs else "values"
    key_dtype = _validate_common_integer_key_dtype(
        kwargs[key_name], operation="radix_sort_pairs"
    )
    kwargs[key_name] = key_dtype
    kwargs[value_name] = _validate_common_numeric_dtype(
        kwargs[value_name], operation="radix_sort_pairs", parameter="value"
    )
    descending = kwargs.pop("descending", False)

    begin_bit = kwargs.get("begin_bit")
    end_bit = kwargs.get("end_bit")
    if begin_bit is not None or end_bit is not None:
        begin_bit = 0 if begin_bit is None else begin_bit
        try:
            end_bit = resolve_static_radix_end_bit(
                begin_bit=begin_bit,
                end_bit=end_bit,
                bit_width=int(key_dtype.bitwidth),
                default_to_bit_width=True,
            )
        except ValueError as exc:
            raise ValueError(f"cuda.coop.radix_sort_pairs {exc}") from exc
        if begin_bit == 0 and end_bit == int(key_dtype.bitwidth):
            kwargs.pop("begin_bit", None)
            kwargs.pop("end_bit", None)
        else:
            kwargs["begin_bit"] = begin_bit
            kwargs["end_bit"] = end_bit

    factory = radix_sort_pairs_descending if descending else radix_sort_pairs
    return factory(**kwargs)
