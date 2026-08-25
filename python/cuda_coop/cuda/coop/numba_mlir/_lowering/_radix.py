# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Radix rank and sort provider lowering for Numba-CUDA-MLIR.

All radix factories share one semantic module. Runtime bit controls and launch-dimension inference are resolved before provider construction.
"""

import operator
from typing import TYPE_CHECKING, Tuple, Union

import numpy as np

from .._compiler._activation import _require_runtime

_require_runtime()

from cuda.coop._core.block import (
    BlockRadixSortBitPolicy,
    make_block_radix_rank_spec,
    make_block_radix_sort_spec,
    resolve_static_radix_end_bit,
)

from .._compiler._parameters import (
    _validate_common_integer_key_dtype,
    _validate_common_numeric_dtype,
    dim3,
    normalize_dim_param,
    normalize_dtype_param,
)
from .._types import make_invocable_from_specialization
from ._core import (
    NumbaMlirArrayInputTransform,
    NumbaMlirCoreAdapter,
)

if TYPE_CHECKING:
    from numba_cuda_mlir import types


_SIGNED_CUB_KEY_DTYPES = {
    "int32": "uint32",
    "int64": "uint64",
}
_SIGNED_INPUT_TRANSFORMS = {
    "int32": "(static_cast<unsigned int>({value}) ^ 0x80000000u)",
    "int64": ("(static_cast<unsigned long long>({value}) ^ 0x8000000000000000ull)"),
}


def _static_index(name, value):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


def _static_bool(name, value):
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def radix_rank(
    dtype: Union[str, type, "np.dtype", "types.Type"],
    threads_per_block: Union[int, Tuple[int, int], Tuple[int, int, int], dim3] = None,
    items_per_thread: int = 1,
    begin_bit: int = 0,
    end_bit: int = None,
    descending: bool = False,
    exclusive_digit_prefix=None,
):
    """Build a block-wide radix-rank invocable.

    The invocable wraps CUB ``BlockRadixRank`` and writes each key's rank within
    the block tile. ``begin_bit`` and ``end_bit`` select the radix window, and
    ``descending`` reverses digit ordering for descending sort pipelines.
    ``exclusive_digit_prefix`` may be provided to select the overload that also
    writes each per-thread digit-bin prefix.
    """
    dim = normalize_dim_param(threads_per_block)
    dtype = normalize_dtype_param(dtype)
    items_per_thread = _static_index("items_per_thread", items_per_thread)
    begin_bit = _static_index("begin_bit", begin_bit)
    if end_bit is not None:
        end_bit = _static_index("end_bit", end_bit)
    descending = _static_bool("descending", descending)

    bitwidth = getattr(dtype, "bitwidth", None)
    if bitwidth is not None:
        bitwidth = int(bitwidth)
    end_bit = resolve_static_radix_end_bit(
        begin_bit=begin_bit,
        end_bit=end_bit,
        bit_width=bitwidth,
        default_radix_bits=4,
        clamp_default=False,
    )
    dtype_name = str(dtype)
    cub_dtype = dtype
    input_transforms = None
    if dtype_name in _SIGNED_CUB_KEY_DTYPES:
        from numba_cuda_mlir import types as numba_mlir_types

        cub_dtype = getattr(numba_mlir_types, _SIGNED_CUB_KEY_DTYPES[dtype_name])
        input_transforms = {
            "keys": NumbaMlirArrayInputTransform(
                source_dtype=dtype,
                cpp_expression=_SIGNED_INPUT_TRANSFORMS[dtype_name],
            )
        }

    core_spec = make_block_radix_rank_spec(
        key_dtype=cub_dtype,
        block_dim=tuple(dim),
        items_per_thread=items_per_thread,
        begin_bit=begin_bit,
        end_bit=end_bit,
        key_bit_width=bitwidth,
        descending=descending,
        with_exclusive_digit_prefix=exclusive_digit_prefix is not None,
    )
    specialization = NumbaMlirCoreAdapter(
        input_transforms=input_transforms
    ).materialize(core_spec.specialization)
    return make_invocable_from_specialization(specialization)


def _common_radix_rank(**kwargs):
    """Materialize one portable rank specialization after dtype inference."""

    kwargs = dict(kwargs)
    kwargs["dtype"] = _validate_common_integer_key_dtype(
        kwargs["dtype"], operation="radix_rank"
    )
    return radix_rank(**kwargs)


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
    items_per_thread = _static_index("items_per_thread", items_per_thread)
    descending = _static_bool("descending", descending)
    blocked_to_striped = _static_bool("blocked_to_striped", blocked_to_striped)
    if begin_bit is not None:
        begin_bit = _static_index("begin_bit", begin_bit)
    if end_bit is not None:
        end_bit = _static_index("end_bit", end_bit)

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
        # Group lowering may either omit the bit range or pass it at runtime.
        bit_policy=BlockRadixSortBitPolicy.BOTH,
    )
    specialization = NumbaMlirCoreAdapter().materialize(core_spec.specialization)
    return make_invocable_from_specialization(specialization)


def radix_sort_keys(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    begin_bit=None,
    end_bit=None,
    blocked_to_striped=False,
):
    """Build an ascending block-wide radix-sort invocable.

    ``begin_bit`` and ``end_bit`` select the runtime bit-range overload, and
    ``blocked_to_striped`` selects CUB's blocked-to-striped output method.
    """
    return _radix_sort(
        key_dtype=dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        descending=False,
        begin_bit=begin_bit,
        end_bit=end_bit,
        blocked_to_striped=blocked_to_striped,
    )


def radix_sort_keys_descending(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    begin_bit=None,
    end_bit=None,
    blocked_to_striped=False,
):
    """Build a descending block-wide radix-sort invocable."""
    return _radix_sort(
        key_dtype=dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        descending=True,
        begin_bit=begin_bit,
        end_bit=end_bit,
        blocked_to_striped=blocked_to_striped,
    )


def radix_sort_pairs(
    key_dtype,
    value_dtype,
    threads_per_block=None,
    items_per_thread=1,
    begin_bit=None,
    end_bit=None,
    blocked_to_striped=False,
):
    """Build an ascending block-wide key/value radix-sort invocable."""
    return _radix_sort(
        key_dtype=key_dtype,
        value_dtype=value_dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        descending=False,
        begin_bit=begin_bit,
        end_bit=end_bit,
        blocked_to_striped=blocked_to_striped,
    )


def radix_sort_pairs_descending(
    key_dtype,
    value_dtype,
    threads_per_block=None,
    items_per_thread=1,
    begin_bit=None,
    end_bit=None,
    blocked_to_striped=False,
):
    """Build a descending block-wide key/value radix-sort invocable."""
    return _radix_sort(
        key_dtype=key_dtype,
        value_dtype=value_dtype,
        threads_per_block=threads_per_block,
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
    key_dtype = _validate_common_integer_key_dtype(
        kwargs["key_dtype"], operation="radix_sort_pairs"
    )
    kwargs["key_dtype"] = key_dtype
    kwargs["value_dtype"] = _validate_common_numeric_dtype(
        kwargs["value_dtype"], operation="radix_sort_pairs", parameter="value"
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
