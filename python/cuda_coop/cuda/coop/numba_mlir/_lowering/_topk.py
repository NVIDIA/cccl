# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""TopK provider lowering for Numba-CUDA-MLIR.

This module owns key and pair provider materialization.  The compiler planner
resolves fresh-result payloads, launch facts, and runtime bit bounds first.
"""

import operator
from typing import Tuple, Union

import numpy as np

from .._compiler._activation import _require_runtime

_require_runtime()

from numba_cuda_mlir import types

from cuda.coop._core.block import ArgumentBinding, BindingKind, make_block_topk_spec

from .._compiler._parameters import (
    _validate_common_integer_key_dtype,
    _validate_common_numeric_dtype,
    dim3,
    normalize_dim_param,
    normalize_dtype_param,
)
from .._types import make_invocable_from_specialization, numba_type_to_cpp
from ._core import NumbaMlirCoreAdapter


def _normalize_topk_args(dtype, threads_per_block, items_per_thread, select):
    if dtype is None:
        raise ValueError("dtype must be provided")
    if threads_per_block is None:
        raise ValueError("threads_per_block must be provided")
    if select not in {"max", "min"}:
        raise ValueError("select must be either 'max' or 'min'")
    if isinstance(items_per_thread, (bool, np.bool_)):
        raise TypeError("items_per_thread must be an integer")
    try:
        items_per_thread = operator.index(items_per_thread)
    except TypeError as e:
        raise TypeError("items_per_thread must be an integer") from e
    if items_per_thread < 1:
        raise ValueError("items_per_thread must be greater than or equal to 1")

    dim = normalize_dim_param(threads_per_block)
    if dim[1] != 1 or dim[2] != 1:
        raise ValueError("BlockTopK currently supports only 1D block dimensions")
    if dim[0] > 1024:
        raise ValueError("BlockTopK block thread count must be <= 1024")

    dtype = normalize_dtype_param(dtype)
    if numba_type_to_cpp(dtype) == "storage_t":
        raise TypeError("TopK does not support user-defined key dtypes yet.")
    return dim, dtype, items_per_thread


def _topk_binding(value, *, name):
    if isinstance(value, ArgumentBinding):
        if value.kind is not BindingKind.STATIC:
            return value
        static_value = value.value
        if isinstance(static_value, (bool, np.bool_)):
            raise TypeError(f"static {name} must be an integer")
        try:
            operator.index(static_value)
        except TypeError as exc:
            raise TypeError(f"static {name} must be an integer") from exc
        return value
    if value is True:
        return ArgumentBinding.runtime()
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer or the runtime sentinel True")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(
            f"{name} must be an integer or the runtime sentinel True"
        ) from exc
    return ArgumentBinding.static(value)


def _topk(
    key_dtype: Union[str, type, "np.dtype", "types.Type"],
    threads_per_block: Union[int, Tuple[int, int], Tuple[int, int, int], dim3],
    items_per_thread: int,
    select: str,
    value_dtype: Union[str, type, "np.dtype", "types.Type", None] = None,
    num_valid=None,
    begin_bit: int = None,
    end_bit: int = None,
):
    """Build a Numba-CUDA-MLIR CUB block TopK invocable."""
    if (begin_bit is None) != (end_bit is None):
        raise ValueError("begin_bit and end_bit must be provided together")

    dim, key_dtype, items_per_thread = _normalize_topk_args(
        key_dtype, threads_per_block, items_per_thread, select
    )
    if value_dtype is not None:
        value_dtype = normalize_dtype_param(value_dtype)
        if numba_type_to_cpp(value_dtype) == "storage_t":
            raise TypeError("TopK does not support user-defined value dtypes yet.")

    num_valid_binding = (
        ArgumentBinding.omitted()
        if num_valid is None
        else _topk_binding(num_valid, name="num_valid")
    )
    begin_bit_binding = (
        ArgumentBinding.omitted()
        if begin_bit is None
        else _topk_binding(begin_bit, name="begin_bit")
    )
    end_bit_binding = (
        ArgumentBinding.omitted()
        if end_bit is None
        else _topk_binding(end_bit, name="end_bit")
    )
    tile_size = dim[0] * items_per_thread
    if num_valid_binding.kind is BindingKind.STATIC and not (
        1 <= num_valid_binding.value <= tile_size
    ):
        raise ValueError(f"num_valid must be in [1, {tile_size}]")
    key_width = int(key_dtype.bitwidth)
    if (
        begin_bit_binding.kind is BindingKind.STATIC
        and not 0 <= begin_bit_binding.value < key_width
    ):
        raise ValueError(f"begin_bit must be in [0, {key_width})")
    if (
        end_bit_binding.kind is BindingKind.STATIC
        and not 0 < end_bit_binding.value <= key_width
    ):
        raise ValueError(f"end_bit must be in (0, {key_width}]")
    if (
        begin_bit_binding.kind is BindingKind.STATIC
        and end_bit_binding.kind is BindingKind.STATIC
        and end_bit_binding.value <= begin_bit_binding.value
    ):
        raise ValueError("end_bit must exceed begin_bit")

    core_spec = make_block_topk_spec(
        key_dtype=key_dtype,
        value_dtype=value_dtype,
        block_dim=tuple(dim),
        items_per_thread=items_per_thread,
        selection=select,
        num_valid=num_valid_binding,
        begin_bit=begin_bit_binding,
        end_bit=end_bit_binding,
    )
    specialization = NumbaMlirCoreAdapter().materialize(core_spec.specialization)
    return make_invocable_from_specialization(specialization)


def topk_max_keys(
    dtype,
    threads_per_block,
    items_per_thread=1,
    num_valid=None,
    begin_bit=None,
    end_bit=None,
):
    """Build a block-wide top-k largest-key selection invocable.

    The first ``min(k, num_valid)`` positions are defined when ``num_valid`` is
    supplied at runtime or as a factory constant. Ties across the ``k`` boundary
    are not expanded.
    """
    return _topk(
        key_dtype=dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        select="max",
        num_valid=num_valid,
        begin_bit=begin_bit,
        end_bit=end_bit,
    )


def topk_min_keys(
    dtype,
    threads_per_block,
    items_per_thread=1,
    num_valid=None,
    begin_bit=None,
    end_bit=None,
):
    """Build a block-wide top-k smallest-key selection invocable."""
    return _topk(
        key_dtype=dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        select="min",
        num_valid=num_valid,
        begin_bit=begin_bit,
        end_bit=end_bit,
    )


def _common_topk_keys(factory, operation, **kwargs):
    """Materialize one portable keys-only specialization after dtype inference."""

    kwargs = dict(kwargs)
    dtype = _validate_common_integer_key_dtype(
        kwargs["dtype"], operation=operation, parameter="key"
    )
    kwargs["dtype"] = dtype

    begin_bit = kwargs.get("begin_bit")
    end_bit = kwargs.get("end_bit")
    if begin_bit is not None or end_bit is not None:
        kwargs["begin_bit"] = 0 if begin_bit is None else begin_bit
        kwargs["end_bit"] = int(dtype.bitwidth) if end_bit is None else end_bit
    return factory(**kwargs)


def _common_topk_max_keys(**kwargs):
    """Materialize one portable largest-keys TopK specialization."""

    return _common_topk_keys(topk_max_keys, "topk_max_keys", **kwargs)


def _common_topk_min_keys(**kwargs):
    """Materialize one portable smallest-keys TopK specialization."""

    return _common_topk_keys(topk_min_keys, "topk_min_keys", **kwargs)


def _common_topk_pairs(factory, operation, **kwargs):
    """Materialize one portable pair specialization after dtype inference."""

    kwargs = dict(kwargs)
    key_dtype = _validate_common_integer_key_dtype(
        kwargs["keys"], operation=operation, parameter="key"
    )
    kwargs["keys"] = key_dtype
    kwargs["values"] = _validate_common_numeric_dtype(
        kwargs["values"], operation=operation, parameter="value"
    )

    begin_bit = kwargs.get("begin_bit")
    end_bit = kwargs.get("end_bit")
    if begin_bit is not None or end_bit is not None:
        kwargs["begin_bit"] = 0 if begin_bit is None else begin_bit
        kwargs["end_bit"] = int(key_dtype.bitwidth) if end_bit is None else end_bit
    return factory(**kwargs)


def _common_topk_max_pairs(**kwargs):
    """Materialize one portable largest-pairs TopK specialization."""

    return _common_topk_pairs(topk_max_pairs, "topk_max_pairs", **kwargs)


def _common_topk_min_pairs(**kwargs):
    """Materialize one portable smallest-pairs TopK specialization."""

    return _common_topk_pairs(topk_min_pairs, "topk_min_pairs", **kwargs)


def _qualified_group_topk(factory, **kwargs):
    """Materialize one qualified group-first specialization after inference."""

    kwargs = dict(kwargs)
    key_dtype = kwargs.get("dtype")
    if key_dtype is None:
        key_dtype = kwargs.get("keys")
    key_dtype = normalize_dtype_param(key_dtype)

    begin_bit = kwargs.get("begin_bit")
    end_bit = kwargs.get("end_bit")
    if begin_bit is not None or end_bit is not None:
        kwargs["begin_bit"] = 0 if begin_bit is None else begin_bit
        kwargs["end_bit"] = int(key_dtype.bitwidth) if end_bit is None else end_bit
    return factory(**kwargs)


def _qualified_group_topk_max_keys(**kwargs):
    """Materialize one qualified group-first largest-keys specialization."""

    return _qualified_group_topk(topk_max_keys, **kwargs)


def _qualified_group_topk_min_keys(**kwargs):
    """Materialize one qualified group-first smallest-keys specialization."""

    return _qualified_group_topk(topk_min_keys, **kwargs)


def topk_max_pairs(
    keys,
    values,
    threads_per_block,
    items_per_thread=1,
    num_valid=None,
    begin_bit=None,
    end_bit=None,
):
    """Build a block-wide top-k largest key/value-pair selection invocable."""
    return _topk(
        key_dtype=keys,
        value_dtype=values,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        select="max",
        num_valid=num_valid,
        begin_bit=begin_bit,
        end_bit=end_bit,
    )


def topk_min_pairs(
    keys,
    values,
    threads_per_block,
    items_per_thread=1,
    num_valid=None,
    begin_bit=None,
    end_bit=None,
):
    """Build a block-wide top-k smallest key/value-pair selection invocable."""
    return _topk(
        key_dtype=keys,
        value_dtype=values,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        select="min",
        num_valid=num_valid,
        begin_bit=begin_bit,
        end_bit=end_bit,
    )


def _qualified_group_topk_max_pairs(**kwargs):
    """Materialize one qualified group-first largest-pairs specialization."""

    return _qualified_group_topk(topk_max_pairs, **kwargs)


def _qualified_group_topk_min_pairs(**kwargs):
    """Materialize one qualified group-first smallest-pairs specialization."""

    return _qualified_group_topk(topk_min_pairs, **kwargs)
