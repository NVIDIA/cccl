# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from typing import TYPE_CHECKING, Tuple, Union

from .. import _require_runtime

_require_runtime()

from numba_cuda_mlir import types

from cuda.coop._core.block import ArgumentBinding, make_block_topk_spec

from .._common import (
    _validate_common_integer_key_dtype,
    _validate_common_numeric_dtype,
    dim3,
    normalize_dim_param,
    normalize_dtype_param,
    resolve_threads_per_block_alias,
)
from .._core_adapter import NumbaMlirCoreAdapter
from .._types import make_invocable_from_specialization, numba_type_to_cpp

if TYPE_CHECKING:
    import numpy as np


def _normalize_topk_args(dtype, threads_per_block, items_per_thread, select):
    if dtype is None:
        raise ValueError("dtype must be provided")
    if threads_per_block is None:
        raise ValueError("threads_per_block must be provided")
    if select not in {"max", "min"}:
        raise ValueError("select must be either 'max' or 'min'")
    try:
        items_per_thread = int(items_per_thread)
    except (TypeError, ValueError) as e:
        raise ValueError("items_per_thread must be an integer") from e
    if items_per_thread < 1:
        raise ValueError("items_per_thread must be greater than or equal to 1")

    dim = normalize_dim_param(threads_per_block)
    if dim[1] != 1 or dim[2] != 1:
        raise ValueError("BlockTopK currently supports only 1D block dimensions")

    dtype = normalize_dtype_param(dtype)
    if numba_type_to_cpp(dtype) == "storage_t":
        raise TypeError("TopK does not support user-defined key dtypes yet.")
    return dim, dtype, items_per_thread


def _topk_binding(value):
    if value is True:
        return ArgumentBinding.runtime()
    return ArgumentBinding.static(int(value))


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

    core_spec = make_block_topk_spec(
        key_dtype=key_dtype,
        value_dtype=value_dtype,
        block_dim=tuple(dim),
        items_per_thread=items_per_thread,
        selection=select,
        num_valid=(
            ArgumentBinding.omitted() if num_valid is None else _topk_binding(num_valid)
        ),
        begin_bit=(
            ArgumentBinding.omitted() if begin_bit is None else _topk_binding(begin_bit)
        ),
        end_bit=(
            ArgumentBinding.omitted() if end_bit is None else _topk_binding(end_bit)
        ),
    )
    specialization = NumbaMlirCoreAdapter().materialize(core_spec.specialization)
    return make_invocable_from_specialization(specialization)


def topk_max_keys(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    num_valid=None,
    begin_bit=None,
    end_bit=None,
    dim=None,
):
    """Build a block-wide top-k largest-key selection invocable.

    The first ``min(k, num_valid)`` positions are defined when ``num_valid`` is
    supplied at runtime or as a factory constant. Ties across the ``k`` boundary
    are not expanded.
    """
    return _topk(
        key_dtype=dtype,
        threads_per_block=resolve_threads_per_block_alias(threads_per_block, dim),
        items_per_thread=items_per_thread,
        select="max",
        num_valid=num_valid,
        begin_bit=begin_bit,
        end_bit=end_bit,
    )


def topk_min_keys(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    num_valid=None,
    begin_bit=None,
    end_bit=None,
    dim=None,
):
    """Build a block-wide top-k smallest-key selection invocable."""
    return _topk(
        key_dtype=dtype,
        threads_per_block=resolve_threads_per_block_alias(threads_per_block, dim),
        items_per_thread=items_per_thread,
        select="min",
        num_valid=num_valid,
        begin_bit=begin_bit,
        end_bit=end_bit,
    )


def _common_topk_keys(factory, operation, **kwargs):
    """Materialize one portable keys-only specialization after dtype inference."""

    kwargs = dict(kwargs)
    dtype = _validate_common_integer_key_dtype(kwargs["dtype"], operation=operation)
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
    key_name = "key_dtype" if "key_dtype" in kwargs else "keys"
    value_name = "value_dtype" if "value_dtype" in kwargs else "values"
    key_dtype = _validate_common_integer_key_dtype(
        kwargs[key_name], operation=operation
    )
    kwargs[key_name] = key_dtype
    kwargs[value_name] = _validate_common_numeric_dtype(
        kwargs[value_name], operation=operation, parameter="value"
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
        key_dtype = kwargs.get("key_dtype")
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
    keys=None,
    values=None,
    threads_per_block=None,
    items_per_thread=1,
    num_valid=None,
    begin_bit=None,
    end_bit=None,
    key_dtype=None,
    value_dtype=None,
    dim=None,
):
    """Build a block-wide top-k largest key/value-pair selection invocable."""
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
    return _topk(
        key_dtype=keys,
        value_dtype=value_dtype,
        threads_per_block=resolve_threads_per_block_alias(threads_per_block, dim),
        items_per_thread=items_per_thread,
        select="max",
        num_valid=num_valid,
        begin_bit=begin_bit,
        end_bit=end_bit,
    )


def topk_min_pairs(
    keys=None,
    values=None,
    threads_per_block=None,
    items_per_thread=1,
    num_valid=None,
    begin_bit=None,
    end_bit=None,
    key_dtype=None,
    value_dtype=None,
    dim=None,
):
    """Build a block-wide top-k smallest key/value-pair selection invocable."""
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
    return _topk(
        key_dtype=keys,
        value_dtype=value_dtype,
        threads_per_block=resolve_threads_per_block_alias(threads_per_block, dim),
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
