# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUB block TopK provider factories."""

from __future__ import annotations

from cuda.coop._core import SynchronizationScope
from cuda.coop._core.block.topk import make_block_topk_spec

from .._compiler._operations import StorageABI, factory_operation, register_factory
from .._compiler._parameters import _validate_common_numeric_dtype, normalize_dim_param
from .._types import make_invocable_from_specialization
from ._core import NumbaMlirCoreAdapter


def _topk(
    factory,
    *,
    key_dtype,
    threads_per_block,
    items_per_thread,
    k,
    selection="max",
    value_dtype=None,
    num_valid=None,
):
    adapter = NumbaMlirCoreAdapter()
    key_dtype = _validate_common_numeric_dtype(
        key_dtype, operation="topk", parameter="keys"
    )
    if value_dtype is not None:
        value_dtype = _validate_common_numeric_dtype(
            value_dtype, operation="topk", parameter="values"
        )
    spec = make_block_topk_spec(
        key_dtype=adapter.core_dtype(key_dtype),
        value_dtype=None if value_dtype is None else adapter.core_dtype(value_dtype),
        block_dim=tuple(normalize_dim_param(threads_per_block)),
        items_per_thread=items_per_thread,
        selection=selection,
        k=k,
        num_valid=num_valid,
    )
    metadata = factory_operation(factory)
    specialization = adapter.materialize(
        spec.specialization,
        storage_abi=metadata.storage_abi,
        execution_scope=metadata.execution_scope,
        synchronization_scope=metadata.synchronization_scope,
    )
    return make_invocable_from_specialization(specialization)


def topk_keys(**kwargs):
    return _topk(topk_keys, **kwargs)


def topk_pairs(**kwargs):
    return _topk(topk_pairs, **kwargs)


for _factory in (topk_keys, topk_pairs):
    register_factory(
        _factory,
        operation=_factory.__name__,
        namespace="block",
        storage_abi=StorageABI.LEADING_POINTER,
        execution_scope=SynchronizationScope.BLOCK,
        synchronization_scope=SynchronizationScope.BLOCK,
    )
del _factory
