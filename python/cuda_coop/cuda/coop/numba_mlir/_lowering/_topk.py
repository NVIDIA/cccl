# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Adapt shared TopK specializations to Numba-callable block providers.

The shared core supplies CUB's checked count wrapper and array signature.
These factories attach compiler types and pass the shared scratch pointer
as the first call argument.
Group rewriting supplies writable result copies before calling the provider.
"""

from __future__ import annotations

from cuda.coop._core import SynchronizationScope
from cuda.coop._core.block.topk import make_block_topk_specialization

from .._compiler._operations import (
    StorageABI,
    factory_operation,
    register_factory,
)
from .._compiler._parameters import (
    _validate_common_numeric_dtype,
    normalize_dim_param,
)
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
    """Bind TopK payload types, launch shape, and counts to one provider.

    Validate key and optional value types independently, then pass core types
    and count bindings to the shared specialization builder. The builder
    selects the full- or partial-tile method from the valid-count binding.
    Its wrapper keeps runtime counts wide until it validates them.

    Use the calling factory's registered storage and synchronization metadata
    when materializing the specialization. Return an invocable that modifies
    its array operands in place; the public rewrite owns their initial copies.
    """

    adapter = NumbaMlirCoreAdapter()
    key_dtype = _validate_common_numeric_dtype(
        key_dtype, operation="topk", parameter="keys"
    )
    if value_dtype is not None:
        value_dtype = _validate_common_numeric_dtype(
            value_dtype, operation="topk", parameter="values"
        )
    specialization = make_block_topk_specialization(
        key_dtype=adapter.core_dtype(key_dtype),
        value_dtype=None
        if value_dtype is None
        else adapter.core_dtype(value_dtype),
        block_dim=tuple(normalize_dim_param(threads_per_block)),
        items_per_thread=items_per_thread,
        selection=selection,
        k=k,
        num_valid=num_valid,
    )
    metadata = factory_operation(factory)
    assert metadata is not None
    specialization = adapter.materialize(
        specialization.specialization,
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
