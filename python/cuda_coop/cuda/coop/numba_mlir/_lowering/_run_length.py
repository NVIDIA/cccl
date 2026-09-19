# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUB complete-operation Run Length Decode factories."""

from cuda.coop._core import SynchronizationScope
from cuda.coop._core.block.run_length import make_block_run_length_decode_spec

from .._compiler._operations import StorageABI, factory_operation, register_factory
from .._compiler._parameters import normalize_dim_param
from .._types import make_invocable_from_specialization
from ._core import NumbaMlirCoreAdapter


def _decode(
    factory,
    *,
    item_dtype,
    run_length_dtype,
    decoded_offset_dtype,
    control_dtype,
    threads_per_block,
    runs_per_thread,
    decoded_items_per_thread,
    offset,
    relative_offsets=False,
):
    adapter = NumbaMlirCoreAdapter()
    spec = make_block_run_length_decode_spec(
        item_dtype=adapter.core_dtype(item_dtype),
        run_length_dtype=adapter.core_dtype(run_length_dtype),
        decoded_offset_dtype=adapter.core_dtype(decoded_offset_dtype),
        control_dtype=adapter.core_dtype(control_dtype),
        block_dim=tuple(normalize_dim_param(threads_per_block)),
        runs_per_thread=runs_per_thread,
        decoded_items_per_thread=decoded_items_per_thread,
        offset=offset,
        bulk=factory is not run_length_decode,
        relative_offsets=relative_offsets,
    )
    metadata = factory_operation(factory)
    return make_invocable_from_specialization(
        adapter.materialize(
            spec.specialization,
            storage_abi=metadata.storage_abi,
            execution_scope=metadata.execution_scope,
            synchronization_scope=metadata.synchronization_scope,
        )
    )


def run_length_decode(**kwargs):
    return _decode(run_length_decode, **kwargs)


def run_length_decode_into(**kwargs):
    return _decode(run_length_decode_into, **kwargs)


def run_length_decode_into_offsets(**kwargs):
    return _decode(run_length_decode_into_offsets, **kwargs)


for _factory in (
    run_length_decode,
    run_length_decode_into,
    run_length_decode_into_offsets,
):
    register_factory(
        _factory,
        operation=_factory.__name__,
        namespace="block",
        storage_abi=StorageABI.LEADING_POINTER,
        execution_scope=SynchronizationScope.BLOCK,
        synchronization_scope=SynchronizationScope.BLOCK,
    )
del _factory
