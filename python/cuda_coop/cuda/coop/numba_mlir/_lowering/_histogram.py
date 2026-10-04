# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Adapt the shared block histogram wrapper to Numba's provider ABI.

The wrapper copies samples, initializes fresh shared counters, and distributes
bins to per-thread output arrays after an internal barrier. This factory
supplies compiler types and the leading scratch pointer. Public result
allocation and any barrier for later scratch reuse belong to group rewriting.
"""

from cuda.coop._core import SynchronizationScope
from cuda.coop._core.block.histogram import (
    make_block_histogram_specialization,
    validate_histogram_dtype,
)

from .._compiler._operations import (
    StorageABI,
    factory_operation,
    register_factory,
)
from .._compiler._parameters import normalize_dim_param, normalize_dtype_param
from .._types import make_invocable_from_specialization
from ._core import NumbaMlirCoreAdapter


def histogram(
    *,
    sample_dtype,
    counter_dtype,
    threads_per_block,
    items_per_thread,
    bins,
    bins_per_thread=1,
    algorithm="atomic",
):
    """Build a provider with independent sample and counter array types.

    Validate the two dtype sets separately and normalize the block dimensions.
    The shared builder checks the one-dimensional shape and bin capacity, then
    supplies the C++ wrapper. Its private sample copy preserves inputs even
    when the sort algorithm reorders its working items.

    Materialize that specialization with registered block synchronization and
    scratch metadata. Return an invocable that writes the supplied counter
    array; the public rewrite allocates that array with bins_per_thread slots.
    """

    adapter = NumbaMlirCoreAdapter()
    sample_dtype = validate_histogram_dtype(normalize_dtype_param(sample_dtype))
    counter_dtype = validate_histogram_dtype(
        normalize_dtype_param(counter_dtype), counter=True
    )
    specialization = make_block_histogram_specialization(
        sample_dtype=adapter.core_dtype(sample_dtype),
        counter_dtype=adapter.core_dtype(counter_dtype),
        block_dim=tuple(normalize_dim_param(threads_per_block)),
        items_per_thread=items_per_thread,
        bins=bins,
        bins_per_thread=bins_per_thread,
        algorithm=algorithm,
    )
    metadata = factory_operation(histogram)
    assert metadata is not None
    specialization = adapter.materialize(
        specialization.specialization,
        storage_abi=metadata.storage_abi,
        execution_scope=metadata.execution_scope,
        synchronization_scope=metadata.synchronization_scope,
    )
    return make_invocable_from_specialization(specialization)


register_factory(
    histogram,
    operation="histogram",
    namespace="block",
    storage_abi=StorageABI.LEADING_POINTER,
    execution_scope=SynchronizationScope.BLOCK,
    synchronization_scope=SynchronizationScope.BLOCK,
)
