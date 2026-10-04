# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Build and register Numba invocables for the core block neighbor wrappers.

Factory metadata supplies the scratch-pointer ABI and block synchronization
scope. The factories specialize wrappers and adapt device callbacks; kernel
execution happens later through the resulting invocables.
"""

from cuda.coop._core import (
    INT8,
    CxxOperator,
    Dependency,
    PythonOperator,
    SynchronizationScope,
)
from cuda.coop._core.block.neighbors import (
    BlockNeighborSemantics,
    make_block_neighbor_specialization,
)

from .._compiler._operations import (
    StorageABI,
    factory_operation,
    register_factory,
)
from .._compiler._parameters import (
    _validate_common_numeric_dtype,
    normalize_dim_param,
)
from .._semantic import _normalize_numba_callable, _numba_semantic_token
from .._types import make_invocable_from_specialization
from ._core import NumbaMlirCoreAdapter


def neighbor_operator(operation, op):
    """Describe a built-in functor or a stateless Python device callback.

    Both callback arguments use the input dtype. Difference callbacks return
    that dtype; flag callbacks use an int8 ABI for the predicate result.
    The public discontinuity payload stores flags as int32. Callable identity
    includes nested device helpers and captured values through the tokenizer.
    """

    dtype = Dependency("T")
    if op is None:
        name = "minus" if operation == "adjacent_difference" else "not_equal_to"
        return CxxOperator(
            cpp=f"::cuda::std::{name}<T>", dtype=dtype, name="op"
        )
    if not callable(op):
        raise TypeError(f"{operation} operator must be a stateless callable")
    return PythonOperator(
        op_tokenizer=_numba_semantic_token,
        ret_dtype=dtype if operation == "adjacent_difference" else INT8,
        arg_dtypes=(dtype, dtype),
        op=_normalize_numba_callable(op),
        name="op",
    )


def _make_provider(operation, *, both=False):
    """Register a block factory with one output or a heads/tails output pair.

    The two-output form needs a distinct factory name: its provider call has
    an extra result-array argument. All forms use block-scoped scratch.
    """

    def provider(
        dtype,
        threads_per_block,
        items_per_thread,
        mode,
        partial=False,
        predecessor=False,
        successor=False,
        op=None,
    ):
        """Build a numeric specialization and adapt it to the call ABI.

        The core factory chooses the neighbor overload and scratch layout.
        Attach Numba operator types and storage metadata before creating the
        invocable that the kernel will call.
        """

        dtype = _validate_common_numeric_dtype(
            dtype, operation=operation, parameter="values"
        )
        adapter = NumbaMlirCoreAdapter()
        call = BlockNeighborSemantics(
            operation=operation,
            dtype=adapter.core_dtype(dtype),
            items_per_thread=items_per_thread,
            mode=mode,
            operator=neighbor_operator(operation, op),
            partial=partial,
            predecessor=predecessor,
            successor=successor,
        )
        specialization = make_block_neighbor_specialization(
            call, block_dim=normalize_dim_param(threads_per_block)
        )
        metadata = factory_operation(provider)
        assert metadata is not None
        specialization = adapter.materialize(
            specialization,
            storage_abi=metadata.storage_abi,
            execution_scope=metadata.execution_scope,
            synchronization_scope=metadata.synchronization_scope,
        )
        return make_invocable_from_specialization(specialization)

    factory_name = operation + ("_both" if both else "")
    provider.__name__ = "block_" + factory_name
    register_factory(
        provider,
        operation=factory_name,
        namespace="block",
        storage_abi=StorageABI.LEADING_POINTER,
        execution_scope=SynchronizationScope.BLOCK,
        synchronization_scope=SynchronizationScope.BLOCK,
    )
    return provider


block_adjacent_difference = _make_provider("adjacent_difference")
block_discontinuity = _make_provider("discontinuity")
block_discontinuity_both = _make_provider("discontinuity", both=True)

__all__: tuple[str, ...] = ()
