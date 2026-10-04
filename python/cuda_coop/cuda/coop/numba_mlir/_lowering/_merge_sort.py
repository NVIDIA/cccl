# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Build the CUB Merge Sort provider factories used by the Numba rewrite.

Register keys-only and key/value factories for block and warp groups, each
with full- and partial-tile variants. Factories specialize shapes, dtypes,
and comparison operators. The compiler supplies copied arrays to sort in
place, followed by runtime count and sentinel operands for a partial tile.
"""

from cuda.coop._core import (
    INT8,
    CxxOperator,
    Dependency,
    PythonOperator,
    SynchronizationScope,
)
from cuda.coop._core.block.merge_sort import (
    make_block_merge_sort_specialization,
)
from cuda.coop._core.warp.merge_sort import make_warp_merge_sort_specialization

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


def comparison_operator(descending, compare_op):
    """Describe the built-in ordering or a stateless Python predicate.

    Require a constant boolean direction. Without a callback, select C++
    less-than or greater-than. A custom predicate owns the ordering, so reject
    its combination with ``descending=True``. Resolve both predicate argument
    types from ``KeyT`` and use an int8 device result whose truth value CUB
    can consume. Callback compilation occurs during provider specialization.
    """

    if not isinstance(descending, bool):
        raise TypeError("Merge Sort descending must be a compile-time bool")
    if compare_op is not None:
        if descending:
            raise ValueError(
                "Merge Sort compare_op and "
                "descending=True are mutually exclusive"
            )
        if not callable(compare_op):
            raise TypeError(
                "Merge Sort compare_op must be a stateless callable"
            )
        return PythonOperator(
            op_tokenizer=_numba_semantic_token,
            ret_dtype=INT8,
            arg_dtypes=(Dependency("KeyT"), Dependency("KeyT")),
            op=_normalize_numba_callable(compare_op),
            name="compare_op",
        )
    return CxxOperator(
        cpp="::cuda::std::greater<KeyT>"
        if descending
        else "::cuda::std::less<KeyT>",
        dtype=Dependency("KeyT"),
        name="compare_op",
    )


def _make_provider(namespace, pairs, partial):
    """Create and register one scope, payload, and tile-policy variant.

    The closure fixes block versus warp, keys versus pairs, and full versus
    partial tiles. Give it a distinct factory name and registry entry so the
    rewrite can recover the matching runtime signature and storage scope.
    All variants use a leading temporary-storage pointer; the compiler decides
    which scratch allocation supplies it.
    """

    operation = "merge_sort_pairs" if pairs else "merge_sort_keys"
    if partial:
        operation += "_partial"

    def provider(
        key_dtype,
        threads_per_block,
        items_per_thread,
        value_dtype=None,
        threads_in_warp=32,
        descending=False,
        compare_op=None,
    ):
        """Specialize one MergeSort variant for concrete shapes and dtypes.

        Normalize block dimensions and numeric key/value types, then build the
        shared CUB specialization. For partial tiles, zero placeholders record
        the presence of count and sentinel operands. Their values arrive
        later as runtime arguments.

        Adapt the shared signature with this factory's registered storage and
        synchronization scopes. A Python comparator compiles during this step.
        Return the invocable, carrying exact block geometry for warp providers
        so their scratch slices use the correct linear thread rank.
        """

        block_dim = normalize_dim_param(threads_per_block)
        key_dtype = _validate_common_numeric_dtype(
            key_dtype, operation=operation, parameter="keys"
        )
        if pairs:
            value_dtype = _validate_common_numeric_dtype(
                value_dtype, operation=operation, parameter="values"
            )
        elif value_dtype is not None:
            raise ValueError("keys-only Merge Sort does not accept value_dtype")
        adapter = NumbaMlirCoreAdapter()
        kwargs = {
            "key_dtype": adapter.core_dtype(key_dtype),
            "value_dtype": adapter.core_dtype(value_dtype) if pairs else None,
            "items_per_thread": items_per_thread,
            "compare_operator": comparison_operator(descending, compare_op),
            "valid_items": 0 if partial else None,
            "oob_default": 0 if partial else None,
        }
        if namespace == "block":
            specialization = make_block_merge_sort_specialization(
                block_dim=tuple(block_dim), **kwargs
            )
        else:
            specialization = make_warp_merge_sort_specialization(
                threads_in_warp=threads_in_warp, **kwargs
            )
        metadata = factory_operation(provider)
        assert metadata is not None
        specialization = adapter.materialize(
            specialization.specialization,
            storage_abi=metadata.storage_abi,
            execution_scope=metadata.execution_scope,
            synchronization_scope=metadata.synchronization_scope,
        )
        if namespace == "warp":
            return make_invocable_from_specialization(
                specialization,
                logical_warp_threads=threads_in_warp,
                block_threads=block_dim,
            )
        return make_invocable_from_specialization(specialization)

    provider.__name__ = f"{namespace}_{operation}"
    scope = (
        SynchronizationScope.BLOCK
        if namespace == "block"
        else SynchronizationScope.WARP
    )
    register_factory(
        provider,
        operation=operation,
        namespace=namespace,
        storage_abi=StorageABI.LEADING_POINTER,
        execution_scope=scope,
        synchronization_scope=scope,
    )
    return provider


block_merge_sort_keys = _make_provider("block", False, False)
block_merge_sort_keys_partial = _make_provider("block", False, True)
block_merge_sort_pairs = _make_provider("block", True, False)
block_merge_sort_pairs_partial = _make_provider("block", True, True)
warp_merge_sort_keys = _make_provider("warp", False, False)
warp_merge_sort_keys_partial = _make_provider("warp", False, True)
warp_merge_sort_pairs = _make_provider("warp", True, False)
warp_merge_sort_pairs_partial = _make_provider("warp", True, True)

__all__: tuple[str, ...] = ()
