# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Block and physical-Warp Load/Store provider lowering."""

import operator
from enum import Enum

from numba_cuda_mlir import types

from cuda.coop._core import ArgumentBinding, BindingKind, SynchronizationScope
from cuda.coop._core.block import make_block_load_spec, make_block_store_spec
from cuda.coop._core.warp import make_warp_load_spec, make_warp_store_spec

from .._compiler._operations import (
    StorageABI,
    factory_operation,
    register_factory,
)
from .._compiler._parameters import (
    _validate_common_numeric_dtype,
    _validate_static_oob_default,
    normalize_dim_param,
)
from .._types import (
    BoundedInteger,
    ExactValue,
    make_invocable_from_specialization,
    numba_type_to_wrapper,
)
from ._core import NumbaMlirCoreAdapter, _optional_binding

_BLOCK_LOAD_STORE_ALGORITHMS = frozenset(
    {
        "direct",
        "striped",
        "vectorize",
        "transpose",
        "warp_transpose",
        "warp_transpose_timesliced",
    }
)
_WARP_LOAD_STORE_ALGORITHMS = frozenset(
    {
        "direct",
        "striped",
        "vectorize",
        "transpose",
    }
)
_STORAGE_FREE_ALGORITHMS = frozenset({"direct", "striped", "vectorize"})


def _positive_int(value, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _resolve_algorithm(algorithm, allowed_algorithms, primitive_name: str) -> str:
    if not isinstance(algorithm, str) or isinstance(algorithm, Enum):
        raise TypeError(f"{primitive_name} algorithm must be a string")
    token = algorithm.strip().lower().replace("-", "_")
    if token in allowed_algorithms:
        return token
    choices = ", ".join(sorted(allowed_algorithms))
    raise ValueError(
        f"Unsupported {primitive_name} algorithm {algorithm!r}; expected one "
        f"of: {choices}"
    )


def _registered_provider_metadata(factory, algorithm):
    registered = factory_operation(factory)
    if registered is None:
        raise RuntimeError(f"unregistered cuda.coop provider {factory!r}")
    expected_storage_abi = (
        StorageABI.NONE
        if algorithm in _STORAGE_FREE_ALGORITHMS
        else StorageABI.LEADING_POINTER
    )
    if registered.storage_abi is not expected_storage_abi:
        raise ValueError(
            f"{registered.operation} algorithm {algorithm!r} "
            f"requires the {expected_storage_abi.value!r} provider"
        )
    return registered


def _materialization_metadata(registered):
    return {
        "storage_abi": registered.storage_abi,
        "execution_scope": registered.execution_scope,
        "synchronization_scope": registered.synchronization_scope,
    }


def _load_store_value_abis(
    *,
    dtype,
    items_per_thread,
    valid_items,
    oob_default=None,
    block_dim=None,
    threads_in_warp=None,
):
    """Declare family-owned runtime scalar ABIs for one specialization."""

    value_abis = {}
    if valid_items.kind is BindingKind.RUNTIME:
        tile_items = items_per_thread
        if (block_dim is None) == (threads_in_warp is None):
            raise ValueError(
                "exactly one of block_dim or threads_in_warp must be provided"
            )
        if block_dim is not None:
            for dimension in block_dim:
                tile_items *= dimension
        else:
            tile_items *= threads_in_warp
        value_abis["num_valid_items"] = BoundedInteger(
            types.int32,
            minimum=0,
            maximum=tile_items,
        )
    if oob_default is not None and oob_default.kind is BindingKind.RUNTIME:
        value_abis["oob_default"] = ExactValue(dtype)
    return value_abis


def _physical_warp_threads(value) -> int:
    value = _positive_int(value, name="threads_in_warp")
    if value != 32:
        raise ValueError("physical WarpLoad and WarpStore require threads_in_warp=32")
    return value


def _load(
    provider_factory,
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    algorithm="direct",
    num_valid_items=None,
    oob_default=None,
    offset=None,
    threads_in_warp=None,
):
    """Build the Load invocable selected by group planning."""

    valid_items_binding = _optional_binding(num_valid_items)
    oob_default_binding = _optional_binding(oob_default)
    offset_binding = _optional_binding(offset)
    if (
        oob_default_binding.kind is not BindingKind.OMITTED
        and valid_items_binding.kind is BindingKind.OMITTED
    ):
        raise ValueError("oob_default requires num_valid_items to be provided")
    if threads_per_block is None:
        raise ValueError("threads_per_block must be provided")
    block_dim = normalize_dim_param(threads_per_block)
    dtype = _validate_common_numeric_dtype(dtype, operation="load")
    if oob_default_binding.kind is BindingKind.STATIC:
        oob_default_binding = ArgumentBinding.static(
            _validate_static_oob_default(oob_default_binding.value, dtype)
        )
    items_per_thread = _positive_int(items_per_thread, name="items_per_thread")
    registered = factory_operation(provider_factory)
    if registered is None:
        raise RuntimeError(f"unregistered cuda.coop provider {provider_factory!r}")
    if registered.namespace == "block":
        if threads_in_warp is not None:
            raise ValueError("block Load does not accept threads_in_warp")
        algorithm = _resolve_algorithm(
            algorithm,
            _BLOCK_LOAD_STORE_ALGORITHMS,
            "block load",
        )
        group_kwargs = {"block_dim": block_dim}
    elif registered.namespace == "warp":
        threads_in_warp = _physical_warp_threads(threads_in_warp)
        algorithm = _resolve_algorithm(
            algorithm,
            _WARP_LOAD_STORE_ALGORITHMS,
            "warp load",
        )
        group_kwargs = {"threads_in_warp": threads_in_warp}
    else:
        raise ValueError(
            f"unsupported cuda.coop Load provider namespace {registered.namespace!r}"
        )
    registered = _registered_provider_metadata(provider_factory, algorithm)
    adapter = NumbaMlirCoreAdapter(
        value_abis=_load_store_value_abis(
            dtype=dtype,
            items_per_thread=items_per_thread,
            valid_items=valid_items_binding,
            oob_default=oob_default_binding,
            **group_kwargs,
        )
    )
    spec_factory = (
        make_block_load_spec if registered.namespace == "block" else make_warp_load_spec
    )
    core_spec = spec_factory(
        dtype=adapter.core_dtype(dtype),
        items_per_thread=items_per_thread,
        algorithm=str(algorithm),
        valid_items=valid_items_binding,
        oob_default=oob_default_binding,
        include_full_tile=(
            not isinstance(num_valid_items, ArgumentBinding)
            and num_valid_items is not None
        ),
        include_pointer_offset=(
            offset_binding if isinstance(offset, ArgumentBinding) else True
        ),
        **(
            {"block_dim": tuple(block_dim)}
            if registered.namespace == "block"
            else {"threads_in_warp": threads_in_warp}
        ),
    )
    specialization = adapter.materialize(
        core_spec.specialization,
        **_materialization_metadata(registered),
        extra_type_definitions=(numba_type_to_wrapper(dtype),),
    )
    invocation_topology = (
        {
            "threads": threads_in_warp,
            "block_threads": block_dim,
        }
        if registered.namespace == "warp"
        else {}
    )
    return make_invocable_from_specialization(
        specialization,
        **invocation_topology,
    )


def load(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    algorithm="direct",
    num_valid_items=None,
    oob_default=None,
    offset=None,
):
    """Build a storage-free BlockLoad invocable."""

    return _load(
        load,
        dtype,
        threads_per_block,
        items_per_thread,
        algorithm,
        num_valid_items,
        oob_default,
        offset,
    )


def _load_with_storage(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    algorithm="transpose",
    num_valid_items=None,
    oob_default=None,
    offset=None,
):
    """Build a storage-bearing transpose BlockLoad invocable."""

    return _load(
        _load_with_storage,
        dtype,
        threads_per_block,
        items_per_thread,
        algorithm,
        num_valid_items,
        oob_default,
        offset,
    )


def warp_load(
    dtype,
    threads_per_block=None,
    threads_in_warp=32,
    items_per_thread=1,
    algorithm="direct",
    num_valid_items=None,
    oob_default=None,
    offset=None,
):
    """Build a storage-free physical WarpLoad invocable."""

    return _load(
        warp_load,
        dtype,
        threads_per_block=threads_per_block,
        threads_in_warp=threads_in_warp,
        items_per_thread=items_per_thread,
        algorithm=algorithm,
        num_valid_items=num_valid_items,
        oob_default=oob_default,
        offset=offset,
    )


def _warp_load_with_storage(
    dtype,
    threads_per_block=None,
    threads_in_warp=32,
    items_per_thread=1,
    algorithm="transpose",
    num_valid_items=None,
    oob_default=None,
    offset=None,
):
    """Build a storage-bearing physical WarpLoad invocable."""

    return _load(
        _warp_load_with_storage,
        dtype,
        threads_per_block=threads_per_block,
        threads_in_warp=threads_in_warp,
        items_per_thread=items_per_thread,
        algorithm=algorithm,
        num_valid_items=num_valid_items,
        oob_default=oob_default,
        offset=offset,
    )


def _store(
    provider_factory,
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    algorithm="direct",
    num_valid_items=None,
    oob_default=None,
    offset=None,
    threads_in_warp=None,
):
    """Build the Store invocable selected by group planning."""

    if oob_default is not None:
        raise ValueError("oob_default is only valid for Load")
    valid_items_binding = _optional_binding(num_valid_items)
    offset_binding = _optional_binding(offset)
    if threads_per_block is None:
        raise ValueError("threads_per_block must be provided")
    block_dim = normalize_dim_param(threads_per_block)
    dtype = _validate_common_numeric_dtype(dtype, operation="store")
    items_per_thread = _positive_int(items_per_thread, name="items_per_thread")
    registered = factory_operation(provider_factory)
    if registered is None:
        raise RuntimeError(f"unregistered cuda.coop provider {provider_factory!r}")
    if registered.namespace == "block":
        if threads_in_warp is not None:
            raise ValueError("block Store does not accept threads_in_warp")
        algorithm = _resolve_algorithm(
            algorithm,
            _BLOCK_LOAD_STORE_ALGORITHMS,
            "block store",
        )
        group_kwargs = {"block_dim": block_dim}
    elif registered.namespace == "warp":
        threads_in_warp = _physical_warp_threads(threads_in_warp)
        algorithm = _resolve_algorithm(
            algorithm,
            _WARP_LOAD_STORE_ALGORITHMS,
            "warp store",
        )
        group_kwargs = {"threads_in_warp": threads_in_warp}
    else:
        raise ValueError(
            f"unsupported cuda.coop Store provider namespace {registered.namespace!r}"
        )
    registered = _registered_provider_metadata(provider_factory, algorithm)
    adapter = NumbaMlirCoreAdapter(
        value_abis=_load_store_value_abis(
            dtype=dtype,
            items_per_thread=items_per_thread,
            valid_items=valid_items_binding,
            **group_kwargs,
        )
    )
    spec_factory = (
        make_block_store_spec
        if registered.namespace == "block"
        else make_warp_store_spec
    )
    core_spec = spec_factory(
        dtype=adapter.core_dtype(dtype),
        items_per_thread=items_per_thread,
        algorithm=str(algorithm),
        valid_items=valid_items_binding,
        include_full_tile=(
            not isinstance(num_valid_items, ArgumentBinding)
            and num_valid_items is not None
        ),
        include_pointer_offset=(
            offset_binding if isinstance(offset, ArgumentBinding) else True
        ),
        **(
            {"block_dim": tuple(block_dim)}
            if registered.namespace == "block"
            else {"threads_in_warp": threads_in_warp}
        ),
    )
    specialization = adapter.materialize(
        core_spec.specialization,
        **_materialization_metadata(registered),
        extra_type_definitions=(numba_type_to_wrapper(dtype),),
    )
    invocation_topology = (
        {
            "threads": threads_in_warp,
            "block_threads": block_dim,
        }
        if registered.namespace == "warp"
        else {}
    )
    return make_invocable_from_specialization(
        specialization,
        **invocation_topology,
    )


def store(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    algorithm="direct",
    num_valid_items=None,
    oob_default=None,
    offset=None,
):
    """Build a storage-free BlockStore invocable."""

    return _store(
        store,
        dtype,
        threads_per_block,
        items_per_thread,
        algorithm,
        num_valid_items,
        oob_default,
        offset,
    )


def _store_with_storage(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    algorithm="transpose",
    num_valid_items=None,
    oob_default=None,
    offset=None,
):
    """Build a storage-bearing transpose BlockStore invocable."""

    return _store(
        _store_with_storage,
        dtype,
        threads_per_block,
        items_per_thread,
        algorithm,
        num_valid_items,
        oob_default,
        offset,
    )


def warp_store(
    dtype,
    threads_per_block=None,
    threads_in_warp=32,
    items_per_thread=1,
    algorithm="direct",
    num_valid_items=None,
    oob_default=None,
    offset=None,
):
    """Build a storage-free physical WarpStore invocable."""

    return _store(
        warp_store,
        dtype,
        threads_per_block=threads_per_block,
        threads_in_warp=threads_in_warp,
        items_per_thread=items_per_thread,
        algorithm=algorithm,
        num_valid_items=num_valid_items,
        oob_default=oob_default,
        offset=offset,
    )


def _warp_store_with_storage(
    dtype,
    threads_per_block=None,
    threads_in_warp=32,
    items_per_thread=1,
    algorithm="transpose",
    num_valid_items=None,
    oob_default=None,
    offset=None,
):
    """Build a storage-bearing physical WarpStore invocable."""

    return _store(
        _warp_store_with_storage,
        dtype,
        threads_per_block=threads_per_block,
        threads_in_warp=threads_in_warp,
        items_per_thread=items_per_thread,
        algorithm=algorithm,
        num_valid_items=num_valid_items,
        oob_default=oob_default,
        offset=offset,
    )


for _factory, _operation in ((load, "load"), (store, "store")):
    register_factory(
        _factory,
        operation=_operation,
        namespace="block",
        storage_abi=StorageABI.NONE,
        execution_scope=SynchronizationScope.BLOCK,
        synchronization_scope=SynchronizationScope.NONE,
    )
for _factory, _operation in (
    (_load_with_storage, "load"),
    (_store_with_storage, "store"),
):
    register_factory(
        _factory,
        operation=_operation,
        namespace="block",
        storage_abi=StorageABI.LEADING_POINTER,
        execution_scope=SynchronizationScope.BLOCK,
        synchronization_scope=SynchronizationScope.BLOCK,
    )
for _factory, _operation in ((warp_load, "load"), (warp_store, "store")):
    register_factory(
        _factory,
        operation=_operation,
        namespace="warp",
        storage_abi=StorageABI.NONE,
        execution_scope=SynchronizationScope.WARP,
        synchronization_scope=SynchronizationScope.NONE,
    )
for _factory, _operation in (
    (_warp_load_with_storage, "load"),
    (_warp_store_with_storage, "store"),
):
    register_factory(
        _factory,
        operation=_operation,
        namespace="warp",
        storage_abi=StorageABI.LEADING_POINTER,
        execution_scope=SynchronizationScope.WARP,
        synchronization_scope=SynchronizationScope.WARP,
    )
del _factory, _operation
