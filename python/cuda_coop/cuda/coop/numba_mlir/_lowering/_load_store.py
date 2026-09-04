# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Load and store provider lowering for Numba-CUDA-MLIR.

Group planning selects a block route here after resolving hierarchy, dtype,
and launch facts.
"""

import operator

from numba_cuda_mlir import types

from cuda.coop._core import ArgumentBinding, BindingKind, SynchronizationScope
from cuda.coop._core.block import make_block_load_spec, make_block_store_spec

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


def _resolve_algorithm(algorithm, primitive_name: str) -> str:
    if not isinstance(algorithm, str):
        raise TypeError(f"{primitive_name} algorithm must be a string")
    token = algorithm.strip().lower().replace("-", "_")
    allowed = {
        "direct",
        "striped",
        "vectorize",
        "transpose",
        "warp_transpose",
        "warp_transpose_timesliced",
    }
    if token == "direct":
        return token
    if token in allowed:
        raise NotImplementedError(
            f"{primitive_name} algorithm {token!r} is not "
            "executable with the Numba-CUDA-MLIR backend; only 'direct' is "
            "currently supported"
        )
    choices = ", ".join(sorted(allowed))
    raise ValueError(
        f"Unsupported {primitive_name} algorithm {algorithm!r}; expected one "
        f"of: {choices}"
    )


def _registered_provider_metadata(factory):
    registered = factory_operation(factory)
    if registered is None:
        raise RuntimeError(f"unregistered cuda.coop provider {factory!r}")
    return {
        "storage_abi": registered.storage_abi,
        "execution_scope": registered.execution_scope,
        "synchronization_scope": registered.synchronization_scope,
    }


def _load_store_value_abis(
    *,
    dtype,
    block_dim,
    items_per_thread,
    valid_items,
    oob_default=None,
):
    """Declare family-owned runtime scalar ABIs for one specialization."""

    value_abis = {}
    if valid_items.kind is BindingKind.RUNTIME:
        tile_items = items_per_thread
        for dimension in block_dim:
            tile_items *= dimension
        value_abis["num_valid_items"] = BoundedInteger(
            types.int32,
            minimum=0,
            maximum=tile_items,
        )
    if oob_default is not None and oob_default.kind is BindingKind.RUNTIME:
        value_abis["oob_default"] = ExactValue(dtype)
    return value_abis


def load(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    algorithm="direct",
    num_valid_items=None,
    oob_default=None,
    offset=None,
):
    """Build the block-load invocable selected by group planning."""

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
    algorithm = _resolve_algorithm(algorithm, "block load")
    adapter = NumbaMlirCoreAdapter(
        value_abis=_load_store_value_abis(
            dtype=dtype,
            block_dim=block_dim,
            items_per_thread=items_per_thread,
            valid_items=valid_items_binding,
            oob_default=oob_default_binding,
        )
    )
    core_spec = make_block_load_spec(
        dtype=adapter.core_dtype(dtype),
        block_dim=tuple(block_dim),
        items_per_thread=items_per_thread,
        algorithm=algorithm,
        valid_items=valid_items_binding,
        oob_default=oob_default_binding,
        include_full_tile=(
            not isinstance(num_valid_items, ArgumentBinding)
            and num_valid_items is not None
        ),
        include_pointer_offset=(
            offset_binding if isinstance(offset, ArgumentBinding) else True
        ),
    )
    specialization = adapter.materialize(
        core_spec.specialization,
        **_registered_provider_metadata(load),
        extra_type_definitions=(numba_type_to_wrapper(dtype),),
    )
    return make_invocable_from_specialization(specialization)


def store(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    algorithm="direct",
    num_valid_items=None,
    oob_default=None,
    offset=None,
):
    """Build the block-store invocable selected by group planning."""

    if oob_default is not None:
        raise ValueError("oob_default is only valid for BlockLoad")
    valid_items_binding = _optional_binding(num_valid_items)
    offset_binding = _optional_binding(offset)
    if threads_per_block is None:
        raise ValueError("threads_per_block must be provided")
    block_dim = normalize_dim_param(threads_per_block)
    dtype = _validate_common_numeric_dtype(dtype, operation="store")
    items_per_thread = _positive_int(items_per_thread, name="items_per_thread")
    algorithm = _resolve_algorithm(algorithm, "block store")
    adapter = NumbaMlirCoreAdapter(
        value_abis=_load_store_value_abis(
            dtype=dtype,
            block_dim=block_dim,
            items_per_thread=items_per_thread,
            valid_items=valid_items_binding,
        )
    )
    core_spec = make_block_store_spec(
        dtype=adapter.core_dtype(dtype),
        block_dim=tuple(block_dim),
        items_per_thread=items_per_thread,
        algorithm=algorithm,
        valid_items=valid_items_binding,
        include_full_tile=(
            not isinstance(num_valid_items, ArgumentBinding)
            and num_valid_items is not None
        ),
        include_pointer_offset=(
            offset_binding if isinstance(offset, ArgumentBinding) else True
        ),
    )
    specialization = adapter.materialize(
        core_spec.specialization,
        **_registered_provider_metadata(store),
        extra_type_definitions=(numba_type_to_wrapper(dtype),),
    )
    return make_invocable_from_specialization(specialization)


for _factory in (load, store):
    register_factory(
        _factory,
        operation=_factory.__name__,
        namespace="block",
        storage_abi=StorageABI.NONE,
        execution_scope=SynchronizationScope.BLOCK,
        synchronization_scope=SynchronizationScope.NONE,
    )
del _factory
