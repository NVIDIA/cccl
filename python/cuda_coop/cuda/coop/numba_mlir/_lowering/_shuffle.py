# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""BlockShuffle scalar and array provider lowering."""

from __future__ import annotations

import operator
from enum import Enum
from typing import Any

from numba_cuda_mlir import types

from cuda.coop._core import ArgumentBinding, BindingKind, SynchronizationScope
from cuda.coop._core.block.shuffle import BlockShuffleMode, make_block_shuffle_spec

from .._compiler._operations import (
    StorageABI,
    factory_operation,
    register_factory,
)
from .._compiler._parameters import (
    _validate_common_numeric_dtype,
    normalize_dim_param,
)
from .._types import (
    BoundedInteger,
    make_invocable_from_specialization,
    numba_type_to_wrapper,
)
from ._core import NumbaMlirCoreAdapter, _optional_binding

_I32_MIN = -(1 << 31)
_I32_MAX = (1 << 31) - 1


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _mode(
    value: Any,
    *,
    allowed: frozenset[BlockShuffleMode],
) -> BlockShuffleMode:
    if not isinstance(value, str) or isinstance(value, Enum):
        raise TypeError("shuffle mode must be a string")
    token = value.strip().lower().replace("-", "_")
    try:
        mode = BlockShuffleMode(token)
    except ValueError as exc:
        choices = ", ".join(member.value for member in sorted(allowed, key=str))
        raise ValueError(f"shuffle mode must be one of: {choices}") from exc
    if mode not in allowed:
        choices = ", ".join(member.value for member in sorted(allowed, key=str))
        raise ValueError(f"shuffle mode must be one of: {choices}")
    return mode


def _block_threads(block_dim) -> int:
    return block_dim.x * block_dim.y * block_dim.z


def _provider_metadata(factory, *, operation: str):
    registered = factory_operation(factory)
    if registered is None:
        raise RuntimeError(f"unregistered cuda.coop provider {factory!r}")
    if registered.operation != operation or registered.namespace != "block":
        raise RuntimeError(f"invalid {operation} provider registration {registered!r}")
    return {
        "storage_abi": registered.storage_abi,
        "execution_scope": registered.execution_scope,
        "synchronization_scope": registered.synchronization_scope,
    }


def _static_rotate_distance(distance: ArgumentBinding, block_threads: int) -> None:
    if distance.kind is BindingKind.RUNTIME:
        return
    value = 1 if distance.kind is BindingKind.OMITTED else distance.value
    if isinstance(value, (bool, Enum)):
        raise TypeError("static rotate distance must be an integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError("static rotate distance must be an integer") from exc
    if not 1 <= value < block_threads:
        raise ValueError(
            "static rotate distance must be between 1 and block_threads - 1"
        )


def shuffle_scalar(
    dtype,
    threads_per_block=None,
    mode="offset",
    distance=None,
):
    """Build a scalar Offset or Rotate BlockShuffle invocable."""

    if threads_per_block is None:
        raise ValueError("threads_per_block must be provided")
    block_dim = normalize_dim_param(threads_per_block)
    block_threads = _block_threads(block_dim)
    dtype = _validate_common_numeric_dtype(dtype, operation="shuffle")
    mode = _mode(
        mode,
        allowed=frozenset({BlockShuffleMode.OFFSET, BlockShuffleMode.ROTATE}),
    )
    distance = _optional_binding(distance)
    if mode is BlockShuffleMode.ROTATE:
        _static_rotate_distance(distance, block_threads)
    value_abis = {}
    if distance.kind is BindingKind.RUNTIME:
        provider_dtype, bounds = (
            (types.int32, (_I32_MIN, _I32_MAX))
            if mode is BlockShuffleMode.OFFSET
            else (types.uint32, (1, block_threads - 1))
        )
        value_abis["distance"] = BoundedInteger(
            provider_dtype,
            minimum=bounds[0],
            maximum=bounds[1],
        )
    adapter = NumbaMlirCoreAdapter(value_abis=value_abis)
    core_spec = make_block_shuffle_spec(
        dtype=adapter.core_dtype(dtype),
        block_dim=tuple(block_dim),
        mode=mode,
        distance=distance,
    )
    specialization = adapter.materialize(
        core_spec.specialization,
        **_provider_metadata(shuffle_scalar, operation="shuffle_scalar"),
        extra_type_definitions=(numba_type_to_wrapper(dtype),),
    )
    return make_invocable_from_specialization(specialization)


def shuffle_array(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    mode="down",
):
    """Build a unit Up or Down array BlockShuffle invocable."""

    if threads_per_block is None:
        raise ValueError("threads_per_block must be provided")
    block_dim = normalize_dim_param(threads_per_block)
    dtype = _validate_common_numeric_dtype(dtype, operation="shuffle")
    items_per_thread = _positive_int(
        items_per_thread,
        name="items_per_thread",
    )
    mode = _mode(
        mode,
        allowed=frozenset({BlockShuffleMode.UP, BlockShuffleMode.DOWN}),
    )
    adapter = NumbaMlirCoreAdapter()
    core_spec = make_block_shuffle_spec(
        dtype=adapter.core_dtype(dtype),
        block_dim=tuple(block_dim),
        mode=mode,
        items_per_thread=items_per_thread,
    )
    specialization = adapter.materialize(
        core_spec.specialization,
        **_provider_metadata(shuffle_array, operation="shuffle_array"),
        extra_type_definitions=(numba_type_to_wrapper(dtype),),
    )
    return make_invocable_from_specialization(specialization)


for _factory, _operation in (
    (shuffle_scalar, "shuffle_scalar"),
    (shuffle_array, "shuffle_array"),
):
    register_factory(
        _factory,
        operation=_operation,
        namespace="block",
        storage_abi=StorageABI.LEADING_POINTER,
        execution_scope=SynchronizationScope.BLOCK,
        synchronization_scope=SynchronizationScope.BLOCK,
    )
del _factory, _operation


__all__: tuple[str, ...] = ()
