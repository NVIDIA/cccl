# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Block and physical or logical WarpExchange provider lowering."""

from __future__ import annotations

import operator
from typing import Any

from numba_cuda_mlir import types

from cuda.coop._core import SynchronizationScope
from cuda.coop._core.block.exchange import (
    BlockExchangeMode,
    BlockExchangeValueForm,
    make_block_exchange_spec,
)
from cuda.coop._core.warp.exchange import (
    WarpExchangeMode,
    WarpExchangeValueForm,
    make_warp_exchange_spec,
)

from .._compiler._operations import (
    StorageABI,
    factory_operation,
    register_factory,
)
from .._compiler._parameters import (
    _validate_common_numeric_dtype,
    normalize_dim_param,
    normalize_dtype_param,
)
from .._types import make_invocable_from_specialization, numba_type_to_wrapper
from ._core import NumbaMlirCoreAdapter


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


def _mode(value: Any, enum_type: type, *, operation: str):
    if not isinstance(value, str):
        raise TypeError(f"{operation} mode must be a string")
    token = value.strip().lower().replace("-", "_")
    try:
        return enum_type(token)
    except ValueError as exc:
        choices = ", ".join(member.value for member in enum_type)
        raise ValueError(f"{operation} mode must be one of: {choices}") from exc


def _rank_dtype(value: Any):
    value = normalize_dtype_param(value)
    if (
        isinstance(value, types.Boolean)
        or not isinstance(value, types.Integer)
        or not value.signed
    ):
        raise TypeError("exchange ranks must have a signed integer dtype")
    return value


def _valid_flag_dtype(value: Any):
    value = normalize_dtype_param(value)
    if isinstance(value, types.Boolean) or not isinstance(value, types.Integer):
        raise TypeError("exchange valid_flags must have an integer dtype")
    return value


def _provider_metadata(factory, *, operation: str, namespace: str):
    registered = factory_operation(factory)
    if registered is None:
        raise RuntimeError(f"unregistered cuda.coop provider {factory!r}")
    if registered.operation != operation or registered.namespace != namespace:
        raise RuntimeError(f"invalid {operation} provider registration {registered!r}")
    return {
        "storage_abi": registered.storage_abi,
        "execution_scope": registered.execution_scope,
        "synchronization_scope": registered.synchronization_scope,
    }


def _block_exchange(
    provider_factory,
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    mode="striped_to_blocked",
    rank_dtype=None,
    valid_flag_dtype=None,
    warp_time_slicing=False,
):
    """Build an out-of-place BlockExchange invocable."""

    if threads_per_block is None:
        raise ValueError("threads_per_block must be provided")
    if not isinstance(warp_time_slicing, bool):
        raise TypeError("warp_time_slicing must be a bool")
    block_dim = normalize_dim_param(threads_per_block)
    dtype = _validate_common_numeric_dtype(dtype, operation="exchange")
    items_per_thread = _positive_int(
        items_per_thread,
        name="items_per_thread",
    )
    mode = _mode(mode, BlockExchangeMode, operation="block exchange")
    expected_operation = (
        "exchange_flagged"
        if mode.uses_valid_flags
        else "exchange_ranked"
        if mode.uses_ranks
        else "exchange"
    )
    rank_dtype = None if rank_dtype is None else _rank_dtype(rank_dtype)
    valid_flag_dtype = (
        None if valid_flag_dtype is None else _valid_flag_dtype(valid_flag_dtype)
    )
    adapter = NumbaMlirCoreAdapter()
    core_spec = make_block_exchange_spec(
        dtype=adapter.core_dtype(dtype),
        block_dim=tuple(block_dim),
        items_per_thread=items_per_thread,
        mode=mode,
        value_form=BlockExchangeValueForm.OUT_OF_PLACE,
        warp_time_slicing=warp_time_slicing,
        rank_dtype=(None if rank_dtype is None else adapter.core_dtype(rank_dtype)),
        valid_flag_dtype=(
            None if valid_flag_dtype is None else adapter.core_dtype(valid_flag_dtype)
        ),
    )
    specialization = adapter.materialize(
        core_spec.specialization,
        **_provider_metadata(
            provider_factory,
            operation=expected_operation,
            namespace="block",
        ),
        extra_type_definitions=(numba_type_to_wrapper(dtype),),
    )
    return make_invocable_from_specialization(specialization)


def exchange(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    mode="striped_to_blocked",
    rank_dtype=None,
    valid_flag_dtype=None,
    warp_time_slicing=False,
):
    """Build an unranked out-of-place BlockExchange invocable."""

    return _block_exchange(
        exchange,
        dtype,
        threads_per_block,
        items_per_thread,
        mode,
        rank_dtype,
        valid_flag_dtype,
        warp_time_slicing,
    )


def exchange_ranked(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    mode="scatter_to_striped",
    rank_dtype=None,
    valid_flag_dtype=None,
    warp_time_slicing=False,
):
    """Build a ranked out-of-place BlockExchange invocable."""

    return _block_exchange(
        exchange_ranked,
        dtype,
        threads_per_block,
        items_per_thread,
        mode,
        rank_dtype,
        valid_flag_dtype,
        warp_time_slicing,
    )


def exchange_flagged(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    mode="scatter_to_striped_flagged",
    rank_dtype=None,
    valid_flag_dtype=None,
    warp_time_slicing=False,
):
    """Build a ranked and flagged out-of-place BlockExchange invocable."""

    return _block_exchange(
        exchange_flagged,
        dtype,
        threads_per_block,
        items_per_thread,
        mode,
        rank_dtype,
        valid_flag_dtype,
        warp_time_slicing,
    )


def _warp_exchange(
    provider_factory,
    dtype,
    threads_per_block=None,
    threads_in_warp=32,
    items_per_thread=1,
    mode="striped_to_blocked",
    rank_dtype=None,
):
    """Build an out-of-place physical or logical WarpExchange invocable."""

    if threads_per_block is None:
        raise ValueError("threads_per_block must be provided")
    block_dim = normalize_dim_param(threads_per_block)
    dtype = _validate_common_numeric_dtype(dtype, operation="exchange")
    items_per_thread = _positive_int(
        items_per_thread,
        name="items_per_thread",
    )
    threads_in_warp = _positive_int(
        threads_in_warp,
        name="threads_in_warp",
    )
    mode = _mode(mode, WarpExchangeMode, operation="warp exchange")
    expected_operation = (
        "exchange_ranked" if mode is WarpExchangeMode.SCATTER_TO_STRIPED else "exchange"
    )
    rank_dtype = None if rank_dtype is None else _rank_dtype(rank_dtype)
    adapter = NumbaMlirCoreAdapter()
    core_spec = make_warp_exchange_spec(
        dtype=adapter.core_dtype(dtype),
        items_per_thread=items_per_thread,
        threads_in_warp=threads_in_warp,
        mode=mode,
        value_form=WarpExchangeValueForm.OUT_OF_PLACE,
        rank_dtype=(None if rank_dtype is None else adapter.core_dtype(rank_dtype)),
    )
    specialization = adapter.materialize(
        core_spec.specialization,
        **_provider_metadata(
            provider_factory,
            operation=expected_operation,
            namespace="warp",
        ),
        extra_type_definitions=(numba_type_to_wrapper(dtype),),
    )
    return make_invocable_from_specialization(
        specialization,
        threads=threads_in_warp,
        block_threads=block_dim,
    )


def warp_exchange(
    dtype,
    threads_per_block=None,
    threads_in_warp=32,
    items_per_thread=1,
    mode="striped_to_blocked",
    rank_dtype=None,
):
    """Build an unranked physical or logical WarpExchange invocable."""

    return _warp_exchange(
        warp_exchange,
        dtype,
        threads_per_block,
        threads_in_warp,
        items_per_thread,
        mode,
        rank_dtype,
    )


def warp_exchange_ranked(
    dtype,
    threads_per_block=None,
    threads_in_warp=32,
    items_per_thread=1,
    mode="scatter_to_striped",
    rank_dtype=None,
):
    """Build a ranked physical or logical WarpExchange invocable."""

    return _warp_exchange(
        warp_exchange_ranked,
        dtype,
        threads_per_block,
        threads_in_warp,
        items_per_thread,
        mode,
        rank_dtype,
    )


for _factory, _operation in (
    (exchange, "exchange"),
    (exchange_ranked, "exchange_ranked"),
    (exchange_flagged, "exchange_flagged"),
):
    register_factory(
        _factory,
        operation=_operation,
        namespace="block",
        storage_abi=StorageABI.LEADING_POINTER,
        execution_scope=SynchronizationScope.BLOCK,
        synchronization_scope=SynchronizationScope.BLOCK,
    )
for _factory, _operation in (
    (warp_exchange, "exchange"),
    (warp_exchange_ranked, "exchange_ranked"),
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


__all__: tuple[str, ...] = ()
