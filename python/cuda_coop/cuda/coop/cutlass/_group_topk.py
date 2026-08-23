# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUTLASS group-first CUB BlockTopK entrypoints."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from cuda.coop._core import ResultVisibility

from ._dsl._launch import infer_launch_facts
from ._internal import ThreadData
from ._internal._thread_data import _coerce_thread_payload
from ._thread_group import ThreadGroup, _resolve_collective_group_from_launch
from ._value_metadata import (
    attach_thread_data_metadata,
    metadata_for_group,
    validate_operand_domains,
)

_SCOPE = __name__.rsplit(".", 1)[0]


@contextmanager
def _topk_temp_storage_context(
    primitive_name: str,
    temp_storage: Any,
    *,
    provider: Any,
    block_threads: int,
    keys: Any,
    values: Any | None = None,
):
    if temp_storage is None:
        yield
        return

    from ._dsl._dispatch import single_phase_transaction
    from ._dsl._single_phase import SinglePhaseContext, activate_single_phase_context
    from ._temp_storage import TempStorage

    if not isinstance(temp_storage, TempStorage):
        raise TypeError(
            f"{_SCOPE}.{primitive_name} expected temp_storage to be TempStorage or None"
        )
    if temp_storage.is_deferred:
        raise NotImplementedError(
            f"{_SCOPE}.{primitive_name} does not yet support inferred "
            "TempStorage; deferred planning is currently limited to "
            "CUTLASS group-first Load, Store, Exchange, Scan, "
            "AdjacentDifference, Discontinuity, RadixSort, and MergeSort"
        )

    if values is None:
        if isinstance(keys, ThreadData):
            key_type, _ = provider._resolve_topk_thread_data_value_type(
                keys,
                allowed=provider._TOPK_KEY_TYPES,
                feature="topk_keys",
            )
            items_per_thread = keys.items_per_thread
        else:
            key_type = provider._resolve_topk_type(
                keys,
                allowed=provider._TOPK_KEY_TYPES,
                feature="topk_keys",
            )
            items_per_thread = 1
        value_type = None
    elif isinstance(keys, ThreadData) or isinstance(values, ThreadData):
        key_type, _, key_data, value_type, _, _ = (
            provider._resolve_topk_thread_data_pair_types(
                key=keys,
                value=values,
                allowed_key_types=provider._TOPK_KEY_TYPES,
                allowed_value_types=provider._TOPK_VALUE_TYPES,
                feature="topk_pairs",
            )
        )
        items_per_thread = key_data.items_per_thread
    else:
        key_type = provider._resolve_topk_type(
            keys,
            allowed=provider._TOPK_KEY_TYPES,
            feature="topk_pairs",
        )
        value_type = provider._resolve_topk_type(
            values,
            allowed=provider._TOPK_VALUE_TYPES,
            feature="topk_pairs",
        )
        items_per_thread = 1
    required_size, required_alignment = provider._topk_temp_storage_requirement(
        block_threads=block_threads,
        items_per_thread=items_per_thread,
        key_type=key_type,
        value_type=value_type,
    )
    single_phase_context = SinglePhaseContext(
        thread_data=None,
        temp_storage=temp_storage,
    )
    with single_phase_transaction(
        single_phase_context,
        snapshot_provider_session=provider._snapshot_active_session_state,
        restore_provider_session=provider._restore_active_session_state,
    ):
        temp_storage.record_use(
            primitive_name,
            required_size_in_bytes=required_size,
            required_alignment=required_alignment,
        )
        with activate_single_phase_context(single_phase_context):
            yield


def _resolve_topk_group(group: ThreadGroup) -> tuple[ThreadGroup, Any]:
    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_SCOPE}.topk group must be a ThreadGroup")
    if group.kind != "block":
        raise NotImplementedError(
            f"{_SCOPE}.topk currently lowers only this_block groups"
        )

    launch = infer_launch_facts(
        {},
        scope=_SCOPE,
        primitive_name="topk",
    )
    resolved = _resolve_collective_group_from_launch(
        group,
        launch,
        feature="topk",
    )
    assert resolved.hierarchy is not None
    block_dim = resolved.hierarchy.block_dim
    if block_dim is None or block_dim[1:] != (1, 1):
        raise NotImplementedError(
            f"{_SCOPE}.topk currently supports one-dimensional block groups only"
        )
    block_threads = resolved.hierarchy.block_thread_count
    assert block_threads is not None
    if block_threads > 1024:
        raise ValueError(f"{_SCOPE}.topk block thread count must be <= 1024")
    return resolved, launch


def _attach_topk_metadata(value: Any, group: ThreadGroup) -> Any:
    if not isinstance(value, ThreadData):
        return value
    return attach_thread_data_metadata(
        value,
        metadata_for_group(group, visibility=ResultVisibility.PER_MEMBER),
    )


def _topk_keys(
    group: ThreadGroup,
    keys: Any,
    k: Any,
    /,
    *,
    valid_items: Any,
    begin_bit: Any,
    end_bit: Any | None,
    temp_storage: Any,
    descending: bool,
) -> Any:
    group, _ = _resolve_topk_group(group)
    keys = _coerce_thread_payload(
        keys,
        scope=_SCOPE,
        primitive_name="topk_max_keys" if descending else "topk_min_keys",
        arg_name="keys",
    )
    validate_operand_domains(
        group,
        {"keys": keys},
        scope=_SCOPE,
        primitive_name="topk_max_keys" if descending else "topk_min_keys",
    )
    assert group.hierarchy is not None
    assert group.hierarchy.block_thread_count is not None

    from ._dsl.block import _provider

    primitive_name = "topk_max_keys" if descending else "topk_min_keys"
    with _topk_temp_storage_context(
        primitive_name,
        temp_storage,
        provider=_provider,
        block_threads=group.hierarchy.block_thread_count,
        keys=keys,
    ):
        result = _provider.provider_topk_keys(
            key=keys,
            k=k,
            num_valid=valid_items,
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=descending,
            block_threads=group.hierarchy.block_thread_count,
            temp_storage_primitive_name=primitive_name,
        )
    return _attach_topk_metadata(result, group)


def _topk_pairs(
    group: ThreadGroup,
    keys: Any,
    values: Any,
    k: Any,
    /,
    *,
    valid_items: Any,
    begin_bit: Any,
    end_bit: Any | None,
    temp_storage: Any,
    descending: bool,
) -> tuple[Any, Any]:
    group, _ = _resolve_topk_group(group)
    primitive_name = "topk_max_pairs" if descending else "topk_min_pairs"
    keys = _coerce_thread_payload(
        keys,
        scope=_SCOPE,
        primitive_name=primitive_name,
        arg_name="keys",
        common_root_payload_kind="thread_data",
    )
    values = _coerce_thread_payload(
        values,
        scope=_SCOPE,
        primitive_name=primitive_name,
        arg_name="values",
        common_root_payload_kind="thread_data",
    )
    validate_operand_domains(
        group,
        {"keys": keys, "values": values},
        scope=_SCOPE,
        primitive_name=primitive_name,
    )
    assert group.hierarchy is not None
    assert group.hierarchy.block_thread_count is not None

    from ._dsl.block import _provider

    with _topk_temp_storage_context(
        primitive_name,
        temp_storage,
        provider=_provider,
        block_threads=group.hierarchy.block_thread_count,
        keys=keys,
        values=values,
    ):
        result_keys, result_values = _provider.provider_topk_pairs(
            key=keys,
            value=values,
            k=k,
            num_valid=valid_items,
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=descending,
            block_threads=group.hierarchy.block_thread_count,
            temp_storage_primitive_name=primitive_name,
        )
    return (
        _attach_topk_metadata(result_keys, group),
        _attach_topk_metadata(result_values, group),
    )


def topk_max_keys(
    group: ThreadGroup,
    keys: Any,
    k: Any,
    /,
    *,
    valid_items: Any = None,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    temp_storage: Any = None,
) -> Any:
    """Select the largest keys with optional caller-owned reusable scratch."""

    return _topk_keys(
        group,
        keys,
        k,
        valid_items=valid_items,
        begin_bit=begin_bit,
        end_bit=end_bit,
        temp_storage=temp_storage,
        descending=True,
    )


def topk_min_keys(
    group: ThreadGroup,
    keys: Any,
    k: Any,
    /,
    *,
    valid_items: Any = None,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    temp_storage: Any = None,
) -> Any:
    """Select the smallest keys with optional caller-owned reusable scratch."""

    return _topk_keys(
        group,
        keys,
        k,
        valid_items=valid_items,
        begin_bit=begin_bit,
        end_bit=end_bit,
        temp_storage=temp_storage,
        descending=False,
    )


def topk_max_pairs(
    group: ThreadGroup,
    keys: Any,
    values: Any,
    k: Any,
    /,
    *,
    valid_items: Any = None,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    temp_storage: Any = None,
) -> tuple[Any, Any]:
    """Select largest-key pairs with optional caller-owned reusable scratch."""

    return _topk_pairs(
        group,
        keys,
        values,
        k,
        valid_items=valid_items,
        begin_bit=begin_bit,
        end_bit=end_bit,
        temp_storage=temp_storage,
        descending=True,
    )


def topk_min_pairs(
    group: ThreadGroup,
    keys: Any,
    values: Any,
    k: Any,
    /,
    *,
    valid_items: Any = None,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    temp_storage: Any = None,
) -> tuple[Any, Any]:
    """Select smallest-key pairs with optional caller-owned reusable scratch."""

    return _topk_pairs(
        group,
        keys,
        values,
        k,
        valid_items=valid_items,
        begin_bit=begin_bit,
        end_bit=end_bit,
        temp_storage=temp_storage,
        descending=False,
    )


__all__ = [
    "topk_max_keys",
    "topk_max_pairs",
    "topk_min_keys",
    "topk_min_pairs",
]
