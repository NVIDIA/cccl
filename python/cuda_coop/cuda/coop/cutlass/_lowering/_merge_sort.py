# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plan-driven public-CUB provider for CUTLASS block and warp MergeSort."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np
from cutlass import cute as _cute
from cutlass.cute.ffi import ffi

from cuda.coop._core import (
    AlgorithmSpec,
    CxxOperator,
    Dependency,
    GroupLoweringPlan,
    GroupLoweringTarget,
    GroupMergeSortSemantics,
    LaunchFacts,
    make_group_primitive_call,
    plan_group_primitive,
)
from cuda.coop._core.block import make_block_merge_sort_semantics

from .._compiler import _state as _provider_state
from .._compiler import _storage as _provider_storage
from .._compiler import _types as _provider_types
from .._compiler._types import ALL_PROVIDER_TYPES as _ALL_PROVIDER_TYPES
from .._compiler._types import TYPE_SPECS as _TYPE_SPECS
from .._thread_data import ThreadData
from .._thread_group import ThreadGroup
from .._value_metadata import (
    attach_thread_data_metadata,
    metadata_for_group,
    validate_operand_domains,
)
from ._core import (
    CutlassCoreAdapter,
    CutlassCoreArtifact,
    CutlassRuntimeIntRange,
    register_cutlass_core_renderer,
    with_caller_owned_core_temp_storage,
)
from ._symbols import block_dim_token as _block_dim_token

_ROOT_SCOPE = "cuda.coop.cutlass"
_BLOCK_REQUEST_KIND = "cuda_coop_cutlass_cub_block_merge_sort"
_WARP_REQUEST_KIND = "cuda_coop_cutlass_cub_warp_merge_sort"

register_cutlass_core_renderer(
    _BLOCK_REQUEST_KIND,
    includes=("cub/block/block_merge_sort.cuh",),
)
register_cutlass_core_renderer(
    _WARP_REQUEST_KIND,
    includes=("cub/warp/warp_merge_sort.cuh",),
)

_resolve_type = _provider_state.make_provider_type_resolver(
    scope=_ROOT_SCOPE,
    root_scope=_ROOT_SCOPE,
    namespace="thread_group",
)


def _compare_operator(descending: bool) -> CxxOperator:
    name = "greater" if descending else "less"
    return CxxOperator(
        f"::cuda::std::{name}<KeyT>",
        Dependency("KeyT"),
        name="compare_op",
    )


def _make_group_merge_sort_plan(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    key_dtype: Any,
    value_dtype: Any | None,
    items_per_thread: int,
    descending: bool,
    valid_items: Any = None,
    oob_default: Any = None,
    source: str = "cutlass_root",
) -> GroupLoweringPlan:
    """Build the canonical shared-core block or warp MergeSort plan."""

    primitive = make_block_merge_sort_semantics(
        key_dtype=key_dtype,
        value_dtype=value_dtype,
        items_per_thread=items_per_thread,
        compare_operator=_compare_operator(descending),
        valid_items=valid_items,
        oob_default=oob_default,
    )
    call = make_group_primitive_call(
        group,
        GroupMergeSortSemantics(primitive),
        source=source,
    )
    return plan_group_primitive(call, launch)


_ORDINARY_MERGE_SORT_DTYPES = {
    int: _provider_types.Int32,
    float: _provider_types.Float32,
    np.uint8: _provider_types.Uint8,
    np.int32: _provider_types.Int32,
    np.uint32: _provider_types.Uint32,
    np.int64: _provider_types.Int64,
    np.uint64: _provider_types.Uint64,
    np.float32: _provider_types.Float32,
    np.float64: _provider_types.Float64,
}


def _resolve_merge_sort_type(
    value: Any,
    *,
    allowed: frozenset[type],
    feature: str,
) -> type:
    if isinstance(value, np.dtype):
        value = value.type
    candidate = value if isinstance(value, type) else type(value)
    value = _ORDINARY_MERGE_SORT_DTYPES.get(candidate, value)
    return _resolve_type(value, allowed=allowed, feature=feature)


def _symbol_name(
    *,
    plan,
    key_type: type,
    value_type: type | None,
    descending: bool,
) -> str:
    operation = plan.call.operation
    assert isinstance(operation, GroupMergeSortSemantics)
    primitive = operation.primitive
    participation = plan.participation
    if participation is None or participation.exact_block_dim is None:
        raise ValueError("MergeSort symbols require exact block dimensions")
    block_token = _block_dim_token(participation.exact_block_dim)
    if plan.target is GroupLoweringTarget.CUB_BLOCK:
        group_token = f"block_{block_token}"
    else:
        group_token = f"warp_{block_token}_w{plan.resolved_group.static_size}"
    payload = "pairs" if primitive.has_values else "keys"
    order = "descending" if descending else "ascending"
    type_token = f"k{_TYPE_SPECS[key_type].token}"
    if value_type is not None:
        type_token += f"_v{_TYPE_SPECS[value_type].token}"
    tile = "partial" if primitive.has_partial_tile else "full"
    return (
        f"cuda_coop_cutlass_cub_merge_sort_{group_token}_{payload}_{order}_"
        f"{type_token}_x{primitive.items_per_thread}_{tile}"
    )


def _make_request(
    *,
    group: ThreadGroup,
    launch,
    key_type: type,
    value_type: type | None,
    items_per_thread: int,
    descending: bool,
    valid_items: Any,
    oob_default: Any,
    source: str,
    external_scratch: bool = False,
) -> CutlassCoreArtifact:
    plan = _make_group_merge_sort_plan(
        group=group,
        launch=launch,
        key_dtype=key_type,
        value_dtype=value_type,
        items_per_thread=items_per_thread,
        descending=descending,
        valid_items=valid_items,
        oob_default=oob_default,
        source=source,
    ).require_supported()
    if not isinstance(plan.implementation, AlgorithmSpec):
        raise TypeError("MergeSort plan requires an AlgorithmSpec")
    if external_scratch:
        plan = with_caller_owned_core_temp_storage(plan)
    kind = (
        _BLOCK_REQUEST_KIND
        if plan.target is GroupLoweringTarget.CUB_BLOCK
        else _WARP_REQUEST_KIND
    )
    operation = plan.call.operation
    assert isinstance(operation, GroupMergeSortSemantics)
    runtime_int_ranges = ()
    if operation.primitive.has_partial_tile:
        group_size = plan.resolved_group.static_size
        assert group_size is not None
        runtime_int_ranges = (
            CutlassRuntimeIntRange(
                "valid_items",
                0,
                group_size * operation.primitive.items_per_thread,
            ),
        )
    return CutlassCoreAdapter().materialize(
        plan.implementation,
        plan=plan,
        kind=kind,
        symbol_name=_symbol_name(
            plan=plan,
            key_type=key_type,
            value_type=value_type,
            descending=descending,
        ),
        runtime_int_ranges=runtime_int_ranges,
        external_scratch=external_scratch,
    )


def _coerce_oob_default(value: Any, key_type: type) -> Any:
    if value is None:
        return value
    plain_value = _provider_types.coerce_plain_scalar(
        value,
        key_type,
        name="merge_sort oob_default",
        scope=_ROOT_SCOPE,
        allow_nonfinite=False,
    )
    if plain_value is not _provider_types._NOT_PLAIN_SCALAR:
        return plain_value
    try:
        actual_type = _resolve_merge_sort_type(
            value,
            allowed=_ALL_PROVIDER_TYPES,
            feature="merge_sort oob_default",
        )
    except (TypeError, NotImplementedError) as exc:
        raise TypeError(
            f"{_ROOT_SCOPE}.merge_sort oob_default must match the key dtype "
            f"{key_type.__name__}"
        ) from exc
    if actual_type is not key_type:
        raise TypeError(
            f"{_ROOT_SCOPE}.merge_sort oob_default must match the key dtype "
            f"{key_type.__name__}"
        )
    if isinstance(value, key_type):
        return value
    if isinstance(value, np.generic):
        value = value.item()
    try:
        return key_type(value)
    except Exception as exc:
        raise TypeError(
            f"{_ROOT_SCOPE}.merge_sort oob_default cannot be converted to "
            f"{key_type.__name__}"
        ) from exc


def _validate_valid_items(
    value: Any,
    *,
    group: ThreadGroup,
    items_per_thread: int,
) -> Any:
    if value is None:
        return None
    from cutlass.base_dsl.typing import Boolean, Integer

    if isinstance(value, (bool, np.bool_, Boolean)):
        raise TypeError("valid_items must be an integer, not bool")
    if isinstance(value, Integral):
        value = int(value)
        group_size = group.static_size
        assert group_size is not None
        maximum = group_size * items_per_thread
        if value < 0 or value > maximum:
            raise ValueError(f"valid_items must be in [0, {maximum}]")
    elif not isinstance(value, Integer):
        raise TypeError("valid_items must be an integer")
    return _provider_types.as_valid_items_arg(value, scope=_ROOT_SCOPE)


def _temp_storage_for_merge_sort(
    *,
    group: ThreadGroup,
    source: str,
    explicit_temp_storage: Any,
) -> Any | None:
    """Resolve caller-owned scratch for a group-first block Merge Sort."""

    from .._compiler._call_context import get_active_single_phase_context

    context = get_active_single_phase_context()
    context_storage = context.temp_storage if context is not None else None
    if (
        explicit_temp_storage is not None
        and context_storage is not None
        and explicit_temp_storage is not context_storage
    ):
        raise ValueError(f"{_ROOT_SCOPE}.merge_sort received two TempStorage objects")
    temp_storage = (
        explicit_temp_storage if explicit_temp_storage is not None else context_storage
    )
    if temp_storage is None:
        return None

    from .._temp_storage import TempStorage

    if not isinstance(temp_storage, TempStorage):
        raise TypeError(
            f"{_ROOT_SCOPE}.merge_sort temp_storage must be {_ROOT_SCOPE}.TempStorage"
        )
    if group.kind != "block":
        raise ValueError(
            f"{_ROOT_SCOPE}.merge_sort TempStorage is supported only for block groups"
        )
    if source != "cutlass_root":
        raise ValueError(
            f"{_ROOT_SCOPE}.merge_sort TempStorage requires a group-first call"
        )
    if not temp_storage.is_deferred and temp_storage.sharing == "exclusive":
        raise ValueError(
            f"{_ROOT_SCOPE}.merge_sort fixed-capacity TempStorage does not "
            "support sharing='exclusive'; use sharing='shared' or deferred "
            "storage"
        )
    return temp_storage


def _external_scratch_args(
    temp_storage: Any,
    *,
    requirement_key: Any,
) -> tuple[Any, Any, Any]:
    if temp_storage.is_deferred:
        return _provider_storage.register_deferred_temp_storage_event(
            temp_storage,
            primitive_name="merge_sort",
            requirement_key=requirement_key,
        )

    binding = _provider_storage.materialize_temp_storage_binding(
        temp_storage,
        scope=_ROOT_SCOPE,
        implicit_alignment=16,
    )
    return (
        binding.smem_addr_u32,
        _provider_types.Int32(binding.size_in_bytes),
        _provider_types.Int32(1 if binding.auto_sync else 0),
    )


def _resolve_inputs(
    *,
    group: ThreadGroup,
    keys: Any,
    values: Any | None,
) -> tuple[
    type,
    tuple[Any, ...],
    ThreadData | None,
    type | None,
    tuple[Any, ...] | None,
    ThreadData | None,
]:
    allowed_key_types = _ALL_PROVIDER_TYPES
    if group.kind != "block" and not isinstance(keys, ThreadData):
        raise TypeError(f"{_ROOT_SCOPE}.merge_sort warp keys must be ThreadData")

    if values is not None and (
        isinstance(keys, ThreadData) or isinstance(values, ThreadData)
    ):
        (
            key_type,
            key_values,
            key_data,
            value_type,
            value_values,
            value_data,
        ) = _provider_types.resolve_thread_data_pair_types(
            key=keys,
            value=values,
            allowed_key_types=allowed_key_types,
            allowed_value_types=_ALL_PROVIDER_TYPES,
            feature="merge_sort_pairs",
            scope=_ROOT_SCOPE,
            resolve_type=_resolve_merge_sort_type,
        )
        return (
            key_type,
            tuple(key_values),
            key_data,
            value_type,
            tuple(value_values),
            value_data,
        )

    if isinstance(keys, ThreadData):
        key_type, key_values = _provider_types.resolve_thread_data_value_type(
            keys,
            allowed=allowed_key_types,
            feature="merge_sort_keys",
            scope=_ROOT_SCOPE,
            resolve_type=_resolve_merge_sort_type,
        )
        return key_type, tuple(key_values), keys, None, None, None

    key_type = _resolve_merge_sort_type(
        keys,
        allowed=allowed_key_types,
        feature="merge_sort_keys" if values is None else "merge_sort_pairs",
    )
    if values is None:
        return key_type, (keys,), None, None, None, None
    value_type = _resolve_merge_sort_type(
        values,
        allowed=_ALL_PROVIDER_TYPES,
        feature="merge_sort_pairs",
    )
    return key_type, (keys,), None, value_type, (values,), None


def provider_merge_sort(
    *,
    group: ThreadGroup,
    launch,
    keys: Any,
    values: Any | None,
    descending: bool,
    valid_items: Any = None,
    oob_default: Any = None,
    source: str = "cutlass_group_merge_sort_provider",
    temp_storage: Any = None,
) -> Any:
    """Materialize one public-CUB Sort call for keys or associated pairs."""

    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_ROOT_SCOPE}.merge_sort group must be a ThreadGroup")
    if not isinstance(descending, bool):
        raise TypeError("descending must be a static bool")
    if (valid_items is None) != (oob_default is None):
        raise ValueError("valid_items and oob_default must be provided together")
    (
        key_type,
        key_values,
        key_data,
        value_type,
        value_values,
        value_data,
    ) = _resolve_inputs(group=group, keys=keys, values=values)
    items_per_thread = len(key_values)
    valid_items_arg = _validate_valid_items(
        valid_items,
        group=group,
        items_per_thread=items_per_thread,
    )
    oob_default = _coerce_oob_default(oob_default, key_type)
    external_temp_storage = _temp_storage_for_merge_sort(
        group=group,
        source=source,
        explicit_temp_storage=temp_storage,
    )
    request = _make_request(
        group=group,
        launch=launch,
        key_type=key_type,
        value_type=value_type,
        items_per_thread=items_per_thread,
        descending=descending,
        valid_items=valid_items,
        oob_default=oob_default,
        source=source,
        external_scratch=external_temp_storage is not None,
    )
    validate_operand_domains(
        request.plan.resolved_group,
        {
            "keys": keys,
            **({"values": values} if values is not None else {}),
        },
        scope=_ROOT_SCOPE,
        primitive_name="merge_sort",
    )
    key_result = _cute.make_rmem_tensor(items_per_thread, key_type)
    output_values = {"keys": key_result.iterator.llvm_ptr}
    runtime_values: dict[str, Any] = {"keys": key_values}
    value_result = None
    if value_type is not None:
        assert value_values is not None
        value_result = _cute.make_rmem_tensor(items_per_thread, value_type)
        runtime_values["values"] = value_values
        output_values["values"] = value_result.iterator.llvm_ptr
    if valid_items_arg is not None:
        runtime_values["valid_items"] = valid_items_arg
        runtime_values["oob_default"] = oob_default

    session_snapshot = (
        _provider_state.snapshot_active_session_state()
        if external_temp_storage is not None
        else None
    )
    try:
        _provider_state.register_request(request)
        scratch_values = ()
        if external_temp_storage is not None:
            scratch_values = _external_scratch_args(
                external_temp_storage,
                requirement_key=request.scratch_requirement_key,
            )
        arguments = request.bind_ffi_arguments(
            runtime_values,
            output_values,
            scratch_values=scratch_values,
        )
        ffi(
            name=request.symbol_name,
            params_types=list(request.ffi_param_types),
            return_type=None,
        )(*arguments)
    except Exception:
        if session_snapshot is not None:
            _provider_state.restore_active_session_state(session_snapshot)
        raise

    assert request.plan.result is not None
    result_metadata = metadata_for_group(
        request.plan.resolved_group,
        visibility=request.plan.result.visibility,
    )
    key_results = tuple(key_result[index] for index in range(items_per_thread))
    if key_data is None:
        sorted_keys = _provider_state.remember_scalar_result_type(
            key_results[0],
            key_type,
            scope=_ROOT_SCOPE,
            compile_options_getter=(
                lambda: _provider_state._get_cute_dsl().compile_options
            ),
            group_metadata=result_metadata,
        )
    else:
        sorted_keys = attach_thread_data_metadata(
            ThreadData.from_values(
                *key_results,
                dtype=_provider_types.thread_data_output_dtype(key_data, key_type),
            ),
            result_metadata,
        )
    if value_result is None:
        return sorted_keys

    assert value_type is not None
    value_results = tuple(value_result[index] for index in range(items_per_thread))
    if value_data is None:
        sorted_values = _provider_state.remember_scalar_result_type(
            value_results[0],
            value_type,
            scope=_ROOT_SCOPE,
            compile_options_getter=(
                lambda: _provider_state._get_cute_dsl().compile_options
            ),
            group_metadata=result_metadata,
        )
    else:
        sorted_values = attach_thread_data_metadata(
            ThreadData.from_values(
                *value_results,
                dtype=_provider_types.thread_data_output_dtype(
                    value_data,
                    value_type,
                ),
            ),
            result_metadata,
        )
    return sorted_keys, sorted_values


__all__ = ["CutlassCoreArtifact", "provider_merge_sort"]
