# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plan-driven public-CUB providers for CUTLASS block radix primitives."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
from cutlass import cute as _cute
from cutlass.base_dsl.typing import Int32, Int64, Uint32, Uint64
from cutlass.cute.ffi import ffi

from cuda.coop._core import (
    AlgorithmSpec,
    GroupLoweringPlan,
    GroupOperandKind,
    GroupRadixRankSemantics,
    GroupRadixSortSemantics,
    LaunchFacts,
    RuntimeValue,
    make_group_primitive_call,
    plan_group_primitive,
)
from cuda.coop._core.block import (
    block_radix_rank_bins_per_thread,
    make_block_radix_rank_semantics,
    make_block_radix_sort_semantics,
)

from .._compiler import _state as _provider_state
from .._compiler import _storage as _provider_storage
from .._compiler import _types as _provider_types
from .._compiler._types import ALL_PROVIDER_TYPES, RADIX_KEY_TYPES, TYPE_SPECS
from .._thread_data import ThreadData
from .._thread_group import ThreadGroup
from .._value_metadata import (
    attach_thread_data_metadata,
    metadata_for_group,
    validate_operand_domains,
)
from ._core import (
    CutlassArrayInputTransform,
    CutlassCoreAdapter,
    CutlassCoreArtifact,
    CutlassRuntimeIntRange,
    register_cutlass_core_renderer,
    with_caller_owned_core_temp_storage,
)
from ._symbols import block_dim_token as _block_dim_token

_ROOT_SCOPE = "cuda.coop.cutlass"
_SORT_REQUEST_KIND = "cuda_coop_cutlass_cub_block_radix_sort"
_RANK_REQUEST_KIND = "cuda_coop_cutlass_cub_block_radix_rank"
_WIDE_RANK_SMEM_CONFIG = "cudaSharedMemBankSizeEightByte"

register_cutlass_core_renderer(
    _SORT_REQUEST_KIND,
    includes=("cub/block/block_radix_sort.cuh",),
)
register_cutlass_core_renderer(
    _RANK_REQUEST_KIND,
    includes=("cub/block/block_radix_rank.cuh",),
)

_resolve_type = _provider_state.make_provider_type_resolver(
    scope=_ROOT_SCOPE,
    root_scope=_ROOT_SCOPE,
    namespace="thread_group",
)


def _make_group_radix_sort_plan(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    key_dtype: Any,
    value_dtype: Any | None,
    items_per_thread: int,
    operand_kind: GroupOperandKind,
    descending: bool,
    key_bit_width: int,
    source: str,
) -> GroupLoweringPlan:
    primitive = make_block_radix_sort_semantics(
        key_dtype=key_dtype,
        value_dtype=value_dtype,
        items_per_thread=items_per_thread,
        descending=descending,
        begin_bit=RuntimeValue("begin_bit"),
        end_bit=RuntimeValue("end_bit"),
        key_bit_width=key_bit_width,
        bit_policy="explicit",
    )
    return plan_group_primitive(
        make_group_primitive_call(
            group,
            GroupRadixSortSemantics(primitive, operand_kind=operand_kind),
            source=source,
        ),
        launch,
    )


def _make_group_radix_rank_plan(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    cub_key_dtype: Any,
    input_dtype: Any,
    items_per_thread: int,
    operand_kind: GroupOperandKind,
    begin_bit: int,
    end_bit: int,
    key_bit_width: int,
    descending: bool,
    exclusive_digit_prefix_items_per_thread: int | None,
    source: str,
) -> GroupLoweringPlan:
    assert launch.exact_block_threads is not None
    primitive = make_block_radix_rank_semantics(
        key_dtype=cub_key_dtype,
        items_per_thread=items_per_thread,
        begin_bit=begin_bit,
        end_bit=end_bit,
        key_bit_width=key_bit_width,
        descending=descending,
        block_threads=launch.exact_block_threads,
        exclusive_digit_prefix_items_per_thread=(
            exclusive_digit_prefix_items_per_thread
        ),
    )
    plan = plan_group_primitive(
        make_group_primitive_call(
            group,
            GroupRadixRankSemantics(
                primitive,
                input_dtype=input_dtype,
                operand_kind=operand_kind,
            ),
            source=source,
        ),
        launch,
    )
    if plan.unsupported is not None:
        return plan
    implementation = plan.implementation
    if not isinstance(implementation, AlgorithmSpec):
        raise TypeError("CUTLASS radix rank plans require an AlgorithmSpec")
    radix_bits = implementation.template_arguments["RADIX_BITS"]
    if radix_bits < 8:
        return plan

    template_arguments = dict(implementation.template_arguments)
    template_arguments["SMEM_CONFIG"] = _WIDE_RANK_SMEM_CONFIG
    implementation = implementation.algorithm.specialize(
        template_arguments,
        metadata=implementation.metadata,
    )
    return dataclasses.replace(plan, implementation=implementation)


_ORDINARY_RADIX_KEY_DTYPES = {
    int: Int32,
    np.int32: Int32,
    np.uint32: Uint32,
    np.int64: Int64,
    np.uint64: Uint64,
}


def _resolve_radix_key_type(
    value: Any,
    *,
    allowed: frozenset[type],
    feature: str,
) -> type:
    if isinstance(value, np.dtype):
        value = value.type
    candidate = value if isinstance(value, type) else type(value)
    value = _ORDINARY_RADIX_KEY_DTYPES.get(candidate, value)
    return _resolve_type(value, allowed=allowed, feature=feature)


_RANK_CUB_TYPES = {
    Int32: Uint32,
    Uint32: Uint32,
    Int64: Uint64,
    Uint64: Uint64,
}
_RANK_INPUT_TRANSFORMS = {
    Int32: CutlassArrayInputTransform(
        source_dtype=Int32,
        cpp_expression=("(static_cast<unsigned int>({value}) ^ 0x80000000u)"),
    ),
    Int64: CutlassArrayInputTransform(
        source_dtype=Int64,
        cpp_expression=(
            "(static_cast<unsigned long long>({value}) ^ 0x8000000000000000ull)"
        ),
    ),
}


def _operand(
    value: Any,
    *,
    allowed: frozenset[type],
    feature: str,
) -> tuple[type, tuple[Any, ...], GroupOperandKind, ThreadData | None]:
    if isinstance(value, ThreadData):
        value_type, values = _provider_types.resolve_thread_data_value_type(
            value,
            allowed=allowed,
            feature=feature,
            scope=_ROOT_SCOPE,
            resolve_type=_resolve_radix_key_type,
        )
        return value_type, tuple(values), GroupOperandKind.ARRAY, value
    value_type = _resolve_radix_key_type(
        value,
        allowed=allowed,
        feature=feature,
    )
    return value_type, (value,), GroupOperandKind.SCALAR, None


def _sort_symbol(
    *,
    plan,
    key_type: type,
    value_type: type | None,
    items_per_thread: int,
    descending: bool,
) -> str:
    participation = plan.participation
    if participation is None or participation.exact_block_dim is None:
        raise ValueError("BlockRadixSort symbols require exact block dimensions")
    payload = "keys" if value_type is None else f"pairs_{TYPE_SPECS[value_type].token}"
    order = "desc" if descending else "asc"
    name = (
        f"cuda_coop_cutlass_radix_sort_{payload}_"
        f"{_block_dim_token(participation.exact_block_dim)}_"
        f"{TYPE_SPECS[key_type].token}_{order}"
    )
    if items_per_thread > 1:
        name += f"_x{items_per_thread}"
    return name


def _rank_symbol(
    *,
    plan,
    input_type: type,
    items_per_thread: int,
    begin_bit: int,
    end_bit: int,
    descending: bool,
    prefix_items: int | None,
) -> str:
    participation = plan.participation
    if participation is None or participation.exact_block_dim is None:
        raise ValueError("BlockRadixRank symbols require exact block dimensions")
    order = "desc" if descending else "asc"
    name = (
        "cuda_coop_cutlass_radix_rank_"
        f"{_block_dim_token(participation.exact_block_dim)}_"
        f"{TYPE_SPECS[input_type].token}_{order}_b{begin_bit}_{end_bit}"
    )
    if items_per_thread > 1:
        name += f"_x{items_per_thread}"
    if prefix_items is not None:
        name += f"_prefix{prefix_items}"
    return name


def _make_sort_request(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    key_type: type,
    value_type: type | None,
    items_per_thread: int,
    operand_kind: GroupOperandKind,
    descending: bool,
    source: str,
    external_scratch: bool = False,
) -> CutlassCoreArtifact:
    plan = _make_group_radix_sort_plan(
        group=group,
        launch=launch,
        key_dtype=key_type,
        value_dtype=value_type,
        items_per_thread=items_per_thread,
        operand_kind=operand_kind,
        descending=descending,
        key_bit_width=TYPE_SPECS[key_type].width_bits,
        source=source,
    ).require_supported()
    if not isinstance(plan.implementation, AlgorithmSpec):
        raise TypeError("BlockRadixSort plan requires an AlgorithmSpec")
    if external_scratch:
        plan = with_caller_owned_core_temp_storage(plan)
    key_bit_width = TYPE_SPECS[key_type].width_bits
    return CutlassCoreAdapter().materialize(
        plan.implementation,
        plan=plan,
        kind=_SORT_REQUEST_KIND,
        symbol_name=_sort_symbol(
            plan=plan,
            key_type=key_type,
            value_type=value_type,
            items_per_thread=items_per_thread,
            descending=descending,
        ),
        runtime_int_ranges=(
            CutlassRuntimeIntRange(
                "begin_bit",
                0,
                key_bit_width - 1,
                less_than_parameter="end_bit",
            ),
            CutlassRuntimeIntRange("end_bit", 1, key_bit_width),
        ),
        external_scratch=external_scratch,
    )


def _make_rank_request(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    input_type: type,
    items_per_thread: int,
    operand_kind: GroupOperandKind,
    begin_bit: int,
    end_bit: int,
    descending: bool,
    prefix_items: int | None,
    source: str,
) -> CutlassCoreArtifact:
    cub_key_type = _RANK_CUB_TYPES[input_type]
    plan = _make_group_radix_rank_plan(
        group=group,
        launch=launch,
        cub_key_dtype=cub_key_type,
        input_dtype=input_type,
        items_per_thread=items_per_thread,
        operand_kind=operand_kind,
        begin_bit=begin_bit,
        end_bit=end_bit,
        key_bit_width=TYPE_SPECS[input_type].width_bits,
        descending=descending,
        exclusive_digit_prefix_items_per_thread=prefix_items,
        source=source,
    ).require_supported()
    if not isinstance(plan.implementation, AlgorithmSpec):
        raise TypeError("BlockRadixRank plan requires an AlgorithmSpec")
    transform = _RANK_INPUT_TRANSFORMS.get(input_type)
    return CutlassCoreAdapter().materialize(
        plan.implementation,
        plan=plan,
        kind=_RANK_REQUEST_KIND,
        symbol_name=_rank_symbol(
            plan=plan,
            input_type=input_type,
            items_per_thread=items_per_thread,
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=descending,
            prefix_items=prefix_items,
        ),
        input_transforms={} if transform is None else {"keys": transform},
        output_initializers=(
            {} if prefix_items is None else {"exclusive_digit_prefix": "-1"}
        ),
    )


def _result_metadata(request: CutlassCoreArtifact, index: int = 0):
    assert request.plan.result is not None
    return metadata_for_group(
        request.plan.resolved_group,
        visibility=request.plan.result.values[index].visibility,
    )


def _pack_result(
    result_tensor: Any,
    *,
    items_per_thread: int,
    input_thread_data: ThreadData | None,
    result_type: type,
    metadata: Any,
) -> Any:
    values = tuple(result_tensor[index] for index in range(items_per_thread))
    if input_thread_data is not None:
        return attach_thread_data_metadata(
            ThreadData.from_values(*values, dtype=result_type),
            metadata,
        )
    return _provider_state.remember_scalar_result_type(
        values[0],
        result_type,
        scope=_ROOT_SCOPE,
        compile_options_getter=lambda: _provider_state._get_cute_dsl().compile_options,
        group_metadata=metadata,
    )


def _temp_storage_for_radix_sort(
    *,
    group: ThreadGroup,
    source: str,
    explicit_temp_storage: Any,
) -> Any | None:
    """Resolve caller-owned scratch for one group-first block radix sort."""

    from .._compiler._call_context import get_active_single_phase_context

    context = get_active_single_phase_context()
    context_storage = context.temp_storage if context is not None else None
    if (
        explicit_temp_storage is not None
        and context_storage is not None
        and explicit_temp_storage is not context_storage
    ):
        raise ValueError(f"{_ROOT_SCOPE}.radix_sort received two TempStorage objects")
    temp_storage = (
        explicit_temp_storage if explicit_temp_storage is not None else context_storage
    )
    if temp_storage is None:
        return None

    from .._temp_storage import TempStorage

    if not isinstance(temp_storage, TempStorage):
        raise TypeError(
            f"{_ROOT_SCOPE}.radix_sort temp_storage must be {_ROOT_SCOPE}.TempStorage"
        )
    if group.kind != "block":
        raise ValueError(
            f"{_ROOT_SCOPE}.radix_sort TempStorage is supported only for block groups"
        )
    if source != "cutlass_root":
        raise ValueError(
            f"{_ROOT_SCOPE}.radix_sort TempStorage requires a group-first call"
        )
    if not temp_storage.is_deferred and temp_storage.sharing == "exclusive":
        raise ValueError(
            f"{_ROOT_SCOPE}.radix_sort fixed-capacity TempStorage does not "
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
            primitive_name="radix_sort",
            requirement_key=requirement_key,
        )

    binding = _provider_storage.materialize_temp_storage_binding(
        temp_storage,
        scope=_ROOT_SCOPE,
        implicit_alignment=16,
    )
    return (
        binding.smem_addr_u32,
        Int32(binding.size_in_bytes),
        Int32(1 if binding.auto_sync else 0),
    )


def provider_radix_sort_keys(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    keys: Any,
    begin_bit: Any,
    end_bit: Any | None,
    descending: bool,
    source: str,
    temp_storage: Any = None,
) -> Any:
    key_type, key_values, operand_kind, key_data = _operand(
        keys,
        allowed=RADIX_KEY_TYPES,
        feature="radix_sort_keys",
    )
    resolved_end = _provider_types.validate_radix_bit_range(
        begin_bit,
        end_bit,
        key_type,
    )
    external_temp_storage = _temp_storage_for_radix_sort(
        group=group,
        source=source,
        explicit_temp_storage=temp_storage,
    )
    request = _make_sort_request(
        group=group,
        launch=launch,
        key_type=key_type,
        value_type=None,
        items_per_thread=len(key_values),
        operand_kind=operand_kind,
        descending=descending,
        source=source,
        external_scratch=external_temp_storage is not None,
    )
    validate_operand_domains(
        request.plan.resolved_group,
        {"keys": keys},
        scope=_ROOT_SCOPE,
        primitive_name="radix_sort_keys",
    )
    result = _cute.make_rmem_tensor(len(key_values), key_type)
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
            {
                "keys": key_values,
                "begin_bit": _provider_types.as_int32(begin_bit),
                "end_bit": _provider_types.as_int32(resolved_end),
            },
            {"keys": result.iterator.llvm_ptr},
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
    return _pack_result(
        result,
        items_per_thread=len(key_values),
        input_thread_data=key_data,
        result_type=_provider_types.thread_data_output_dtype(key_data, key_type)
        if key_data is not None
        else key_type,
        metadata=_result_metadata(request),
    )


def provider_radix_sort_pairs(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    keys: Any,
    values: Any,
    begin_bit: Any,
    end_bit: Any | None,
    descending: bool,
    source: str,
    temp_storage: Any = None,
) -> tuple[Any, Any]:
    key_is_data = isinstance(keys, ThreadData)
    value_is_data = isinstance(values, ThreadData)
    if key_is_data != value_is_data:
        raise TypeError(
            f"{_ROOT_SCOPE}.radix_sort_pairs requires both key and value to be "
            "ThreadData when one argument uses ThreadData"
        )
    key_type, key_values, operand_kind, key_data = _operand(
        keys,
        allowed=RADIX_KEY_TYPES,
        feature="radix_sort_pairs",
    )
    value_type, value_values, value_kind, value_data = _operand(
        values,
        allowed=ALL_PROVIDER_TYPES,
        feature="radix_sort_pairs",
    )
    if operand_kind is not value_kind:
        raise TypeError("radix_sort_pairs operands must use the same payload form")
    if len(key_values) != len(value_values):
        raise ValueError(
            f"{_ROOT_SCOPE}.radix_sort_pairs requires matching "
            "ThreadData.items_per_thread for key and value"
        )
    resolved_end = _provider_types.validate_radix_bit_range(
        begin_bit,
        end_bit,
        key_type,
    )
    external_temp_storage = _temp_storage_for_radix_sort(
        group=group,
        source=source,
        explicit_temp_storage=temp_storage,
    )
    request = _make_sort_request(
        group=group,
        launch=launch,
        key_type=key_type,
        value_type=value_type,
        items_per_thread=len(key_values),
        operand_kind=operand_kind,
        descending=descending,
        source=source,
        external_scratch=external_temp_storage is not None,
    )
    validate_operand_domains(
        request.plan.resolved_group,
        {"keys": keys, "values": values},
        scope=_ROOT_SCOPE,
        primitive_name="radix_sort_pairs",
    )
    key_result = _cute.make_rmem_tensor(len(key_values), key_type)
    value_result = _cute.make_rmem_tensor(len(value_values), value_type)
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
            {
                "keys": key_values,
                "values": value_values,
                "begin_bit": _provider_types.as_int32(begin_bit),
                "end_bit": _provider_types.as_int32(resolved_end),
            },
            {
                "keys": key_result.iterator.llvm_ptr,
                "values": value_result.iterator.llvm_ptr,
            },
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
    return (
        _pack_result(
            key_result,
            items_per_thread=len(key_values),
            input_thread_data=key_data,
            result_type=_provider_types.thread_data_output_dtype(key_data, key_type)
            if key_data is not None
            else key_type,
            metadata=_result_metadata(request, 0),
        ),
        _pack_result(
            value_result,
            items_per_thread=len(value_values),
            input_thread_data=value_data,
            result_type=_provider_types.thread_data_output_dtype(value_data, value_type)
            if value_data is not None
            else value_type,
            metadata=_result_metadata(request, 1),
        ),
    )


def provider_radix_rank(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    keys: Any,
    begin_bit: int,
    end_bit: int,
    descending: bool,
    exclusive_digit_prefix: Any,
    source: str,
) -> Any:
    key_type, key_values, operand_kind, key_data = _operand(
        keys,
        allowed=RADIX_KEY_TYPES,
        feature="radix_rank",
    )
    resolved_end = _provider_types.validate_radix_bit_range(
        begin_bit,
        end_bit,
        key_type,
    )
    assert isinstance(resolved_end, int)
    assert launch.exact_block_threads is not None
    prefix_items = None
    if exclusive_digit_prefix is not None:
        prefix_items = block_radix_rank_bins_per_thread(
            resolved_end - begin_bit,
            launch.exact_block_threads,
        )
        exclusive_digit_prefix = _provider_types.validate_thread_data_output(
            output=exclusive_digit_prefix,
            expected_items_per_thread=prefix_items,
            resolved_dtype=Int32,
            scope=_ROOT_SCOPE,
            primitive_name="radix_rank",
            output_name="exclusive_digit_prefix",
            resolve_type=_resolve_type,
        )
    request = _make_rank_request(
        group=group,
        launch=launch,
        input_type=key_type,
        items_per_thread=len(key_values),
        operand_kind=operand_kind,
        begin_bit=begin_bit,
        end_bit=resolved_end,
        descending=descending,
        prefix_items=prefix_items,
        source=source,
    )
    validate_operand_domains(
        request.plan.resolved_group,
        {"keys": keys},
        scope=_ROOT_SCOPE,
        primitive_name="radix_rank",
    )
    _provider_state.register_request(request)
    ranks = _cute.make_rmem_tensor(len(key_values), Int32)
    outputs = {"ranks": ranks.iterator.llvm_ptr}
    prefix_result = None
    if prefix_items is not None:
        prefix_result = _cute.make_rmem_tensor(prefix_items, Int32)
        outputs["exclusive_digit_prefix"] = prefix_result.iterator.llvm_ptr
    arguments = request.bind_ffi_arguments({"keys": key_values}, outputs)
    ffi(
        name=request.symbol_name,
        params_types=list(request.ffi_param_types),
        return_type=None,
    )(*arguments)
    if prefix_result is not None:
        assert exclusive_digit_prefix is not None
        for index in range(prefix_items):
            exclusive_digit_prefix[index] = prefix_result[index]
        attach_thread_data_metadata(
            exclusive_digit_prefix,
            _result_metadata(request, 1),
        )
    return _pack_result(
        ranks,
        items_per_thread=len(key_values),
        input_thread_data=key_data,
        result_type=Int32,
        metadata=_result_metadata(request, 0),
    )


__all__ = [
    "CutlassCoreArtifact",
    "provider_radix_rank",
    "provider_radix_sort_keys",
    "provider_radix_sort_pairs",
]
