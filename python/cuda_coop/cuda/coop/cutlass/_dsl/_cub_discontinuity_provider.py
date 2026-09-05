# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plan-driven public-CUB provider for CUTLASS BlockDiscontinuity."""

from __future__ import annotations

from typing import Any

from cutlass import cute as _cute
from cutlass.base_dsl.typing import Int32
from cutlass.cute.ffi import ffi

from cuda.coop._core import AlgorithmSpec, GroupDiscontinuitySemantics, LaunchFacts
from cuda.coop._core.block import BlockDiscontinuityMode

from .._value_metadata import (
    attach_thread_data_metadata,
    metadata_for_group,
    validate_operand_domains,
)
from . import _provider as _provider_support
from ._core_adapter import (
    CutlassCoreAdapter,
    CutlassCoreArtifact,
    register_cutlass_core_renderer,
    resolve_deferred_core_temp_storage,
    with_caller_owned_core_temp_storage,
)
from ._provider import ALL_PROVIDER_TYPES as _ALL_PROVIDER_TYPES
from ._provider import TYPE_SPECS as _TYPE_SPECS
from ._symbols import block_dim_token as _block_dim_token
from ._thread_data import ThreadData
from ._thread_group import ThreadGroup

_ROOT_SCOPE = __name__.split("._dsl.", 1)[0]
_REQUEST_KIND = "cuda_coop_cutlass_cub_block_discontinuity"
_HEADER = "cub/block/block_discontinuity.cuh"

register_cutlass_core_renderer(_REQUEST_KIND, includes=(_HEADER,))

_resolve_type = _provider_support.make_provider_type_resolver(
    scope=_ROOT_SCOPE,
    root_scope=_ROOT_SCOPE,
    namespace="thread_group",
)


def _symbol_name(*, plan: Any, value_type: type) -> str:
    operation = plan.call.operation
    assert isinstance(operation, GroupDiscontinuitySemantics)
    primitive = operation.primitive
    participation = plan.participation
    if participation is None or participation.exact_block_dim is None:
        raise ValueError("BlockDiscontinuity symbols require exact block dimensions")
    name = (
        "cuda_coop_cutlass_discontinuity_"
        f"{_block_dim_token(participation.exact_block_dim)}_"
        f"{primitive.mode.value}_{_TYPE_SPECS[value_type].token}"
    )
    if primitive.items_per_thread > 1:
        name += f"_x{primitive.items_per_thread}"
    if primitive.has_tile_predecessor:
        name += "_predecessor"
    if primitive.has_tile_successor:
        name += "_successor"
    return name


def _make_request(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    value_type: type,
    items_per_thread: int,
    mode: BlockDiscontinuityMode,
    tile_predecessor_item: Any,
    tile_successor_item: Any,
    source: str,
    external_scratch: bool = False,
) -> CutlassCoreArtifact:
    from .. import _group_discontinuity as _group_frontend

    plan = _group_frontend._make_group_discontinuity_plan(
        group=group,
        launch=launch,
        dtype=value_type,
        flag_dtype=Int32,
        items_per_thread=items_per_thread,
        mode=mode,
        tile_predecessor_item=tile_predecessor_item,
        tile_successor_item=tile_successor_item,
        source=source,
    ).require_supported()
    if not isinstance(plan.implementation, AlgorithmSpec):
        raise TypeError("BlockDiscontinuity plan requires an AlgorithmSpec")
    if external_scratch:
        plan = with_caller_owned_core_temp_storage(plan)
    return CutlassCoreAdapter().materialize(
        plan.implementation,
        plan=plan,
        kind=_REQUEST_KIND,
        symbol_name=_symbol_name(plan=plan, value_type=value_type),
        external_scratch=external_scratch,
    )


def _coerce_boundary(value: Any, value_type: type, *, name: str) -> Any:
    if value is None or isinstance(value, value_type):
        return value
    try:
        return value_type(value)
    except Exception as exc:
        raise TypeError(
            f"{_ROOT_SCOPE}.discontinuity {name} cannot be converted to "
            f"{value_type.__name__}"
        ) from exc


def provider_discontinuity(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    value: Any,
    mode: BlockDiscontinuityMode,
    tile_predecessor_item: Any = None,
    tile_successor_item: Any = None,
    source: str = "cutlass_group_discontinuity_provider",
    temp_storage: Any = None,
) -> Any:
    """Materialize one whole-array CUB discontinuity call."""

    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_ROOT_SCOPE}.discontinuity group must be a ThreadGroup")
    if group.kind != "block":
        raise NotImplementedError(f"{_ROOT_SCOPE}.discontinuity requires a block group")
    if not isinstance(mode, BlockDiscontinuityMode):
        mode = BlockDiscontinuityMode(mode)

    is_thread_data = isinstance(value, ThreadData)
    if is_thread_data:
        value_type, values = _provider_support.resolve_thread_data_value_type(
            value,
            allowed=_ALL_PROVIDER_TYPES,
            feature="discontinuity",
            scope=_ROOT_SCOPE,
            resolve_type=_resolve_type,
        )
        items_per_thread = value.items_per_thread
        values = tuple(values)
    else:
        value_type = _resolve_type(
            value,
            allowed=_ALL_PROVIDER_TYPES,
            feature="discontinuity",
        )
        items_per_thread = 1
        values = (value,)

    tile_predecessor_item = _coerce_boundary(
        tile_predecessor_item,
        value_type,
        name="tile_predecessor_item",
    )
    tile_successor_item = _coerce_boundary(
        tile_successor_item,
        value_type,
        name="tile_successor_item",
    )
    deferred_temp_storage = resolve_deferred_core_temp_storage(
        group=group,
        primitive_name="discontinuity",
        source=source,
        explicit_temp_storage=temp_storage,
    )
    request = _make_request(
        group=group,
        launch=launch,
        value_type=value_type,
        items_per_thread=items_per_thread,
        mode=mode,
        tile_predecessor_item=tile_predecessor_item,
        tile_successor_item=tile_successor_item,
        source=source,
        external_scratch=deferred_temp_storage is not None,
    )
    validate_operand_domains(
        request.plan.resolved_group,
        {
            "value": value,
            **(
                {"tile_predecessor_item": tile_predecessor_item}
                if tile_predecessor_item is not None
                else {}
            ),
            **(
                {"tile_successor_item": tile_successor_item}
                if tile_successor_item is not None
                else {}
            ),
        },
        scope=_ROOT_SCOPE,
        primitive_name="discontinuity",
    )
    output_names = []
    if mode.has_heads:
        output_names.append("head_flags")
    if mode.has_tails:
        output_names.append("tail_flags")
    output_tensors = {
        name: _cute.make_rmem_tensor(items_per_thread, Int32) for name in output_names
    }
    runtime_values = {"input_items": values}
    if tile_predecessor_item is not None:
        runtime_values["tile_predecessor_item"] = tile_predecessor_item
    if tile_successor_item is not None:
        runtime_values["tile_successor_item"] = tile_successor_item
    session_snapshot = (
        _provider_support.snapshot_active_session_state()
        if deferred_temp_storage is not None
        else None
    )
    try:
        _provider_support.register_request(request)
        scratch_values = (
            _provider_support.register_deferred_temp_storage_event(
                deferred_temp_storage,
                primitive_name="discontinuity",
                requirement_key=request.scratch_requirement_key,
            )
            if deferred_temp_storage is not None
            else ()
        )
        arguments = request.bind_ffi_arguments(
            runtime_values,
            {
                name: output_tensor.iterator.llvm_ptr
                for name, output_tensor in output_tensors.items()
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
            _provider_support.restore_active_session_state(session_snapshot)
        raise

    assert request.plan.result is not None
    result_contracts = {result.name: result for result in request.plan.result.values}

    def make_result(name: str) -> Any:
        output_tensor = output_tensors[name]
        result_values = tuple(output_tensor[index] for index in range(items_per_thread))
        result_metadata = metadata_for_group(
            request.plan.resolved_group,
            visibility=result_contracts[name].visibility,
        )
        if is_thread_data:
            return attach_thread_data_metadata(
                ThreadData.from_values(*result_values, dtype=Int32),
                result_metadata,
            )
        return _provider_support.remember_scalar_result_type(
            result_values[0],
            Int32,
            scope=_ROOT_SCOPE,
            compile_options_getter=lambda: (
                _provider_support._get_cute_dsl().compile_options
            ),
            group_metadata=result_metadata,
        )

    results = tuple(make_result(name) for name in output_names)
    return results[0] if len(results) == 1 else results


__all__ = ["CutlassCoreArtifact", "provider_discontinuity"]
