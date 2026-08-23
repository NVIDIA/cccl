# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plan-driven public-CUB provider for CUTLASS BlockAdjacentDifference."""

from __future__ import annotations

from numbers import Integral
from typing import Any

from cutlass import cute as _cute
from cutlass.base_dsl.typing import Int32
from cutlass.cute.ffi import ffi

from cuda.coop._core import AlgorithmSpec, GroupAdjacentDifferenceSemantics, LaunchFacts
from cuda.coop._core.block import (
    BlockAdjacentDifferenceBoundary,
    BlockAdjacentDifferenceDirection,
)

from .._value_metadata import (
    attach_thread_data_metadata,
    metadata_for_group,
    validate_operand_domains,
)
from . import _provider as _provider_support
from ._core_adapter import (
    CutlassCoreAdapter,
    CutlassCoreArtifact,
    CutlassRuntimeIntRange,
    register_cutlass_core_renderer,
    with_caller_owned_core_temp_storage,
)
from ._provider import ALL_PROVIDER_TYPES as _ALL_PROVIDER_TYPES
from ._provider import TYPE_SPECS as _TYPE_SPECS
from ._single_phase import get_active_single_phase_context
from ._symbols import block_dim_token as _block_dim_token
from ._thread_data import ThreadData
from ._thread_group import ThreadGroup

_ROOT_SCOPE = __name__.split("._dsl.", 1)[0]
_REQUEST_KIND = "cuda_coop_cutlass_cub_block_adjacent_difference"
_HEADER = "cub/block/block_adjacent_difference.cuh"

register_cutlass_core_renderer(_REQUEST_KIND, includes=(_HEADER,))

_resolve_type = _provider_support.make_provider_type_resolver(
    scope=_ROOT_SCOPE,
    root_scope=_ROOT_SCOPE,
    namespace="thread_group",
)


def _symbol_name(
    *,
    plan,
    value_type: type,
) -> str:
    operation = plan.call.operation
    assert isinstance(operation, GroupAdjacentDifferenceSemantics)
    primitive = operation.primitive
    participation = plan.participation
    if participation is None or participation.exact_block_dim is None:
        raise ValueError(
            "BlockAdjacentDifference symbols require exact block dimensions"
        )
    block_token = _block_dim_token(participation.exact_block_dim)
    direction = (
        "subtract_left"
        if primitive.direction is BlockAdjacentDifferenceDirection.LEFT
        else "subtract_right"
    )
    name = (
        f"cuda_coop_cutlass_adjacent_difference_{block_token}_{direction}_"
        f"{_TYPE_SPECS[value_type].token}"
    )
    if primitive.items_per_thread > 1:
        name += f"_x{primitive.items_per_thread}"
    if primitive.has_partial_tile:
        name += "_partial"
    if primitive.boundary is BlockAdjacentDifferenceBoundary.PREDECESSOR:
        name += "_predecessor"
    elif primitive.boundary is BlockAdjacentDifferenceBoundary.SUCCESSOR:
        name += "_successor"
    return name


def _make_request(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    value_type: type,
    items_per_thread: int,
    direction: BlockAdjacentDifferenceDirection,
    valid_items: Any,
    tile_predecessor_item: Any,
    tile_successor_item: Any,
    source: str,
    external_scratch: bool = False,
) -> CutlassCoreArtifact:
    from .. import _group_adjacent_difference as _group_frontend

    plan = _group_frontend._make_group_adjacent_difference_plan(
        group=group,
        launch=launch,
        dtype=value_type,
        items_per_thread=items_per_thread,
        direction=direction,
        valid_items=valid_items,
        tile_predecessor_item=tile_predecessor_item,
        tile_successor_item=tile_successor_item,
        source=source,
    ).require_supported()
    if not isinstance(plan.implementation, AlgorithmSpec):
        raise TypeError("BlockAdjacentDifference plan requires an AlgorithmSpec")
    if external_scratch:
        plan = with_caller_owned_core_temp_storage(plan)
    runtime_int_ranges = ()
    if valid_items is not None:
        block_threads = plan.resolved_group.static_size
        assert block_threads is not None
        runtime_int_ranges = (
            CutlassRuntimeIntRange(
                "valid_items",
                0,
                block_threads * items_per_thread,
            ),
        )
    return CutlassCoreAdapter().materialize(
        plan.implementation,
        plan=plan,
        kind=_REQUEST_KIND,
        symbol_name=_symbol_name(plan=plan, value_type=value_type),
        runtime_int_ranges=runtime_int_ranges,
        external_scratch=external_scratch,
    )


def _coerce_boundary(value: Any, value_type: type, *, name: str) -> Any:
    if value is None or isinstance(value, value_type):
        return value
    try:
        return value_type(value)
    except Exception as exc:
        raise TypeError(
            f"{_ROOT_SCOPE}.adjacent_difference {name} cannot be converted to "
            f"{value_type.__name__}"
        ) from exc


def _temp_storage_for_adjacent_difference(
    *,
    group: ThreadGroup,
    explicit_temp_storage: Any,
) -> Any | None:
    context = get_active_single_phase_context()
    context_storage = context.temp_storage if context is not None else None
    if (
        explicit_temp_storage is not None
        and context_storage is not None
        and explicit_temp_storage is not context_storage
    ):
        raise ValueError(
            f"{_ROOT_SCOPE}.adjacent_difference received two TempStorage objects"
        )
    temp_storage = (
        explicit_temp_storage if explicit_temp_storage is not None else context_storage
    )
    if temp_storage is None:
        return None

    from .._temp_storage import TempStorage

    if not isinstance(temp_storage, TempStorage):
        raise TypeError(
            f"{_ROOT_SCOPE}.adjacent_difference temp_storage must be "
            f"{_ROOT_SCOPE}.TempStorage"
        )
    if group.kind != "block":
        raise ValueError(
            f"{_ROOT_SCOPE}.adjacent_difference TempStorage is supported only "
            "for block groups"
        )
    if not temp_storage.is_deferred and temp_storage.sharing == "exclusive":
        raise ValueError(
            f"{_ROOT_SCOPE}.adjacent_difference fixed-capacity TempStorage "
            "does not support sharing='exclusive'; use sharing='shared' or "
            "deferred storage"
        )
    return temp_storage


def _external_scratch_args(
    temp_storage: Any,
    *,
    requirement_key: Any,
) -> tuple[Any, Any, Any]:
    if temp_storage.is_deferred:
        return _provider_support.register_deferred_temp_storage_event(
            temp_storage,
            primitive_name="adjacent_difference",
            requirement_key=requirement_key,
        )

    binding = _provider_support.materialize_temp_storage_binding(
        temp_storage,
        scope=_ROOT_SCOPE,
        implicit_alignment=16,
    )
    return (
        binding.smem_addr_u32,
        Int32(binding.size_in_bytes),
        Int32(1 if binding.auto_sync else 0),
    )


def provider_adjacent_difference(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    value: Any,
    direction: BlockAdjacentDifferenceDirection,
    valid_items: Any = None,
    tile_predecessor_item: Any = None,
    tile_successor_item: Any = None,
    source: str = "cutlass_group_adjacent_difference_provider",
    temp_storage: Any = None,
) -> Any:
    """Materialize one whole-register-array CUB adjacent-difference call."""

    if not isinstance(group, ThreadGroup):
        raise TypeError(
            f"{_ROOT_SCOPE}.adjacent_difference group must be a ThreadGroup"
        )
    if group.kind != "block":
        raise NotImplementedError(
            f"{_ROOT_SCOPE}.adjacent_difference requires a block group"
        )
    if not isinstance(direction, BlockAdjacentDifferenceDirection):
        direction = BlockAdjacentDifferenceDirection(direction)

    is_thread_data = isinstance(value, ThreadData)
    if is_thread_data:
        value_type, values = _provider_support.resolve_thread_data_value_type(
            value,
            allowed=_ALL_PROVIDER_TYPES,
            feature="adjacent_difference",
            scope=_ROOT_SCOPE,
            resolve_type=_resolve_type,
        )
        items_per_thread = value.items_per_thread
        values = tuple(values)
    else:
        value_type = _resolve_type(
            value,
            allowed=_ALL_PROVIDER_TYPES,
            feature="adjacent_difference",
        )
        items_per_thread = 1
        values = (value,)

    if isinstance(valid_items, bool):
        raise TypeError("valid_items must be an integer, not bool")
    valid_items_arg = (
        None
        if valid_items is None
        else _provider_support.as_valid_items_arg(
            valid_items,
            scope=_ROOT_SCOPE,
        )
    )
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
    external_temp_storage = _temp_storage_for_adjacent_difference(
        group=group,
        explicit_temp_storage=temp_storage,
    )
    request = _make_request(
        group=group,
        launch=launch,
        value_type=value_type,
        items_per_thread=items_per_thread,
        direction=direction,
        valid_items=valid_items,
        tile_predecessor_item=tile_predecessor_item,
        tile_successor_item=tile_successor_item,
        source=source,
        external_scratch=external_temp_storage is not None,
    )
    if isinstance(valid_items, Integral) and not isinstance(valid_items, bool):
        runtime_range = request.runtime_int_ranges[0]
        if not runtime_range.minimum <= int(valid_items) <= runtime_range.maximum:
            raise ValueError(
                f"{_ROOT_SCOPE}.adjacent_difference valid_items must be between "
                f"zero and the block tile size ({runtime_range.maximum})"
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
        primitive_name="adjacent_difference",
    )
    result_tensor = _cute.make_rmem_tensor(items_per_thread, value_type)
    runtime_values = {"input_items": values}
    if valid_items_arg is not None:
        runtime_values["valid_items"] = valid_items_arg
    if tile_predecessor_item is not None:
        runtime_values["tile_predecessor_item"] = tile_predecessor_item
    if tile_successor_item is not None:
        runtime_values["tile_successor_item"] = tile_successor_item
    session_snapshot = (
        _provider_support.snapshot_active_session_state()
        if external_temp_storage is not None
        else None
    )
    try:
        _provider_support.register_request(request)
        scratch_values = (
            _external_scratch_args(
                external_temp_storage,
                requirement_key=request.scratch_requirement_key,
            )
            if external_temp_storage is not None
            else ()
        )
        arguments = request.bind_ffi_arguments(
            runtime_values,
            {"output_items": result_tensor.iterator.llvm_ptr},
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
    result_values = tuple(result_tensor[index] for index in range(items_per_thread))
    assert request.plan.result is not None
    result_metadata = metadata_for_group(
        request.plan.resolved_group,
        visibility=request.plan.result.visibility,
    )
    if is_thread_data:
        return attach_thread_data_metadata(
            ThreadData.from_values(
                *result_values,
                dtype=_provider_support.thread_data_output_dtype(value, value_type),
            ),
            result_metadata,
        )
    return _provider_support.remember_scalar_result_type(
        result_values[0],
        value_type,
        scope=_ROOT_SCOPE,
        compile_options_getter=lambda: (
            _provider_support._get_cute_dsl().compile_options
        ),
        group_metadata=result_metadata,
    )


__all__ = ["CutlassCoreArtifact", "provider_adjacent_difference"]
