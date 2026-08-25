# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plan-driven public-CUB provider for CUTLASS BlockDiscontinuity."""

from __future__ import annotations

from typing import Any

from cutlass import cute as _cute
from cutlass.base_dsl.typing import Int32
from cutlass.cute.ffi import ffi

from cuda.coop._core import (
    AlgorithmSpec,
    CxxOperator,
    Dependency,
    GroupDiscontinuitySemantics,
    GroupLoweringPlan,
    LaunchFacts,
    make_group_primitive_call,
    plan_group_primitive,
)
from cuda.coop._core.block import (
    BlockDiscontinuityMode,
    make_block_discontinuity_semantics,
)

from .._compiler import _state as _provider_state
from .._compiler import _storage as _provider_storage
from .._compiler import _types as _provider_types
from .._compiler._call_context import get_active_single_phase_context
from .._compiler._types import ALL_PROVIDER_TYPES, TYPE_SPECS
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
    register_cutlass_core_renderer,
    with_caller_owned_core_temp_storage,
)
from ._symbols import block_dim_token as _block_dim_token

_ROOT_SCOPE = "cuda.coop.cutlass"
_REQUEST_KIND = "cuda_coop_cutlass_cub_block_discontinuity"
_HEADER = "cub/block/block_discontinuity.cuh"

register_cutlass_core_renderer(_REQUEST_KIND, includes=(_HEADER,))

_MODE_ALIASES = {
    "head": BlockDiscontinuityMode.HEADS,
    "heads": BlockDiscontinuityMode.HEADS,
    "tail": BlockDiscontinuityMode.TAILS,
    "tails": BlockDiscontinuityMode.TAILS,
    "both": BlockDiscontinuityMode.HEADS_AND_TAILS,
    "head_tail": BlockDiscontinuityMode.HEADS_AND_TAILS,
    "heads_and_tails": BlockDiscontinuityMode.HEADS_AND_TAILS,
    "headsandtails": BlockDiscontinuityMode.HEADS_AND_TAILS,
}

_resolve_type = _provider_state.make_provider_type_resolver(
    scope=_ROOT_SCOPE,
    root_scope=_ROOT_SCOPE,
    namespace="thread_group",
)


def _normalize_mode(mode: Any) -> BlockDiscontinuityMode:
    try:
        return BlockDiscontinuityMode(mode)
    except (TypeError, ValueError):
        token = getattr(mode, "name", mode)
        token = str(token).split(".")[-1].replace("-", "_").lower()
        try:
            return _MODE_ALIASES[token]
        except KeyError as exc:
            raise ValueError(
                f"{_ROOT_SCOPE}.discontinuity mode must be 'heads', 'tails', "
                "or 'heads_and_tails'"
            ) from exc


def _make_group_discontinuity_plan(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    dtype: Any,
    flag_dtype: Any,
    items_per_thread: int,
    mode: Any,
    tile_predecessor_item: Any = None,
    tile_successor_item: Any = None,
    source: str = "cutlass_root",
) -> GroupLoweringPlan:
    """Build the canonical shared-core discontinuity plan."""

    primitive = make_block_discontinuity_semantics(
        dtype=dtype,
        flag_dtype=flag_dtype,
        items_per_thread=items_per_thread,
        mode=_normalize_mode(mode),
        flag_operator=CxxOperator(
            "::cuda::std::not_equal_to<T>",
            Dependency("T"),
            name="flag_op",
        ),
        tile_predecessor_item=tile_predecessor_item,
        tile_successor_item=tile_successor_item,
    )
    call = make_group_primitive_call(
        group,
        GroupDiscontinuitySemantics(primitive),
        source=source,
    )
    return plan_group_primitive(call, launch)


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
        f"{primitive.mode.value}_{TYPE_SPECS[value_type].token}"
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
    plan = _make_group_discontinuity_plan(
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


def _temp_storage_for_discontinuity(
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
            f"{_ROOT_SCOPE}.discontinuity received two TempStorage objects"
        )
    temp_storage = (
        explicit_temp_storage if explicit_temp_storage is not None else context_storage
    )
    if temp_storage is None:
        return None

    from .._temp_storage import TempStorage

    if not isinstance(temp_storage, TempStorage):
        raise TypeError(
            f"{_ROOT_SCOPE}.discontinuity temp_storage must be "
            f"{_ROOT_SCOPE}.TempStorage"
        )
    if group.kind != "block":
        raise ValueError(
            f"{_ROOT_SCOPE}.discontinuity TempStorage is supported only for "
            "block groups"
        )
    if not temp_storage.is_deferred and temp_storage.sharing == "exclusive":
        raise ValueError(
            f"{_ROOT_SCOPE}.discontinuity fixed-capacity TempStorage does not "
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
            primitive_name="discontinuity",
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
        value_type, values = _provider_types.resolve_thread_data_value_type(
            value,
            allowed=ALL_PROVIDER_TYPES,
            feature="discontinuity",
            scope=_ROOT_SCOPE,
            resolve_type=_resolve_type,
        )
        items_per_thread = value.items_per_thread
        values = tuple(values)
    else:
        value_type = _resolve_type(
            value,
            allowed=ALL_PROVIDER_TYPES,
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
    external_temp_storage = _temp_storage_for_discontinuity(
        group=group,
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
        external_scratch=external_temp_storage is not None,
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
        _provider_state.snapshot_active_session_state()
        if external_temp_storage is not None
        else None
    )
    try:
        _provider_state.register_request(request)
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
            _provider_state.restore_active_session_state(session_snapshot)
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
        return _provider_state.remember_scalar_result_type(
            result_values[0],
            Int32,
            scope=_ROOT_SCOPE,
            compile_options_getter=lambda: (
                _provider_state._get_cute_dsl().compile_options
            ),
            group_metadata=result_metadata,
        )

    results = tuple(make_result(name) for name in output_names)
    return results[0] if len(results) == 1 else results


__all__ = ["CutlassCoreArtifact", "provider_discontinuity"]
