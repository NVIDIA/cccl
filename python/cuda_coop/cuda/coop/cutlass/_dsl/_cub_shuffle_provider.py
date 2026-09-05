# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plan-driven public-CUB provider for representable BlockShuffle routes."""

from __future__ import annotations

from numbers import Integral
from typing import Any

from cutlass import cute as _cute
from cutlass.base_dsl.typing import Int32
from cutlass.cute.ffi import ffi

from cuda.coop._core import AlgorithmSpec, GroupShuffleSemantics, LaunchFacts
from cuda.coop._core.block import BlockShuffleMode

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
)
from ._provider import ALL_PROVIDER_TYPES as _ALL_PROVIDER_TYPES
from ._provider import TYPE_SPECS as _TYPE_SPECS
from ._symbols import block_dim_token as _block_dim_token
from ._thread_data import ThreadData
from ._thread_group import ThreadGroup

_ROOT_SCOPE = __name__.split("._dsl.", 1)[0]
_REQUEST_KIND = "cuda_coop_cutlass_cub_block_shuffle"
_HEADER = "cub/block/block_shuffle.cuh"

register_cutlass_core_renderer(_REQUEST_KIND, includes=(_HEADER,))

_resolve_type = _provider_support.make_provider_type_resolver(
    scope=_ROOT_SCOPE,
    root_scope=_ROOT_SCOPE,
    namespace="thread_group",
)


def _symbol_name(*, plan: Any, value_type: type) -> str:
    operation = plan.call.operation
    assert isinstance(operation, GroupShuffleSemantics)
    primitive = operation.primitive
    participation = plan.participation
    if participation is None or participation.exact_block_dim is None:
        raise ValueError("BlockShuffle symbols require exact block dimensions")
    name = (
        f"cuda_coop_cutlass_shuffle_"
        f"{_block_dim_token(participation.exact_block_dim)}_"
        f"{primitive.mode.value}_{_TYPE_SPECS[value_type].token}"
    )
    if primitive.items_per_thread is not None:
        name += f"_x{primitive.items_per_thread}"
    if primitive.block_prefix:
        name += "_prefix"
    if primitive.block_suffix:
        name += "_suffix"
    return name


def _make_request(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    value_type: type,
    items_per_thread: int | None,
    mode: BlockShuffleMode,
    block_prefix: bool,
    block_suffix: bool,
    source: str,
) -> CutlassCoreArtifact:
    from .. import _group_shuffle as _group_frontend

    plan = _group_frontend._make_group_shuffle_plan(
        group=group,
        launch=launch,
        dtype=value_type,
        items_per_thread=items_per_thread,
        mode=mode,
        block_prefix=block_prefix,
        block_suffix=block_suffix,
        source=source,
    ).require_supported()
    if not isinstance(plan.implementation, AlgorithmSpec):
        raise TypeError("BlockShuffle plan requires an AlgorithmSpec")
    runtime_int_ranges = ()
    if items_per_thread is None:
        block_threads = plan.resolved_group.static_size
        assert block_threads is not None
        if mode is BlockShuffleMode.ROTATE:
            runtime_int_ranges = (
                CutlassRuntimeIntRange(
                    "distance",
                    0,
                    block_threads - 1,
                    modulus=block_threads,
                ),
            )
        elif mode is BlockShuffleMode.OFFSET:
            runtime_int_ranges = (
                CutlassRuntimeIntRange(
                    "distance",
                    -block_threads,
                    block_threads,
                    clamp=True,
                ),
            )
    return CutlassCoreAdapter().materialize(
        plan.implementation,
        plan=plan,
        kind=_REQUEST_KIND,
        symbol_name=_symbol_name(plan=plan, value_type=value_type),
        output_initializers=(
            (("output_items", "input_items"),)
            if items_per_thread is not None
            else (("output_item", "input_item"),)
        ),
        runtime_int_ranges=runtime_int_ranges,
    )


def _validate_edge_output(
    *,
    output: Any,
    name: str,
    value_type: type,
) -> ThreadData | None:
    return _provider_support.validate_thread_data_output(
        output=output,
        expected_items_per_thread=1,
        resolved_dtype=value_type,
        scope=_ROOT_SCOPE,
        primitive_name="shuffle",
        output_name=name,
        resolve_type=_resolve_type,
        type_label=f"a {_ROOT_SCOPE}.ThreadData output",
    )


def _normalize_static_distance(
    distance: Any,
    runtime_range: CutlassRuntimeIntRange,
) -> Any:
    """Apply the generated-wrapper range policy before narrowing Python ints."""

    if not isinstance(distance, Integral):
        return distance
    distance = int(distance)
    if runtime_range.modulus is not None:
        return distance % runtime_range.modulus
    if runtime_range.clamp:
        return min(max(distance, runtime_range.minimum), runtime_range.maximum)
    return distance


def provider_shuffle(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    value: Any,
    mode: BlockShuffleMode,
    distance: int,
    block_prefix: Any = None,
    block_suffix: Any = None,
    source: str = "cutlass_group_shuffle_provider",
) -> Any:
    """Invoke exactly one representable public CUB BlockShuffle collective."""

    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_ROOT_SCOPE}.shuffle group must be a ThreadGroup")
    if group.kind != "block":
        raise NotImplementedError(f"{_ROOT_SCOPE}.shuffle requires a block group")
    if not isinstance(mode, BlockShuffleMode):
        mode = BlockShuffleMode(mode)

    from .. import _group_shuffle as _group_frontend

    distance = _group_frontend._normalize_shuffle_route(
        value,
        mode=mode,
        distance=distance,
        block_prefix=block_prefix,
        block_suffix=block_suffix,
    )
    is_thread_data = isinstance(value, ThreadData)
    if is_thread_data:
        value_type, values = _provider_support.resolve_thread_data_value_type(
            value,
            allowed=_ALL_PROVIDER_TYPES,
            feature="shuffle",
            scope=_ROOT_SCOPE,
            resolve_type=_resolve_type,
        )
        items_per_thread = value.items_per_thread
        values = tuple(values)
    else:
        value_type = _resolve_type(
            value,
            allowed=_ALL_PROVIDER_TYPES,
            feature="shuffle",
        )
        items_per_thread = None
        values = (value,)
        if not isinstance(distance, Integral):
            _resolve_type(
                distance,
                allowed=(Int32,),
                feature="shuffle distance",
            )

    prefix_output = _validate_edge_output(
        output=block_prefix,
        name="block_prefix",
        value_type=value_type,
    )
    suffix_output = _validate_edge_output(
        output=block_suffix,
        name="block_suffix",
        value_type=value_type,
    )
    request = _make_request(
        group=group,
        launch=launch,
        value_type=value_type,
        items_per_thread=items_per_thread,
        mode=mode,
        block_prefix=prefix_output is not None,
        block_suffix=suffix_output is not None,
        source=source,
    )
    validate_operand_domains(
        request.plan.resolved_group,
        {"value": value},
        scope=_ROOT_SCOPE,
        primitive_name="shuffle",
    )
    _provider_support.register_request(request)

    result_items = 1 if items_per_thread is None else items_per_thread
    result_tensor = _cute.make_rmem_tensor(result_items, value_type)
    output_tensors = {
        ("output_item" if items_per_thread is None else "output_items"): (
            result_tensor.iterator.llvm_ptr
        )
    }
    boundary_tensors = {}
    for name, output in (
        ("block_prefix", prefix_output),
        ("block_suffix", suffix_output),
    ):
        if output is not None:
            tensor = _cute.make_rmem_tensor(1, value_type)
            boundary_tensors[name] = tensor
            output_tensors[name] = tensor.iterator.llvm_ptr

    runtime_values: dict[str, Any]
    if items_per_thread is None:
        (distance_range,) = request.runtime_int_ranges
        distance = _normalize_static_distance(distance, distance_range)
        runtime_values = {
            "input_item": values[0],
            "distance": _provider_support.as_int32(distance),
        }
    else:
        runtime_values = {"input_items": values}
    arguments = request.bind_ffi_arguments(runtime_values, output_tensors)
    ffi(
        name=request.symbol_name,
        params_types=list(request.ffi_param_types),
        return_type=None,
    )(*arguments)

    assert request.plan.result is not None
    result_contracts = {result.name: result for result in request.plan.result.values}
    for name, output in (
        ("block_prefix", prefix_output),
        ("block_suffix", suffix_output),
    ):
        if output is None:
            continue
        output[0] = boundary_tensors[name][0]
        attach_thread_data_metadata(
            output,
            metadata_for_group(
                request.plan.resolved_group,
                visibility=result_contracts[name].visibility,
            ),
        )

    result_values = tuple(result_tensor[index] for index in range(result_items))
    result_metadata = metadata_for_group(
        request.plan.resolved_group,
        visibility=request.plan.result.primary.visibility,
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


__all__ = ["CutlassCoreArtifact", "provider_shuffle"]
