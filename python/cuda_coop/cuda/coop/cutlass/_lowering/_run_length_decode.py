# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plan-driven public-CUB provider for CUTLASS BlockRunLengthDecode."""

from __future__ import annotations

import dataclasses
from numbers import Integral
from typing import Any

import numpy as np
from cutlass import cute as _cute
from cutlass._mlir.dialects import llvm
from cutlass.base_dsl.typing import Int32, Int64, Uint8, Uint32, Uint64
from cutlass.cute.ffi import ffi

from cuda.coop._core import (
    AlgorithmSpec,
    GroupLoweringPlan,
    GroupRunLengthDecodeSemantics,
    LaunchFacts,
    make_group_primitive_call,
    plan_group_primitive,
)
from cuda.coop._core.block import (
    make_block_run_length_decode_semantics,
    normalize_positive_int,
)

from .._compiler import _rendering as _provider_rendering
from .._compiler import _state as _provider_state
from .._compiler import _types as _provider_types
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
    _render_template_argument,
    _render_type_definitions,
)
from ._symbols import block_dim_token as _block_dim_token

_ROOT_SCOPE = "cuda.coop.cutlass"
_REQUEST_KIND = "cuda_coop_cutlass_cub_block_run_length_decode"
_HEADER = "cub/block/block_run_length_decode.cuh"
_TYPE_TRAITS_HEADER = "cuda/std/type_traits"
_VALUE_TYPES = frozenset({Uint8, Int32, Uint32, Int64, Uint64})
_LENGTH_TYPES = frozenset({Int32, Uint32, Int64, Uint64})


def _make_group_run_length_decode_plan(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    item_dtype: Any,
    run_length_dtype: Any,
    runs_per_thread: int,
    decoded_items_per_thread: int,
    with_relative_offsets: bool,
    source: str = "cutlass_root",
) -> GroupLoweringPlan:
    """Build the canonical fused public-CUB decode plan."""

    primitive = make_block_run_length_decode_semantics(
        item_dtype=item_dtype,
        run_length_dtype=run_length_dtype,
        decoded_offset_dtype=run_length_dtype,
        total_decoded_size_dtype=run_length_dtype,
        relative_offset_dtype=(run_length_dtype if with_relative_offsets else None),
        runs_per_thread=runs_per_thread,
        decoded_items_per_thread=decoded_items_per_thread,
        with_relative_offsets=with_relative_offsets,
        with_decoded_window_offset=True,
        returns_total_decoded_size=True,
    )
    return plan_group_primitive(
        make_group_primitive_call(
            group,
            GroupRunLengthDecodeSemantics(primitive),
            source=source,
        ),
        launch,
    )


_ORDINARY_RUN_LENGTH_DTYPES = {
    int: Int32,
    np.uint8: Uint8,
    np.int32: Int32,
    np.uint32: Uint32,
    np.int64: Int64,
    np.uint64: Uint64,
}

_resolve_type = _provider_state.make_provider_type_resolver(
    scope=_ROOT_SCOPE,
    root_scope=_ROOT_SCOPE,
    namespace="thread_group",
)


def _resolve_run_length_type(
    value: Any,
    *,
    allowed: frozenset[type],
    feature: str,
) -> type:
    if isinstance(value, np.dtype):
        value = value.type
    candidate = value if isinstance(value, type) else type(value)
    value = _ORDINARY_RUN_LENGTH_DTYPES.get(candidate, value)
    return _resolve_type(value, allowed=allowed, feature=feature)


@dataclasses.dataclass(frozen=True, eq=False)
class CutlassRunLengthDecodeArtifact:
    """One fused public-CUB decode plus its thread-local OOB projection."""

    plan: Any
    specialization: AlgorithmSpec
    core_artifact: CutlassCoreArtifact
    value_type: type
    length_type: type
    symbol_name: str
    kind: str = _REQUEST_KIND

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        assert self.plan.artifact_key is not None
        return self.plan.artifact_key, "oob_postmask"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CutlassRunLengthDecodeArtifact):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)

    @property
    def operation(self) -> GroupRunLengthDecodeSemantics:
        operation = self.plan.call.operation
        assert isinstance(operation, GroupRunLengthDecodeSemantics)
        return operation

    @property
    def runs_per_thread(self) -> int:
        return self.operation.primitive.runs_per_thread

    @property
    def decoded_items_per_thread(self) -> int:
        return self.operation.primitive.decoded_items_per_thread

    @property
    def has_relative_offsets(self) -> bool:
        return self.operation.primitive.has_relative_offsets

    @property
    def ffi_param_types(self) -> tuple[Any, ...]:
        pointer = llvm.PointerType.get(0)
        return (
            *((self.value_type,) * self.runs_per_thread),
            *((self.length_type,) * self.runs_per_thread),
            self.length_type,
            pointer,
            *((pointer,) if self.has_relative_offsets else ()),
            pointer,
        )

    def bind_ffi_arguments(
        self,
        *,
        run_values: tuple[Any, ...],
        run_lengths: tuple[Any, ...],
        decoded_window_offset: Any,
        decoded_items_pointer: Any,
        relative_offsets_pointer: Any | None,
        total_decoded_size_pointer: Any,
    ) -> tuple[Any, ...]:
        if len(run_values) != self.runs_per_thread:
            raise ValueError("run value count does not match its artifact")
        if len(run_lengths) != self.runs_per_thread:
            raise ValueError("run length count does not match its artifact")
        if self.has_relative_offsets != (relative_offsets_pointer is not None):
            raise ValueError("relative-offset pointer does not match its artifact")
        return (
            *run_values,
            *run_lengths,
            decoded_window_offset,
            decoded_items_pointer,
            *(
                (relative_offsets_pointer,)
                if relative_offsets_pointer is not None
                else ()
            ),
            total_decoded_size_pointer,
        )


def _symbol_name(*, plan: Any, value_type: type, length_type: type) -> str:
    operation = plan.call.operation
    assert isinstance(operation, GroupRunLengthDecodeSemantics)
    primitive = operation.primitive
    participation = plan.participation
    if participation is None or participation.exact_block_dim is None:
        raise ValueError("BlockRunLengthDecode symbols require exact block dimensions")
    offsets = "_offsets" if primitive.has_relative_offsets else ""
    return (
        "cuda_coop_cutlass_cub_run_length_decode_"
        f"{_block_dim_token(participation.exact_block_dim)}_"
        f"v{_TYPE_SPECS[value_type].token}_l{_TYPE_SPECS[length_type].token}_"
        f"r{primitive.runs_per_thread}_x{primitive.decoded_items_per_thread}"
        f"{offsets}"
    )


def _make_request(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    value_type: type,
    length_type: type,
    runs_per_thread: int,
    decoded_items_per_thread: int,
    with_relative_offsets: bool,
    source: str,
) -> CutlassRunLengthDecodeArtifact:
    plan = _make_group_run_length_decode_plan(
        group=group,
        launch=launch,
        item_dtype=value_type,
        run_length_dtype=length_type,
        runs_per_thread=runs_per_thread,
        decoded_items_per_thread=decoded_items_per_thread,
        with_relative_offsets=with_relative_offsets,
        source=source,
    ).require_supported()
    if not isinstance(plan.implementation, AlgorithmSpec):
        raise TypeError("BlockRunLengthDecode plan requires an AlgorithmSpec")
    symbol_name = _symbol_name(
        plan=plan,
        value_type=value_type,
        length_type=length_type,
    )
    core_artifact = CutlassCoreAdapter().materialize(
        plan.implementation,
        plan=plan,
        kind=_REQUEST_KIND,
        symbol_name=symbol_name,
    )
    return CutlassRunLengthDecodeArtifact(
        plan=plan,
        specialization=plan.implementation,
        core_artifact=core_artifact,
        value_type=value_type,
        length_type=length_type,
        symbol_name=symbol_name,
    )


def render_run_length_decode_artifact(
    artifact: CutlassRunLengthDecodeArtifact,
) -> list[str]:
    """Render one fused CUB lifecycle with thread-local OOB postprocessing."""

    adapter = CutlassCoreAdapter()
    validated = adapter.materialize(
        artifact.specialization,
        plan=artifact.plan,
        kind=artifact.kind,
        symbol_name=artifact.symbol_name,
    )
    if validated.semantic_key != artifact.core_artifact.semantic_key:
        raise ValueError("BlockRunLengthDecode artifact no longer matches its plan")
    primitive = artifact.operation.primitive
    value_cpp = adapter.cpp_type(artifact.value_type)
    length_cpp = adapter.cpp_type(artifact.length_type)
    value_parameters = ", ".join(
        f"{value_cpp} run_value_{index}" for index in range(primitive.runs_per_thread)
    )
    length_parameters = ", ".join(
        f"{length_cpp} run_length_{index}" for index in range(primitive.runs_per_thread)
    )
    pointer_parameters = [f"{value_cpp}* decoded_items_result"]
    if primitive.has_relative_offsets:
        pointer_parameters.append(f"{length_cpp}* relative_offsets_result")
    pointer_parameters.append(f"{length_cpp}* total_decoded_size_result")
    signature = ", ".join(
        (
            value_parameters,
            length_parameters,
            f"{length_cpp} decoded_window_offset",
            *pointer_parameters,
        )
    )
    template_arguments = ", ".join(
        _render_template_argument(adapter, value)
        for _, value in artifact.specialization.ordered_template_arguments
    )
    run_values = ", ".join(
        f"run_value_{index}" for index in range(primitive.runs_per_thread)
    )
    run_lengths = ", ".join(
        f"run_length_{index}" for index in range(primitive.runs_per_thread)
    )
    call_arguments = [
        "run_values",
        "run_lengths",
        "total_decoded_size",
        "decoded_items",
    ]
    if primitive.has_relative_offsets:
        call_arguments.append("relative_offsets")
    call_arguments.append("decoded_window_offset")
    relative_oob_literal = "-1" if artifact.length_type in {Int32, Int64} else "~0ull"
    projection = []
    for item in range(primitive.decoded_items_per_thread):
        projection.extend(
            (
                "  unsigned long long local_target_"
                f"{item} = first_local_target + {item}ull;",
                f"  bool valid_{item} = offset_in_range &&",
                f"      local_target_{item} < decoded_remaining;",
                f"  decoded_items_result[{item}] = valid_{item}",
                f"      ? decoded_items[{item}] : static_cast<{value_cpp}>(0);",
            )
        )
        if primitive.has_relative_offsets:
            projection.extend(
                (
                    f"  relative_offsets_result[{item}] = valid_{item}",
                    f"      ? relative_offsets[{item}]",
                    f"      : static_cast<{length_cpp}>({relative_oob_literal});",
                )
            )
    type_definitions = _render_type_definitions(validated)
    linkage_safe_definitions = (
        ["}", *type_definitions, 'extern "C" {'] if type_definitions else []
    )
    relative_declaration = (
        [f"  {length_cpp} relative_offsets[{primitive.decoded_items_per_thread}]{{}};"]
        if primitive.has_relative_offsets
        else []
    )
    return [
        *linkage_safe_definitions,
        f"void {artifact.symbol_name}({signature}) {{",
        "  using implementation_type = "
        f"::cub::{artifact.specialization.struct_name}<{template_arguments}>;",
        "  __shared__ typename implementation_type::TempStorage storage;",
        f"  {value_cpp} run_values[{primitive.runs_per_thread}] = {{{run_values}}};",
        f"  {length_cpp} run_lengths[{primitive.runs_per_thread}] = {{{run_lengths}}};",
        f"  {length_cpp} total_decoded_size{{}};",
        f"  {value_cpp} decoded_items[{primitive.decoded_items_per_thread}]{{}};",
        *relative_declaration,
        "  implementation_type(storage)."
        f"{artifact.specialization.method_name}("
        f"{', '.join(call_arguments)});",
        "  cuda_coop_cutlass_block_sync();",
        "  unsigned int linear_tid = cuda_coop_cutlass_linear_tid();",
        "  unsigned long long decoded_offset =",
        "      static_cast<unsigned long long>(decoded_window_offset);",
        "  unsigned long long decoded_total =",
        "      static_cast<unsigned long long>(total_decoded_size);",
        "  bool offset_in_range = decoded_offset < decoded_total;",
        "  unsigned long long decoded_remaining = offset_in_range",
        "      ? decoded_total - decoded_offset : 0ull;",
        "  unsigned long long first_local_target =",
        f"      static_cast<unsigned long long>(linear_tid) * "
        f"{primitive.decoded_items_per_thread}ull;",
        *projection,
        "  *total_decoded_size_result = total_decoded_size;",
        "}",
    ]


_provider_rendering.register_bundle_renderer(
    _REQUEST_KIND,
    render=render_run_length_decode_artifact,
    include_lines=(
        f"#include <{_HEADER}>",
        f"#include <{_TYPE_TRAITS_HEADER}>",
    ),
    cccl_headers=(
        (f"#include <{_HEADER}>", _HEADER),
        (f"#include <{_TYPE_TRAITS_HEADER}>", _TYPE_TRAITS_HEADER),
    ),
)


def _resolve_run_inputs(
    *,
    run_values: Any,
    run_lengths: Any,
) -> tuple[type, type, tuple[Any, ...], tuple[Any, ...], int]:
    if isinstance(run_values, ThreadData) != isinstance(run_lengths, ThreadData):
        raise TypeError(
            f"{_ROOT_SCOPE}.run_length_decode requires run_values and run_lengths "
            "to both be ThreadData or both be scalar values"
        )
    if isinstance(run_values, ThreadData):
        if run_values.items_per_thread != run_lengths.items_per_thread:
            raise ValueError(
                f"{_ROOT_SCOPE}.run_length_decode requires matching "
                "ThreadData.items_per_thread for run_values and run_lengths"
            )
        value_type, value_items = _provider_types.resolve_thread_data_value_type(
            run_values,
            allowed=_VALUE_TYPES,
            feature="run_length_decode",
            scope=_ROOT_SCOPE,
            resolve_type=_resolve_run_length_type,
        )
        length_type, length_items = _provider_types.resolve_thread_data_value_type(
            run_lengths,
            allowed=_LENGTH_TYPES,
            feature="run_length_decode",
            scope=_ROOT_SCOPE,
            resolve_type=_resolve_run_length_type,
        )
        return (
            value_type,
            length_type,
            tuple(value_items),
            tuple(length_items),
            run_values.items_per_thread,
        )
    value_type = _resolve_run_length_type(
        run_values,
        allowed=_VALUE_TYPES,
        feature="run_length_decode",
    )
    length_type = _resolve_run_length_type(
        run_lengths,
        allowed=_LENGTH_TYPES,
        feature="run_length_decode",
    )
    return value_type, length_type, (run_values,), (run_lengths,), 1


def _validate_output(
    *,
    name: str,
    output: Any,
    items_per_thread: int,
    dtype: type,
) -> ThreadData | None:
    return _provider_types.validate_thread_data_output(
        output=output,
        expected_items_per_thread=items_per_thread,
        resolved_dtype=dtype,
        scope=_ROOT_SCOPE,
        primitive_name="run_length_decode",
        output_name=name,
        resolve_type=_resolve_run_length_type,
    )


def _as_decoded_window_offset(value: Any, *, length_type: type) -> Any:
    """Convert one validated offset to the artifact's resolved offset type."""

    if not isinstance(value, Integral):
        offset_type = _resolve_run_length_type(
            value,
            allowed=_LENGTH_TYPES,
            feature="run_length_decode",
        )
        if offset_type is not length_type:
            raise TypeError(
                f"{_ROOT_SCOPE}.run_length_decode decoded_window_offset dtype "
                "must match the run-length dtype"
            )
    literal = value if isinstance(value, Integral) else getattr(value, "value", None)
    if isinstance(literal, Integral) and not isinstance(literal, bool):
        width = _TYPE_SPECS[length_type].width_bits
        value_bits = width - 1 if length_type in {Int32, Int64} else width
        if int(literal) >= 1 << value_bits:
            raise ValueError(
                f"{_ROOT_SCOPE}.run_length_decode decoded_window_offset does "
                f"not fit {length_type.__name__}"
            )
    if isinstance(value, length_type):
        return value
    try:
        return length_type(value)
    except Exception as exc:
        raise TypeError(
            f"{_ROOT_SCOPE}.run_length_decode decoded_window_offset must be "
            f"convertible to {length_type.__name__}"
        ) from exc


def provider_run_length_decode(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    run_values: Any,
    run_lengths: Any,
    decoded_items_per_thread: int,
    decoded_window_offset: Any = 0,
    relative_offsets: Any = None,
    total_decoded_size: Any = None,
    decoded_offset_dtype: Any = None,
    source: str = "cutlass_group_run_length_decode_provider",
) -> ThreadData:
    """Materialize one fused public-CUB BlockRunLengthDecode lifecycle."""

    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_ROOT_SCOPE}.run_length_decode group must be a ThreadGroup")
    if group.kind != "block":
        raise NotImplementedError(
            f"{_ROOT_SCOPE}.run_length_decode requires a block group"
        )
    try:
        decoded_items_per_thread = normalize_positive_int(
            "decoded_items_per_thread",
            decoded_items_per_thread,
        )
    except ValueError as exc:
        raise ValueError(f"{_ROOT_SCOPE}.run_length_decode: {exc}") from exc
    value_type, length_type, value_items, length_items, runs_per_thread = (
        _resolve_run_inputs(
            run_values=run_values,
            run_lengths=run_lengths,
        )
    )
    if decoded_offset_dtype is not None:
        requested_offset_type = _resolve_run_length_type(
            decoded_offset_dtype,
            allowed=_LENGTH_TYPES,
            feature="run_length_decode",
        )
        if requested_offset_type is not length_type:
            raise TypeError(
                f"{_ROOT_SCOPE}.run_length_decode decoded_offset_dtype must "
                "match the run-length dtype"
            )
    relative_offsets_td = _validate_output(
        name="relative_offsets",
        output=relative_offsets,
        items_per_thread=decoded_items_per_thread,
        dtype=length_type,
    )
    total_decoded_size_td = _validate_output(
        name="total_decoded_size",
        output=total_decoded_size,
        items_per_thread=1,
        dtype=length_type,
    )
    request = _make_request(
        group=group,
        launch=launch,
        value_type=value_type,
        length_type=length_type,
        runs_per_thread=runs_per_thread,
        decoded_items_per_thread=decoded_items_per_thread,
        with_relative_offsets=relative_offsets_td is not None,
        source=source,
    )
    validate_operand_domains(
        request.plan.resolved_group,
        {"run_values": run_values, "run_lengths": run_lengths},
        scope=_ROOT_SCOPE,
        primitive_name="run_length_decode",
    )
    _provider_state.register_request(request)

    decoded_tensor = _cute.make_rmem_tensor(decoded_items_per_thread, value_type)
    relative_tensor = (
        _cute.make_rmem_tensor(decoded_items_per_thread, length_type)
        if relative_offsets_td is not None
        else None
    )
    total_tensor = _cute.make_rmem_tensor(1, length_type)
    arguments = request.bind_ffi_arguments(
        run_values=value_items,
        run_lengths=length_items,
        decoded_window_offset=_as_decoded_window_offset(
            decoded_window_offset,
            length_type=length_type,
        ),
        decoded_items_pointer=decoded_tensor.iterator.llvm_ptr,
        relative_offsets_pointer=(
            relative_tensor.iterator.llvm_ptr if relative_tensor is not None else None
        ),
        total_decoded_size_pointer=total_tensor.iterator.llvm_ptr,
    )
    ffi(
        name=request.symbol_name,
        params_types=list(request.ffi_param_types),
        return_type=None,
    )(*arguments)

    assert request.plan.result is not None
    decoded_contract = request.plan.result.values[0]
    decoded_metadata = metadata_for_group(
        request.plan.resolved_group,
        visibility=decoded_contract.visibility,
    )
    decoded = attach_thread_data_metadata(
        ThreadData.from_values(
            *(decoded_tensor[index] for index in range(decoded_items_per_thread)),
            dtype=value_type,
        ),
        decoded_metadata,
    )
    if relative_offsets_td is not None:
        assert relative_tensor is not None
        relative_contract = next(
            value
            for value in request.plan.result.values
            if value.name == "relative_offsets"
        )
        for index in range(decoded_items_per_thread):
            relative_offsets_td[index] = relative_tensor[index]
        attach_thread_data_metadata(
            relative_offsets_td,
            metadata_for_group(
                request.plan.resolved_group,
                visibility=relative_contract.visibility,
            ),
        )
    if total_decoded_size_td is not None:
        total_contract = next(
            value
            for value in request.plan.result.values
            if value.name == "total_decoded_size"
        )
        total_decoded_size_td[0] = total_tensor[0]
        attach_thread_data_metadata(
            total_decoded_size_td,
            metadata_for_group(
                request.plan.resolved_group,
                visibility=total_contract.visibility,
            ),
        )
    return decoded


__all__ = ["CutlassRunLengthDecodeArtifact", "provider_run_length_decode"]
