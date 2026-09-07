# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plan-driven public-CUB provider for CUTLASS BlockHistogram."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
from cutlass import cute as _cute
from cutlass._mlir.dialects import llvm
from cutlass.base_dsl.typing import Int32, Int64, Uint8, Uint32, Uint64
from cutlass.cute.ffi import ffi

from cuda.coop._core import AlgorithmSpec, GroupHistogramSemantics, LaunchFacts

from .._value_metadata import (
    attach_thread_data_metadata,
    metadata_for_group,
    validate_operand_domains,
)
from . import _provider as _provider_support
from ._core_adapter import CutlassCoreAdapter, CutlassCoreArtifact
from ._provider import TYPE_SPECS as _TYPE_SPECS
from ._symbols import block_dim_token as _block_dim_token
from ._thread_data import ThreadData
from ._thread_group import ThreadGroup

_ROOT_SCOPE = __name__.split("._dsl.", 1)[0]
_REQUEST_KIND = "cuda_coop_cutlass_cub_block_histogram"
_HEADER = "cub/block/block_histogram.cuh"
_HISTOGRAM_SAMPLE_TYPES = frozenset({Uint8, Int32, Uint32, Int64, Uint64})
_HISTOGRAM_COUNTER_TYPES = frozenset({Int32, Uint32, Int64, Uint64})
_ORDINARY_HISTOGRAM_DTYPES = {
    int: Int32,
    np.uint8: Uint8,
    np.int32: Int32,
    np.uint32: Uint32,
    np.int64: Int64,
    np.uint64: Uint64,
}

_resolve_type = _provider_support.make_provider_type_resolver(
    scope=_ROOT_SCOPE,
    root_scope=_ROOT_SCOPE,
    namespace="thread_group",
)


def _resolve_histogram_type(
    value: Any,
    *,
    allowed: frozenset[type],
    feature: str,
) -> type:
    if isinstance(value, np.dtype):
        value = value.type
    candidate = value if isinstance(value, type) else type(value)
    value = _ORDINARY_HISTOGRAM_DTYPES.get(candidate, value)
    return _resolve_type(value, allowed=allowed, feature=feature)


@dataclasses.dataclass(frozen=True, eq=False)
class CutlassHistogramArtifact:
    """One typed CUB histogram plus its shared-output projection contract."""

    plan: Any
    specialization: AlgorithmSpec
    core_artifact: CutlassCoreArtifact
    sample_type: type
    counter_type: type
    bins_per_thread: int
    symbol_name: str
    kind: str = _REQUEST_KIND

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        assert self.plan.artifact_key is not None
        return self.plan.artifact_key, "shared_histogram_projection"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CutlassHistogramArtifact):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)

    @property
    def items_per_thread(self) -> int:
        operation = self.plan.call.operation
        assert isinstance(operation, GroupHistogramSemantics)
        return operation.primitive.items_per_thread

    @property
    def ffi_param_types(self) -> tuple[Any, ...]:
        return (
            *((self.sample_type,) * self.items_per_thread),
            llvm.PointerType.get(0),
        )

    def bind_ffi_arguments(
        self,
        samples: tuple[Any, ...],
        result_pointer: Any,
    ) -> tuple[Any, ...]:
        if len(samples) != self.items_per_thread:
            raise ValueError("histogram sample count does not match its artifact")
        return *samples, result_pointer


def _symbol_name(
    *,
    plan: Any,
    sample_type: type,
    counter_type: type,
) -> str:
    operation = plan.call.operation
    assert isinstance(operation, GroupHistogramSemantics)
    primitive = operation.primitive
    participation = plan.participation
    if participation is None or participation.exact_block_dim is None:
        raise ValueError("BlockHistogram symbols require exact block dimensions")
    return (
        "cuda_coop_cutlass_cub_histogram_"
        f"{_block_dim_token(participation.exact_block_dim)}_"
        f"{primitive.algorithm.value}_"
        f"{_TYPE_SPECS[sample_type].token}_x{primitive.items_per_thread}_"
        f"count_{_TYPE_SPECS[counter_type].token}_"
        f"bins{primitive.bins}_out{operation.bins_per_thread}"
    )


def _make_request(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    sample_type: type,
    counter_type: type,
    items_per_thread: int,
    bins: int,
    bins_per_thread: int,
    algorithm: Any,
    source: str,
) -> CutlassHistogramArtifact:
    from .. import _group_histogram as _group_frontend

    plan = _group_frontend._make_group_histogram_plan(
        group=group,
        launch=launch,
        item_dtype=sample_type,
        counter_dtype=counter_type,
        items_per_thread=items_per_thread,
        bins=bins,
        bins_per_thread=bins_per_thread,
        algorithm=algorithm,
        source=source,
    ).require_supported()
    if not isinstance(plan.implementation, AlgorithmSpec):
        raise TypeError("BlockHistogram plan requires an AlgorithmSpec")
    symbol_name = _symbol_name(
        plan=plan,
        sample_type=sample_type,
        counter_type=counter_type,
    )
    core_artifact = CutlassCoreAdapter().materialize(
        plan.implementation,
        plan=plan,
        kind=_REQUEST_KIND,
        symbol_name=symbol_name,
    )
    return CutlassHistogramArtifact(
        plan=plan,
        specialization=plan.implementation,
        core_artifact=core_artifact,
        sample_type=sample_type,
        counter_type=counter_type,
        bins_per_thread=bins_per_thread,
        symbol_name=symbol_name,
    )


def _render_template_argument(adapter: CutlassCoreAdapter, value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return value
    return adapter.cpp_type(value)


def render_histogram_artifact(artifact: CutlassHistogramArtifact) -> list[str]:
    """Render one CUB call with block-shared counters and local projection."""

    adapter = CutlassCoreAdapter()
    validated = adapter.materialize(
        artifact.specialization,
        plan=artifact.plan,
        kind=artifact.kind,
        symbol_name=artifact.symbol_name,
    )
    if validated.semantic_key != artifact.core_artifact.semantic_key:
        raise ValueError("BlockHistogram artifact no longer matches its core plan")
    operation = artifact.plan.call.operation
    assert isinstance(operation, GroupHistogramSemantics)
    primitive = operation.primitive
    assert primitive.bins is not None
    participation = artifact.plan.participation
    assert participation is not None
    assert participation.exact_block_dim is not None
    block_threads = participation.exact_group_size
    sample_cpp = adapter.cpp_type(artifact.sample_type)
    counter_cpp = adapter.cpp_type(artifact.counter_type)
    internal_counter_cpp = (
        "unsigned long long"
        if _TYPE_SPECS[artifact.counter_type].width_bits == 64
        else "unsigned int"
    )
    sample_parameters = ", ".join(
        f"{sample_cpp} sample_{index}" for index in range(primitive.items_per_thread)
    )
    signature = f"{sample_parameters}, {counter_cpp}* histogram_result"
    samples = ", ".join(
        f"sample_{index}" for index in range(primitive.items_per_thread)
    )
    template_arguments = ", ".join(
        _render_template_argument(adapter, value)
        for _, value in artifact.specialization.ordered_template_arguments
    )
    projection: list[str] = []
    for track in range(artifact.bins_per_thread):
        bin_offset = track * block_threads
        projection.extend(
            (
                f"  unsigned int bin_{track} = linear_tid + {bin_offset}u;",
                f"  histogram_result[{track}] = bin_{track} < {primitive.bins}u",
                f"      ? static_cast<{counter_cpp}>(histogram[bin_{track}])",
                f"      : static_cast<{counter_cpp}>(0);",
            )
        )
    return [
        f"void {artifact.symbol_name}({signature}) {{",
        f"  using implementation_type = ::cub::BlockHistogram<{template_arguments}>;",
        "  __shared__ typename implementation_type::TempStorage storage;",
        f"  __shared__ {internal_counter_cpp} histogram[{primitive.bins}];",
        f"  {sample_cpp} items[{primitive.items_per_thread}] = {{{samples}}};",
        "  implementation_type(storage).Histogram(items, histogram);",
        "  cuda_coop_cutlass_block_sync();",
        "  unsigned int linear_tid = cuda_coop_cutlass_linear_tid();",
        *projection,
        "}",
    ]


_provider_support.register_bundle_renderer(
    _REQUEST_KIND,
    render=render_histogram_artifact,
    include_lines=(f"#include <{_HEADER}>",),
    cccl_headers=((f"#include <{_HEADER}>", _HEADER),),
)


def provider_histogram(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    samples: Any,
    bins: int,
    bins_per_thread: int = 1,
    counter_dtype: Any = None,
    algorithm: Any = "atomic",
    source: str = "cutlass_group_histogram_provider",
) -> ThreadData:
    """Materialize one public-CUB BlockHistogram call."""

    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_ROOT_SCOPE}.histogram group must be a ThreadGroup")
    if group.kind != "block":
        raise NotImplementedError(f"{_ROOT_SCOPE}.histogram requires a block group")
    if isinstance(samples, ThreadData):
        sample_type, sample_values = _provider_support.resolve_thread_data_value_type(
            samples,
            allowed=_HISTOGRAM_SAMPLE_TYPES,
            feature="histogram",
            scope=_ROOT_SCOPE,
            resolve_type=_resolve_histogram_type,
        )
        items_per_thread = samples.items_per_thread
        sample_values = tuple(sample_values)
    else:
        sample_type = _resolve_histogram_type(
            samples,
            allowed=_HISTOGRAM_SAMPLE_TYPES,
            feature="histogram",
        )
        items_per_thread = 1
        sample_values = (samples,)
    counter_type = (
        Int32
        if counter_dtype is None
        else _resolve_histogram_type(
            counter_dtype,
            allowed=_HISTOGRAM_COUNTER_TYPES,
            feature="histogram",
        )
    )
    request = _make_request(
        group=group,
        launch=launch,
        sample_type=sample_type,
        counter_type=counter_type,
        items_per_thread=items_per_thread,
        bins=bins,
        bins_per_thread=bins_per_thread,
        algorithm=algorithm,
        source=source,
    )
    validate_operand_domains(
        request.plan.resolved_group,
        {"samples": samples},
        scope=_ROOT_SCOPE,
        primitive_name="histogram",
    )
    _provider_support.register_request(request)
    result_tensor = _cute.make_rmem_tensor(bins_per_thread, counter_type)
    arguments = request.bind_ffi_arguments(
        sample_values,
        result_tensor.iterator.llvm_ptr,
    )
    ffi(
        name=request.symbol_name,
        params_types=list(request.ffi_param_types),
        return_type=None,
    )(*arguments)
    result_values = tuple(result_tensor[index] for index in range(bins_per_thread))
    assert request.plan.result is not None
    result_metadata = metadata_for_group(
        request.plan.resolved_group,
        visibility=request.plan.result.visibility,
    )
    return attach_thread_data_metadata(
        ThreadData.from_values(*result_values, dtype=counter_type),
        result_metadata,
    )


__all__ = ["CutlassHistogramArtifact", "provider_histogram"]
