# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Type independent Histogram counters and bind the existing provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import _bridges as nb
from . import _templates
from ._callbacks import constant
from ._errors import TypedPipelineError
from ._results import ArrayAllocation
from ._types import PRODUCER, CoopGroupType, ThreadDataType


def _histogram(group, samples, **kwargs):
    raise NotImplementedError("cuda.coop: kernel only")


def _qualified_histogram(group, samples, **kwargs):
    raise NotImplementedError("cuda.coop: kernel only")


OPERATIONS = {_histogram: False, _qualified_histogram: True}


def _facts(args, kws, qualified):
    from ..._core.block._common import normalize_positive_int
    from ..._core.block.histogram import (
        normalize_histogram_algorithm,
        validate_histogram_dtype,
    )
    from .._compiler._parameters import normalize_dtype_param

    if len(args) != 2 or not isinstance(args[0], CoopGroupType):
        raise TypedPipelineError("coop.histogram: expected group and samples")
    samples = args[1]
    if not qualified and not isinstance(samples, ThreadDataType):
        raise TypedPipelineError(
            "cuda.coop.histogram samples must be ThreadData"
        )
    is_array = isinstance(samples, nb.types.Array)
    if is_array and (samples.ndim != 1 or samples.layout != "C"):
        raise TypedPipelineError(
            "histogram samples require a fixed-size 1D array"
        )
    dtype = samples.dtype if is_array else nb.unliteral(samples)
    if not dtype.is_precise():
        return None
    validate_histogram_dtype(dtype)
    counter_type = kws.get("counter_dtype")
    if _templates._omitted(counter_type):
        counter = nb.types.int32
    else:
        counter = _templates._payload_dtype(counter_type)
        if counter is None:
            counter = normalize_dtype_param(constant(counter_type))
    validate_histogram_dtype(counter, counter=True)
    bins = normalize_positive_int("bins", constant(kws.get("bins")))
    slots = normalize_positive_int(
        "bins_per_thread",
        constant(kws.get("bins_per_thread", nb.types.literal(1))),
    )
    algorithm = normalize_histogram_algorithm(
        constant(kws.get("algorithm", nb.types.literal("atomic")))
    )
    return dtype, counter, bins, slots, algorithm, is_array


@dataclass(frozen=True, eq=False)
class HistogramRequest:
    kind: str
    dtype: Any
    counter_dtype: Any
    items_per_thread: int
    bins: int
    bins_per_thread: int
    algorithm: str
    group: CoopGroupType
    call: Any
    is_array: bool
    storage_options: tuple = ()
    controls: tuple = ()

    @property
    def key(self):
        return self.call.semantic_key, self.storage_options, self.is_array

    @property
    def allocations(self):
        return (ArrayAllocation(self.counter_dtype, self.bins_per_thread),)

    @property
    def result(self):
        return ("allocation", 0)

    @property
    def provider_operands(self):
        samples = (
            ("pointer", 1) if self.is_array else ("box", 1, self.dtype, None)
        )
        return samples, ("allocation", 0)

    def complete(self, block, plan):
        from .._lowering import _histogram as providers

        return providers.histogram, {
            "sample_dtype": self.dtype,
            "counter_dtype": self.counter_dtype,
            "threads_per_block": block,
            "items_per_thread": self.items_per_thread,
            "bins": self.bins,
            "bins_per_thread": self.bins_per_thread,
            "algorithm": self.algorithm,
        }


_REQUESTS = {}


def request_for(args, kws, *, qualified, extent=None):
    from ..._core import make_group_primitive_call
    from ..._core.group.histogram import GroupHistogramSemantics
    from .._lowering._core import NumbaMlirCoreAdapter

    facts = _facts(args, kws, qualified)
    if facts is None:
        return None
    dtype, counter, bins, slots, algorithm, is_array = facts
    items = args[1].items if isinstance(args[1], ThreadDataType) else extent
    if not is_array:
        items = 1
    if items is None:
        raise TypedPipelineError("histogram samples require a static extent")
    adapter = NumbaMlirCoreAdapter()
    call = make_group_primitive_call(
        args[0].group,
        GroupHistogramSemantics(
            adapter.core_dtype(dtype),
            items,
            bins,
            slots,
            adapter.core_dtype(counter),
            algorithm,
        ),
    )
    storage = tuple(_templates.storage_options(kws.get("temp_storage")).items())
    key = call.semantic_key, storage, is_array
    if key not in _REQUESTS:
        _REQUESTS[key] = HistogramRequest(
            "histogram",
            dtype,
            counter,
            items,
            bins,
            slots,
            algorithm,
            args[0],
            call,
            is_array,
            storage,
        )
    return _REQUESTS[key]


def _typing(qualified):
    def generic(self, args, kws):
        facts = _facts(args, kws, qualified)
        if facts is None:
            return None
        _, counter, _, slots, _, _ = facts
        return nb.signature(
            ThreadDataType(counter, slots, PRODUCER), *args, *kws.values()
        )

    return generic


def site_request(analysis, stmt, kind, qualified):
    from ._inference import static_call_arguments

    call = stmt.value
    args, kws = static_call_arguments(
        analysis.func_ir,
        call,
        analysis.typemap[call.func.name].typing_key,
        [analysis.typemap[arg.name] for arg in call.args],
        {name: analysis.typemap[value.name] for name, value in call.kws},
    )
    return request_for(
        args,
        kws,
        qualified=qualified,
        extent=analysis._array_extent(stmt, kind, index=1),
    )


def register():
    from ._inference import numpy_integer_literal, operation_literal_arguments

    for operation, qualified in OPERATIONS.items():
        _templates.OPERATIONS[operation] = "histogram"
        operation_literal_arguments[operation] = {
            name: numpy_integer_literal for name in ("bins", "bins_per_thread")
        }
        if qualified:
            _templates.QUALIFIED_OPERATIONS.add(operation)
        _templates.register_template(
            operation,
            _typing(qualified),
            operation.__name__,
            prefer_literal=True,
        )
    for public, name in _templates.kernel_primitives().items():
        if _templates.group_operation(public) != "histogram":
            continue
        qualified = name.startswith("cuda.coop.numba_mlir.")
        operation = _qualified_histogram if qualified else _histogram
        nb.attach_inline_wrapper(
            public, _templates.result_wrapper(public, name, operation)
        )
        _templates.register_template(
            public, _templates._not_inlined(name), name
        )
