# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral CUB BlockHistogram semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Any

from .._algorithm import Algorithm, AlgorithmSpec
from .._symbols import semantic_token
from .._types import Array, Dependency, TemplateParameter, TempStorageParameter


class BlockHistogramAlgorithm(str, Enum):
    """CUB algorithm used to construct a block-wide histogram."""

    ATOMIC = "atomic"
    SORT = "sort"

    @property
    def cpp(self) -> str:
        return f"::cub::BLOCK_HISTO_{self.name}"


class BlockHistogramOperation(str, Enum):
    """BlockHistogram instance lifecycle or public member operation."""

    INSTANCE = "instance"
    INIT = "init"
    HISTOGRAM = "histogram"
    COMPOSITE = "composite"

    @property
    def method_name(self) -> str:
        if self is BlockHistogramOperation.INSTANCE:
            return "Histogram"
        if self is BlockHistogramOperation.INIT:
            return "InitHistogram"
        return self.value.capitalize()

    @property
    def c_name(self) -> str:
        if self in {
            BlockHistogramOperation.INSTANCE,
            BlockHistogramOperation.HISTOGRAM,
        }:
            return "block_histogram"
        return f"block_histogram_{self.value}"


_ALGORITHM_ALIASES = {
    "atomic": BlockHistogramAlgorithm.ATOMIC,
    "block_histo_atomic": BlockHistogramAlgorithm.ATOMIC,
    "sort": BlockHistogramAlgorithm.SORT,
    "block_histo_sort": BlockHistogramAlgorithm.SORT,
}
_T = Dependency("T")
_ITEMS_PER_THREAD = Dependency("ITEMS_PER_THREAD")
_BINS = Dependency("BINS")
_TEMPLATE_PARAMETERS = (
    TemplateParameter("T"),
    TemplateParameter("BLOCK_DIM_X"),
    TemplateParameter("ITEMS_PER_THREAD"),
    TemplateParameter("BINS"),
    TemplateParameter("ALGORITHM"),
    TemplateParameter("BLOCK_DIM_Y"),
    TemplateParameter("BLOCK_DIM_Z"),
)


def normalize_block_histogram_algorithm(
    algorithm: Any,
) -> BlockHistogramAlgorithm:
    """Normalize frontend enum and CUB spellings to one algorithm value."""

    if algorithm is None:
        return BlockHistogramAlgorithm.ATOMIC
    if isinstance(algorithm, BlockHistogramAlgorithm):
        return algorithm

    token = getattr(algorithm, "name", algorithm)
    if isinstance(token, str):
        token = token.strip().split(".")[-1].split("::")[-1]
        token = token.lower().replace("-", "_")
        try:
            return _ALGORITHM_ALIASES[token]
        except KeyError:
            pass
    raise ValueError(f"unsupported BlockHistogram algorithm {algorithm!r}")


def normalize_block_histogram_positive_int(
    name: str,
    value: Any,
    *,
    scope: str = "histogram",
) -> int:
    """Normalize one compile-time positive-integer Histogram parameter."""

    message = f"{scope} {name} must be a compile-time positive integer"
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(message)
    value = int(value)
    if value < 1:
        raise ValueError(message)
    return value


def _positive_int(name: str, value: Any) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def validate_block_histogram_output_capacity(
    *,
    bins: Any,
    bins_per_thread: Any,
    block_threads: Any,
    scope: str = "histogram",
) -> None:
    """Require enough striped per-thread result slots for every bin."""

    bins = normalize_block_histogram_positive_int("bins", bins, scope=scope)
    bins_per_thread = normalize_block_histogram_positive_int(
        "bins_per_thread",
        bins_per_thread,
        scope=scope,
    )
    block_threads = normalize_block_histogram_positive_int(
        "block_threads",
        block_threads,
        scope=scope,
    )
    capacity = block_threads * bins_per_thread
    if bins > capacity:
        required = (bins + block_threads - 1) // block_threads
        raise ValueError(
            "histogram bins_per_thread is too small for "
            f"{bins} bins and block size {block_threads}; "
            f"need at least {required}"
        )


def _histogram_array(counter_dtype: Any, *, is_output: bool) -> Array:
    return Array(
        counter_dtype,
        _BINS,
        name="histogram",
        is_output=is_output,
        is_inout=not is_output,
        is_return=False if is_output else None,
    )


@dataclass(frozen=True)
class BlockHistogramSemantics:
    """Dimension-independent BlockHistogram member-call contract.

    ``bins`` may be ``None`` for providers with a runtime-width shim. Such a
    record remains useful for common operation, dtype, item-count, algorithm,
    and CUB parameter-shape validation, but cannot be specialized as a CUB
    class until a static bin count is available.
    """

    item_dtype: Any
    counter_dtype: Any
    items_per_thread: int
    bins: int | None
    algorithm: BlockHistogramAlgorithm
    operation: BlockHistogramOperation
    parameters: tuple[Any, ...]

    @property
    def method_name(self) -> str:
        return self.operation.method_name

    @property
    def c_name(self) -> str:
        return self.operation.c_name

    @property
    def algorithm_cpp(self) -> str:
        return self.algorithm.cpp

    @property
    def has_static_bins(self) -> bool:
        return self.bins is not None

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            "block_histogram",
            semantic_token(self.item_dtype),
            semantic_token(self.counter_dtype),
            self.items_per_thread,
            self.bins,
            self.algorithm.value,
            self.operation.value,
            semantic_token(self.parameters),
        )


@dataclass(frozen=True)
class BlockHistogramSpec:
    """Fully specialized CUB BlockHistogram call semantics."""

    specialization: AlgorithmSpec
    call: BlockHistogramSemantics
    block_dim: tuple[int, int, int]

    @property
    def item_dtype(self) -> Any:
        return self.call.item_dtype

    @property
    def counter_dtype(self) -> Any:
        return self.call.counter_dtype

    @property
    def items_per_thread(self) -> int:
        return self.call.items_per_thread

    @property
    def bins(self) -> int:
        assert self.call.bins is not None
        return self.call.bins

    @property
    def algorithm(self) -> BlockHistogramAlgorithm:
        return self.call.algorithm

    @property
    def operation(self) -> BlockHistogramOperation:
        return self.call.operation

    @property
    def method_name(self) -> str:
        return self.specialization.method_name

    @property
    def algorithm_cpp(self) -> str:
        return self.call.algorithm_cpp

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.specialization.semantic_key


def make_block_histogram_semantics(
    *,
    item_dtype: Any,
    counter_dtype: Any,
    items_per_thread: int,
    bins: int | None,
    algorithm: Any = None,
    operation: str | BlockHistogramOperation = BlockHistogramOperation.HISTOGRAM,
) -> BlockHistogramSemantics:
    """Build the normalized dimension-independent histogram contract."""

    if item_dtype is None:
        raise ValueError("item dtype must be provided")
    if counter_dtype is None:
        raise ValueError("counter dtype must be provided")
    items_per_thread = _positive_int("items_per_thread", items_per_thread)
    if bins is not None:
        bins = _positive_int("bins", bins)
    algorithm = normalize_block_histogram_algorithm(algorithm)
    operation = BlockHistogramOperation(operation)

    parameters: list[Any] = [TempStorageParameter()]
    items = Array(_T, _ITEMS_PER_THREAD, name="items")
    if operation is BlockHistogramOperation.INSTANCE:
        pass
    elif operation is BlockHistogramOperation.INIT:
        parameters.append(_histogram_array(counter_dtype, is_output=True))
    elif operation is BlockHistogramOperation.HISTOGRAM:
        parameters.extend((items, _histogram_array(counter_dtype, is_output=True)))
    else:
        parameters.extend((items, _histogram_array(counter_dtype, is_output=False)))

    return BlockHistogramSemantics(
        item_dtype=item_dtype,
        counter_dtype=counter_dtype,
        items_per_thread=items_per_thread,
        bins=bins,
        algorithm=algorithm,
        operation=operation,
        parameters=tuple(parameters),
    )


def make_block_histogram_spec(
    *,
    item_dtype: Any,
    counter_dtype: Any,
    block_dim: tuple[int, int, int],
    items_per_thread: int,
    bins: int,
    algorithm: Any = None,
    operation: str | BlockHistogramOperation = BlockHistogramOperation.HISTOGRAM,
) -> BlockHistogramSpec:
    """Build a fully specialized CUB BlockHistogram description."""

    block_dim = tuple(block_dim)
    if len(block_dim) != 3 or any(
        not isinstance(dim, Integral) or isinstance(dim, bool) or dim < 1
        for dim in block_dim
    ):
        raise ValueError("block_dim must contain three positive dimensions")
    block_dim = tuple(int(dim) for dim in block_dim)
    call = make_block_histogram_semantics(
        item_dtype=item_dtype,
        counter_dtype=counter_dtype,
        items_per_thread=items_per_thread,
        bins=bins,
        algorithm=algorithm,
        operation=operation,
    )
    assert call.bins is not None
    specialization = Algorithm(
        struct_name="BlockHistogram",
        method_name=call.method_name,
        c_name=call.c_name,
        includes=("cub/block/block_histogram.cuh",),
        template_parameters=_TEMPLATE_PARAMETERS,
        parameters=(call.parameters,),
    ).specialize(
        {
            "T": item_dtype,
            "BLOCK_DIM_X": block_dim[0],
            "ITEMS_PER_THREAD": items_per_thread,
            "BINS": call.bins,
            "ALGORITHM": call.algorithm_cpp,
            "BLOCK_DIM_Y": block_dim[1],
            "BLOCK_DIM_Z": block_dim[2],
        },
        metadata={
            "scope": "block",
            "primitive": "histogram",
            "operation": call.operation,
            "algorithm": call.algorithm,
            "counter_dtype": counter_dtype,
            "bins": call.bins,
        },
    )
    return BlockHistogramSpec(
        specialization=specialization,
        call=call,
        block_dim=block_dim,
    )


__all__ = [
    "BlockHistogramAlgorithm",
    "BlockHistogramOperation",
    "BlockHistogramSemantics",
    "BlockHistogramSpec",
    "make_block_histogram_semantics",
    "make_block_histogram_spec",
    "normalize_block_histogram_algorithm",
]
