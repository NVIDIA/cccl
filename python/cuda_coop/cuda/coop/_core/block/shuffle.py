# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral CUB BlockShuffle semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Any

from .._algorithm import Algorithm, AlgorithmSpec
from .._bindings import ArgumentBinding, BindingKind, i32_parameter
from .._symbols import semantic_token
from .._types import (
    UINT32,
    Array,
    CxxFunction,
    Dependency,
    Reference,
    TemplateParameter,
    TempStorageParameter,
    Value,
)
from ._common import normalize_block_dim, normalize_positive_int


class BlockShuffleMode(str, Enum):
    OFFSET = "offset"
    ROTATE = "rotate"
    UP = "up"
    DOWN = "down"

    @property
    def cub_method_name(self) -> str:
        return self.value.capitalize()

    @property
    def allows_negative_distance(self) -> bool:
        return self is BlockShuffleMode.OFFSET

    @classmethod
    def from_cub_method_name(cls, method_name: str) -> "BlockShuffleMode":
        try:
            return cls(method_name.lower())
        except (AttributeError, ValueError) as exc:
            raise ValueError(
                f"unsupported CUB BlockShuffle method name: {method_name!r}"
            ) from exc


class BlockShuffleValueKind(str, Enum):
    SCALAR = "scalar"
    ARRAY = "array"


_T = Dependency("T")
_ITEMS_PER_THREAD = Dependency("ITEMS_PER_THREAD")
_U32_MAX = (1 << 32) - 1
_TEMPLATE_PARAMETERS = (
    TemplateParameter("T"),
    TemplateParameter("BLOCK_DIM_X"),
    TemplateParameter("BLOCK_DIM_Y"),
    TemplateParameter("BLOCK_DIM_Z"),
)


def _u32_parameter(
    option: ArgumentBinding,
    *,
    name: str,
    omitted_value: int | None = None,
) -> Value | CxxFunction | None:
    if option.kind is BindingKind.OMITTED:
        if omitted_value is None:
            return None
        value = omitted_value
    elif option.kind is BindingKind.RUNTIME:
        return Value(UINT32, name=name)
    else:
        value = option.value
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"static {name} must be an integer")
    normalized = int(value)
    if not 0 <= normalized <= _U32_MAX:
        raise ValueError(f"static {name} must fit an unsigned 32-bit integer")
    return CxxFunction(str(normalized), UINT32, name=name)


@dataclass(frozen=True)
class BlockShuffleSemantics:
    """Dimension-independent scalar or array BlockShuffle contract."""

    dtype: Any
    mode: BlockShuffleMode
    value_kind: BlockShuffleValueKind
    items_per_thread: int | None
    distance: ArgumentBinding
    parameters: tuple[Any, ...]

    @property
    def is_array(self) -> bool:
        return self.value_kind is BlockShuffleValueKind.ARRAY

    @property
    def method_name(self) -> str:
        return self.mode.cub_method_name

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            "block_shuffle",
            semantic_token(self.dtype),
            self.mode.value,
            self.value_kind.value,
            self.items_per_thread,
            self.distance.semantic_key,
            semantic_token(self.parameters),
        )


@dataclass(frozen=True)
class BlockShuffleSpec:
    """Fully specialized CUB BlockShuffle semantics."""

    specialization: AlgorithmSpec
    call: BlockShuffleSemantics
    block_dim: tuple[int, int, int]

    @property
    def mode(self) -> BlockShuffleMode:
        return self.call.mode

    @property
    def value_kind(self) -> BlockShuffleValueKind:
        return self.call.value_kind

    @property
    def items_per_thread(self) -> int | None:
        return self.call.items_per_thread

    @property
    def distance(self) -> ArgumentBinding:
        return self.call.distance

    @property
    def method_name(self) -> str:
        return self.specialization.method_name

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.specialization.semantic_key


def make_block_shuffle_semantics(
    *,
    dtype: Any,
    mode: str | BlockShuffleMode,
    items_per_thread: int | None = None,
    distance: ArgumentBinding | None = None,
) -> BlockShuffleSemantics:
    """Build a normalized dimension-independent BlockShuffle contract."""

    if dtype is None:
        raise ValueError("dtype must be provided")
    mode = BlockShuffleMode(mode)
    distance = ArgumentBinding.omitted() if distance is None else distance
    if not isinstance(distance, ArgumentBinding):
        raise TypeError("distance must be an ArgumentBinding")
    if distance.kind is BindingKind.STATIC:
        # Materializing the parameter performs exact scalar-ABI validation.
        if mode is BlockShuffleMode.ROTATE:
            _u32_parameter(distance, name="distance")
        else:
            i32_parameter(distance, name="distance")
        if int(distance.value) < 0 and not mode.allows_negative_distance:
            raise ValueError(f"{mode.value} distance must be non-negative")
        distance = ArgumentBinding.static(int(distance.value))
    if items_per_thread is None:
        value_kind = BlockShuffleValueKind.SCALAR
    else:
        items_per_thread = normalize_positive_int(
            "items_per_thread",
            items_per_thread,
        )
        value_kind = BlockShuffleValueKind.ARRAY

    parameters: list[Any] = [TempStorageParameter()]
    if value_kind is BlockShuffleValueKind.SCALAR:
        distance_parameter = (
            _u32_parameter(distance, name="distance", omitted_value=1)
            if mode is BlockShuffleMode.ROTATE
            else i32_parameter(distance, name="distance", omitted_value=1)
        )
        assert distance_parameter is not None
        parameters.extend(
            (
                Reference(_T, name="input_item"),
                Reference(_T, name="output_item", is_output=True),
                distance_parameter,
            )
        )
    else:
        parameters.extend(
            (
                Array(_T, _ITEMS_PER_THREAD, name="input_items"),
                Array(
                    _T,
                    _ITEMS_PER_THREAD,
                    name="output_items",
                    is_output=True,
                    is_return=False,
                ),
            )
        )
        if distance.kind is not BindingKind.OMITTED:
            parameters.append(i32_parameter(distance, name="distance"))

    return BlockShuffleSemantics(
        dtype=dtype,
        mode=mode,
        value_kind=value_kind,
        items_per_thread=items_per_thread,
        distance=distance,
        parameters=tuple(parameters),
    )


def make_block_shuffle_spec(
    *,
    dtype: Any,
    block_dim: tuple[int, int, int],
    mode: str | BlockShuffleMode,
    items_per_thread: int | None = None,
    distance: ArgumentBinding | None = None,
) -> BlockShuffleSpec:
    """Build a fully specialized public CUB BlockShuffle description."""

    block_dim = normalize_block_dim(block_dim)
    call = make_block_shuffle_semantics(
        dtype=dtype,
        mode=mode,
        items_per_thread=items_per_thread,
        distance=distance,
    )
    if call.is_array:
        if call.mode not in {BlockShuffleMode.UP, BlockShuffleMode.DOWN}:
            raise ValueError("CUB array BlockShuffle supports only Up and Down")
        if call.distance.kind is not BindingKind.OMITTED:
            raise ValueError("CUB array BlockShuffle does not accept distance")
    elif call.mode not in {BlockShuffleMode.OFFSET, BlockShuffleMode.ROTATE}:
        raise ValueError("CUB scalar BlockShuffle supports only Offset and Rotate")
    elif (
        call.mode is BlockShuffleMode.ROTATE
        and call.distance.kind is not BindingKind.RUNTIME
    ):
        block_threads = block_dim[0] * block_dim[1] * block_dim[2]
        distance_value = (
            1 if call.distance.kind is BindingKind.OMITTED else int(call.distance.value)
        )
        if not 1 <= distance_value < block_threads:
            raise ValueError(
                "static rotate distance must satisfy "
                f"1 <= distance < block_threads ({block_threads})"
            )

    template_arguments = {
        "T": dtype,
        "BLOCK_DIM_X": block_dim[0],
        "BLOCK_DIM_Y": block_dim[1],
        "BLOCK_DIM_Z": block_dim[2],
    }
    if call.is_array:
        template_arguments["ITEMS_PER_THREAD"] = call.items_per_thread
    specialization = Algorithm(
        struct_name="BlockShuffle",
        method_name=call.method_name,
        c_name="block_shuffle",
        includes=("cub/block/block_shuffle.cuh",),
        template_parameters=_TEMPLATE_PARAMETERS,
        parameters=(call.parameters,),
        fake_return=True,
    ).specialize(
        template_arguments,
        metadata={
            "scope": "block",
            "primitive": "shuffle",
            "mode": call.mode.value,
            "value_kind": call.value_kind.value,
            "distance": call.distance.kind.value,
        },
    )
    return BlockShuffleSpec(
        specialization=specialization,
        call=call,
        block_dim=block_dim,
    )


__all__ = [
    "BlockShuffleMode",
    "BlockShuffleSemantics",
    "BlockShuffleSpec",
    "BlockShuffleValueKind",
    "make_block_shuffle_semantics",
    "make_block_shuffle_spec",
]
