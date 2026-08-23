# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral block-shuffle semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .._algorithm import Algorithm, AlgorithmSpec
from .._bindings import ArgumentBinding, BindingKind, i32_parameter
from .._symbols import semantic_token
from .._types import (
    Array,
    Dependency,
    Pointer,
    Reference,
    TemplateParameter,
    TempStorageParameter,
)


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

    @property
    def allows_block_prefix(self) -> bool:
        return self in {BlockShuffleMode.DOWN, BlockShuffleMode.OFFSET}

    @property
    def allows_block_suffix(self) -> bool:
        return self is BlockShuffleMode.UP

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
_TEMPLATE_PARAMETERS = (
    TemplateParameter("T"),
    TemplateParameter("BLOCK_DIM_X"),
    TemplateParameter("BLOCK_DIM_Y"),
    TemplateParameter("BLOCK_DIM_Z"),
)


def _boundary_parameter(name: str) -> Pointer:
    return Pointer(
        _T,
        name=name,
        is_output=True,
        is_return=False,
        is_array_pointer=True,
        deref_on_call=True,
    )


@dataclass(frozen=True)
class BlockShuffleSemantics:
    """Dimension-independent block-shuffle call contract.

    The semantic contract also represents CuTe's extended multi-item
    offset/rotate and arbitrary-distance up/down forms. The CUB specialization
    builder below accepts only CUB's scalar Offset/Rotate and array Up/Down
    method shapes. Scalar parameter graphs place distance before an optional
    boundary output; array graphs preserve the CUB input/output/boundary order
    and append distance only for extended non-CUB forms. CuTe consumes the
    named semantic fields rather than lowering this tuple positionally.
    """

    dtype: Any
    mode: BlockShuffleMode
    value_kind: BlockShuffleValueKind
    items_per_thread: int | None
    distance: ArgumentBinding
    block_prefix: bool
    block_suffix: bool
    parameters: tuple[Any, ...]

    @property
    def is_array(self) -> bool:
        return self.value_kind is BlockShuffleValueKind.ARRAY

    @property
    def has_boundary_output(self) -> bool:
        return self.block_prefix or self.block_suffix

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
            semantic_token(self.distance),
            self.block_prefix,
            self.block_suffix,
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
    def block_prefix(self) -> bool:
        return self.call.block_prefix

    @property
    def block_suffix(self) -> bool:
        return self.call.block_suffix

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
    block_prefix: bool = False,
    block_suffix: bool = False,
) -> BlockShuffleSemantics:
    """Build the normalized dimension-independent block-shuffle contract."""

    if dtype is None:
        raise ValueError("dtype must be provided")
    mode = BlockShuffleMode(mode)
    distance = ArgumentBinding.omitted() if distance is None else distance
    if not isinstance(distance, ArgumentBinding):
        raise TypeError("distance must be an ArgumentBinding")
    if distance.kind is BindingKind.STATIC:
        if not isinstance(distance.value, int) or isinstance(distance.value, bool):
            raise ValueError("static distance must be an integer")
        if distance.value < 0 and not mode.allows_negative_distance:
            raise ValueError(f"{mode.value} distance must be non-negative")
    if items_per_thread is None:
        value_kind = BlockShuffleValueKind.SCALAR
    else:
        if (
            not isinstance(items_per_thread, int)
            or isinstance(items_per_thread, bool)
            or items_per_thread < 1
        ):
            raise ValueError("items_per_thread must be a positive integer")
        value_kind = BlockShuffleValueKind.ARRAY
    if not isinstance(block_prefix, bool) or not isinstance(block_suffix, bool):
        raise ValueError("block boundary selectors must be boolean")
    if block_prefix and block_suffix:
        raise ValueError("block_prefix and block_suffix are mutually exclusive")
    if block_prefix and not mode.allows_block_prefix:
        raise ValueError(f"block_prefix is not valid for {mode.value} shuffles")
    if block_suffix and not mode.allows_block_suffix:
        raise ValueError(f"block_suffix is not valid for {mode.value} shuffles")

    parameters: list[Any] = [TempStorageParameter()]
    if value_kind is BlockShuffleValueKind.SCALAR:
        parameters.extend(
            (
                Reference(_T, name="input_item"),
                Reference(_T, name="output_item", is_output=True),
                i32_parameter(distance, name="distance", omitted_value=1),
            )
        )
        if block_suffix:
            parameters.append(_boundary_parameter("block_suffix"))
        if block_prefix:
            parameters.append(_boundary_parameter("block_prefix"))
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
        if block_suffix:
            parameters.append(_boundary_parameter("block_suffix"))
        if block_prefix:
            parameters.append(_boundary_parameter("block_prefix"))
        if distance.kind is not BindingKind.OMITTED:
            parameters.append(i32_parameter(distance, name="distance"))

    return BlockShuffleSemantics(
        dtype=dtype,
        mode=mode,
        value_kind=value_kind,
        items_per_thread=items_per_thread,
        distance=distance,
        block_prefix=block_prefix,
        block_suffix=block_suffix,
        parameters=tuple(parameters),
    )


def make_block_shuffle_spec(
    *,
    dtype: Any,
    block_dim: tuple[int, int, int],
    mode: str | BlockShuffleMode,
    items_per_thread: int | None = None,
    distance: ArgumentBinding | None = None,
    block_prefix: bool = False,
    block_suffix: bool = False,
) -> BlockShuffleSpec:
    """Build a fully specialized CUB BlockShuffle description."""

    block_dim = tuple(block_dim)
    if len(block_dim) != 3 or any(dim < 1 for dim in block_dim):
        raise ValueError("block_dim must contain three positive dimensions")
    call = make_block_shuffle_semantics(
        dtype=dtype,
        mode=mode,
        items_per_thread=items_per_thread,
        distance=distance,
        block_prefix=block_prefix,
        block_suffix=block_suffix,
    )
    if call.is_array:
        if call.mode not in {BlockShuffleMode.UP, BlockShuffleMode.DOWN}:
            raise ValueError("CUB array BlockShuffle supports only Up and Down")
        if call.distance.kind is not BindingKind.OMITTED:
            raise ValueError("CUB array BlockShuffle does not accept distance")
    else:
        if call.mode not in {BlockShuffleMode.OFFSET, BlockShuffleMode.ROTATE}:
            raise ValueError("CUB scalar BlockShuffle supports only Offset and Rotate")
        if call.has_boundary_output:
            raise ValueError("CUB scalar BlockShuffle does not return block boundaries")

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
            "block_prefix": call.block_prefix,
            "block_suffix": call.block_suffix,
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
