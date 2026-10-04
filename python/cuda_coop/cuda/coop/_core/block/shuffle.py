# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Describe CUB BlockShuffle scalar and array calls.

Scalar Offset and Rotate select another thread's value. Array Up and Down
shift the flattened blocked tile by one item. These forms have different CUB
overloads: scalar calls use a distance, while array calls use a fixed
per-thread extent. The builders keep these choices separate, then attach exact
block dimensions for backend materialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Any

from .._algorithm import Algorithm
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
    """Select scalar thread movement or a unit array shift.

    ``OFFSET`` permits signed distances, including zero and negative values.
    ``ROTATE`` wraps within the block. ``UP`` and ``DOWN`` select CUB's array
    shift methods; their missing boundary item has no defined value.
    """

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


class BlockShuffleValueKind(str, Enum):
    """Distinguish one scalar from a fixed per-thread array.

    This choice selects the CUB overload and whether an item-count template
    argument is required. An array of one item remains an array call.
    """

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
    """Translate a distance binding to CUB's unsigned scalar parameter.

    An omitted binding uses ``omitted_value`` when supplied, otherwise adds no
    parameter. A runtime binding becomes a value operand. A static binding
    must fit unsigned 32 bits and becomes an embedded C++ literal.
    """

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
    """Hold Shuffle choices before a block shape is attached.

    ``value_kind`` and ``items_per_thread`` distinguish scalar and fixed-array
    calls. ``distance`` preserves whether the control is omitted, embedded, or
    passed at runtime. ``parameters`` describes the resulting CUB call; the
    specialization builder checks that its mode and value form agree.
    """

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
class BlockShuffleSpecialization:
    """Pair a block-specific CUB Shuffle algorithm with its call semantics.

    The algorithm holds native parameters and bound template arguments for a
    backend. The call record retains mode, value form, and distance binding
    for planning; ``block_dim`` records the exact shape used to build it.
    """

    specialization: Algorithm
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
    """Normalize Shuffle operands before selecting a block implementation.

    A missing item count selects scalar references; a positive count selects
    input and output arrays. Preserve distance binding so a runtime control
    stays an operand and a static control becomes an embedded literal. Offset
    accepts signed 32-bit distances; Rotate uses unsigned 32 bits.

    This step validates dtype, extent, and distance representation. The
    specialization builder additionally checks mode/value-form combinations
    and any static Rotate bound against the block size.
    """

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


def make_block_shuffle_specialization(
    *,
    dtype: Any,
    block_dim: tuple[int, int, int],
    mode: str | BlockShuffleMode,
    items_per_thread: int | None = None,
    distance: ArgumentBinding | None = None,
) -> BlockShuffleSpecialization:
    """Build the CUB Shuffle algorithm for one exact block shape.

    Array calls must use Up or Down with no distance operand. Scalar calls
    must use Offset or Rotate. A known Rotate distance must be at least one
    and smaller than the block size; runtime controls retain their binding for
    backend validation.

    Return the normalized semantics and algorithm with dtype, block shape, and
    any array extent bound to template arguments. Backend materialization
    supplies concrete compiler types and generates the callable.
    """

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
        raise ValueError(
            "CUB scalar BlockShuffle supports only Offset and Rotate"
        )
    elif (
        call.mode is BlockShuffleMode.ROTATE
        and call.distance.kind is not BindingKind.RUNTIME
    ):
        block_threads = block_dim[0] * block_dim[1] * block_dim[2]
        distance_value = (
            1
            if call.distance.kind is BindingKind.OMITTED
            else int(call.distance.value)
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
        template_arguments=template_arguments,
        metadata={
            "scope": "block",
            "primitive": "shuffle",
            "mode": call.mode.value,
            "value_kind": call.value_kind.value,
            "distance": call.distance.kind.value,
        },
    )
    return BlockShuffleSpecialization(
        specialization=specialization,
        call=call,
        block_dim=block_dim,
    )


__all__ = [
    "BlockShuffleMode",
    "BlockShuffleSemantics",
    "BlockShuffleSpecialization",
    "BlockShuffleValueKind",
    "make_block_shuffle_semantics",
    "make_block_shuffle_specialization",
]
