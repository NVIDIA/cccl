# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Describe CUB BlockExchange calls without compiler-specific types.

Exchange rearranges a fixed number of items held by each block thread. Layout
modes convert between blocked, striped, and warp-striped ordering; scatter
modes select destinations from rank arrays. The semantics record validates
operands without a launch shape. The specialization builder adds exact block
dimensions and CUB template arguments. Both produce descriptions for a backend
to materialize; neither performs the exchange.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .._algorithm import Algorithm
from .._symbols import semantic_token
from .._types import Array, Dependency, TemplateParameter, TempStorageParameter
from ._common import normalize_block_dim, normalize_positive_int


class BlockExchangeMode(str, Enum):
    """Select the CUB layout conversion or rank-directed scatter.

    Scatter modes consume one destination rank per input item. The guarded
    variant skips negative ranks; the flagged variant uses explicit validity
    flags. The properties below select the required operands and CUB method.
    """

    STRIPED_TO_BLOCKED = "striped_to_blocked"
    BLOCKED_TO_STRIPED = "blocked_to_striped"
    WARP_STRIPED_TO_BLOCKED = "warp_striped_to_blocked"
    BLOCKED_TO_WARP_STRIPED = "blocked_to_warp_striped"
    SCATTER_TO_BLOCKED = "scatter_to_blocked"
    SCATTER_TO_STRIPED = "scatter_to_striped"
    SCATTER_TO_STRIPED_GUARDED = "scatter_to_striped_guarded"
    SCATTER_TO_STRIPED_FLAGGED = "scatter_to_striped_flagged"

    @property
    def uses_ranks(self) -> bool:
        return self in {
            BlockExchangeMode.SCATTER_TO_BLOCKED,
            BlockExchangeMode.SCATTER_TO_STRIPED,
            BlockExchangeMode.SCATTER_TO_STRIPED_GUARDED,
            BlockExchangeMode.SCATTER_TO_STRIPED_FLAGGED,
        }

    @property
    def uses_valid_flags(self) -> bool:
        return self is BlockExchangeMode.SCATTER_TO_STRIPED_FLAGGED

    @property
    def cub_method_name(self) -> str:
        return _CUB_METHOD_NAMES[self]


class BlockExchangeValueForm(str, Enum):
    """Select whether CUB reads and writes the same per-thread array.

    ``IN_PLACE`` uses one input/output array. ``OUT_OF_PLACE`` uses separate
    input and output arrays. ``BOTH`` describes both overloads for a backend
    that supports either call form.
    """

    IN_PLACE = "in_place"
    OUT_OF_PLACE = "out_of_place"
    BOTH = "both"


_CUB_METHOD_NAMES = {
    BlockExchangeMode.STRIPED_TO_BLOCKED: "StripedToBlocked",
    BlockExchangeMode.BLOCKED_TO_STRIPED: "BlockedToStriped",
    BlockExchangeMode.WARP_STRIPED_TO_BLOCKED: "WarpStripedToBlocked",
    BlockExchangeMode.BLOCKED_TO_WARP_STRIPED: "BlockedToWarpStriped",
    BlockExchangeMode.SCATTER_TO_BLOCKED: "ScatterToBlocked",
    BlockExchangeMode.SCATTER_TO_STRIPED: "ScatterToStriped",
    BlockExchangeMode.SCATTER_TO_STRIPED_GUARDED: "ScatterToStripedGuarded",
    BlockExchangeMode.SCATTER_TO_STRIPED_FLAGGED: "ScatterToStripedFlagged",
}
_T = Dependency("T")
_ITEMS_PER_THREAD = Dependency("ITEMS_PER_THREAD")
_OFFSET_T = Dependency("OffsetT")
_VALID_FLAG_T = Dependency("ValidFlag")
_TEMPLATE_PARAMETERS = (
    TemplateParameter("T"),
    TemplateParameter("BLOCK_DIM_X"),
    TemplateParameter("ITEMS_PER_THREAD"),
    TemplateParameter("WARP_TIME_SLICING"),
    TemplateParameter("BLOCK_DIM_Y"),
    TemplateParameter("BLOCK_DIM_Z"),
)


def _in_place_parameters(mode: BlockExchangeMode) -> tuple[Any, ...]:
    """Describe scratch, one mutable item array, and any scatter controls.

    The item array remains a runtime operand. Mark it as modified, but do not
    turn it into a backend return value; the caller already owns that array.
    """

    parameters: list[Any] = [
        TempStorageParameter(),
        Array(
            _T,
            _ITEMS_PER_THREAD,
            name="input_items",
            is_inout=True,
            is_return=False,
        ),
    ]
    if mode.uses_ranks:
        parameters.append(Array(_OFFSET_T, _ITEMS_PER_THREAD, name="ranks"))
    if mode.uses_valid_flags:
        parameters.append(
            Array(_VALID_FLAG_T, _ITEMS_PER_THREAD, name="valid_flags")
        )
    return tuple(parameters)


def _out_of_place_parameters(mode: BlockExchangeMode) -> tuple[Any, ...]:
    """Describe separate input and output arrays for one Exchange call.

    Ranks and flags follow the output when the mode requires them. Mark the
    output as writable but not as a backend return value, so group lowering
    can return the independently allocated payload through its own alias.
    """

    parameters: list[Any] = [
        TempStorageParameter(),
        Array(_T, _ITEMS_PER_THREAD, name="input_items"),
        Array(
            _T,
            _ITEMS_PER_THREAD,
            name="output_items",
            is_output=True,
            is_return=False,
        ),
    ]
    if mode.uses_ranks:
        parameters.append(Array(_OFFSET_T, _ITEMS_PER_THREAD, name="ranks"))
    if mode.uses_valid_flags:
        parameters.append(
            Array(_VALID_FLAG_T, _ITEMS_PER_THREAD, name="valid_flags")
        )
    return tuple(parameters)


@dataclass(frozen=True)
class BlockExchangeSemantics:
    """Hold Exchange arguments that do not depend on block dimensions.

    The normalized mode determines whether rank and validity arrays are
    present. ``value_form`` determines which input/output overloads appear in
    ``parameters``. ``warp_time_slicing`` controls the CUB implementation; it
    is part of call identity even though it adds no runtime operand.

    Use ``make_block_exchange_semantics`` to validate these related choices.
    The group planner can inspect this record before exact block dimensions
    are attached to a CUB specialization.
    """

    dtype: Any
    mode: BlockExchangeMode
    value_form: BlockExchangeValueForm
    items_per_thread: int
    warp_time_slicing: bool
    rank_dtype: Any | None
    valid_flag_dtype: Any | None
    parameters: tuple[tuple[Any, ...], ...]

    @property
    def method_name(self) -> str:
        return self.mode.cub_method_name

    @property
    def uses_ranks(self) -> bool:
        return self.mode.uses_ranks

    @property
    def uses_valid_flags(self) -> bool:
        return self.mode.uses_valid_flags

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            "block_exchange",
            semantic_token(self.dtype),
            self.mode.value,
            self.value_form.value,
            self.items_per_thread,
            self.warp_time_slicing,
            semantic_token(self.rank_dtype),
            semantic_token(self.valid_flag_dtype),
            semantic_token(self.parameters),
        )


@dataclass(frozen=True)
class BlockExchangeSpecialization:
    """Pair validated call semantics with a block-specific CUB algorithm.

    ``specialization`` holds the native method, template arguments, and
    parameter descriptions for backend materialization. ``call`` retains the
    logical choices used by group planning and cache keys. ``block_dim`` is
    the exact three-dimensional block shape used in those arguments.
    """

    specialization: Algorithm
    call: BlockExchangeSemantics
    block_dim: tuple[int, int, int]

    @property
    def mode(self) -> BlockExchangeMode:
        return self.call.mode

    @property
    def value_form(self) -> BlockExchangeValueForm:
        return self.call.value_form

    @property
    def items_per_thread(self) -> int:
        return self.call.items_per_thread

    @property
    def warp_time_slicing(self) -> bool:
        return self.call.warp_time_slicing

    @property
    def rank_dtype(self) -> Any | None:
        return self.call.rank_dtype

    @property
    def valid_flag_dtype(self) -> Any | None:
        return self.call.valid_flag_dtype

    @property
    def method_name(self) -> str:
        return self.specialization.method_name

    @property
    def uses_ranks(self) -> bool:
        return self.call.uses_ranks

    @property
    def uses_valid_flags(self) -> bool:
        return self.call.uses_valid_flags

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.specialization.semantic_key


def make_block_exchange_semantics(
    *,
    dtype: Any,
    items_per_thread: int,
    mode: str | BlockExchangeMode,
    value_form: str
    | BlockExchangeValueForm = BlockExchangeValueForm.OUT_OF_PLACE,
    warp_time_slicing: bool = False,
    rank_dtype: Any | None = None,
    valid_flag_dtype: Any | None = None,
) -> BlockExchangeSemantics:
    """Validate Exchange choices and describe the selected overloads.

    Require a dtype and a positive per-thread item count. Scatter modes
    require a rank dtype; only flagged scatter accepts a validity dtype. Warp
    time slicing must be boolean and cannot accompany guarded or flagged
    scatter. These checks concern call structure, not runtime rank contents or
    destination uniqueness.

    Return normalized semantics that can be reused before block dimensions are
    known. No compiler types are lowered and no device code is generated.
    """

    if dtype is None:
        raise ValueError("dtype must be provided")
    mode = BlockExchangeMode(mode)
    value_form = BlockExchangeValueForm(value_form)
    items_per_thread = normalize_positive_int(
        "items_per_thread",
        items_per_thread,
    )
    if not isinstance(warp_time_slicing, bool):
        # Keep the established ValueError contract for invalid controls.
        raise ValueError("warp_time_slicing must be a boolean")  # noqa: TRY004
    if warp_time_slicing and mode in {
        BlockExchangeMode.SCATTER_TO_STRIPED_GUARDED,
        BlockExchangeMode.SCATTER_TO_STRIPED_FLAGGED,
    }:
        raise ValueError(
            "warp_time_slicing is not supported for guarded or flagged "
            "scatter-to-striped exchange"
        )
    if mode.uses_ranks and rank_dtype is None:
        raise ValueError("rank_dtype is required for scatter modes")
    if not mode.uses_ranks and rank_dtype is not None:
        raise ValueError("rank_dtype is only valid for scatter modes")
    if mode.uses_valid_flags and valid_flag_dtype is None:
        raise ValueError(
            "valid_flag_dtype is required for scatter_to_striped_flagged"
        )
    if not mode.uses_valid_flags and valid_flag_dtype is not None:
        raise ValueError(
            "valid_flag_dtype is only valid for scatter_to_striped_flagged"
        )

    methods: list[tuple[Any, ...]] = []
    if value_form in {
        BlockExchangeValueForm.IN_PLACE,
        BlockExchangeValueForm.BOTH,
    }:
        methods.append(_in_place_parameters(mode))
    if value_form in {
        BlockExchangeValueForm.OUT_OF_PLACE,
        BlockExchangeValueForm.BOTH,
    }:
        methods.append(_out_of_place_parameters(mode))

    return BlockExchangeSemantics(
        dtype=dtype,
        mode=mode,
        value_form=value_form,
        items_per_thread=items_per_thread,
        warp_time_slicing=warp_time_slicing,
        rank_dtype=rank_dtype,
        valid_flag_dtype=valid_flag_dtype,
        parameters=tuple(methods),
    )


def make_block_exchange_specialization(
    *,
    dtype: Any,
    block_dim: tuple[int, int, int],
    items_per_thread: int,
    mode: str | BlockExchangeMode,
    value_form: str
    | BlockExchangeValueForm = BlockExchangeValueForm.OUT_OF_PLACE,
    warp_time_slicing: bool = False,
    rank_dtype: Any | None = None,
    valid_flag_dtype: Any | None = None,
) -> BlockExchangeSpecialization:
    """Bind Exchange semantics to exact CUB block template arguments.

    Normalize the block shape, validate the call, and construct an
    ``Algorithm`` for its CUB method and requested array overloads. Bind the
    item dtype, block dimensions, item count, and time-slicing choice. Rank
    and flag dtypes resolve parameter dependencies when the mode uses them.

    Return the algorithm together with its call semantics and block shape. A
    backend must still translate its types and materialize the invocable.
    """

    block_dim = normalize_block_dim(block_dim)
    call = make_block_exchange_semantics(
        dtype=dtype,
        items_per_thread=items_per_thread,
        mode=mode,
        value_form=value_form,
        warp_time_slicing=warp_time_slicing,
        rank_dtype=rank_dtype,
        valid_flag_dtype=valid_flag_dtype,
    )
    template_arguments = {
        "T": dtype,
        "BLOCK_DIM_X": block_dim[0],
        "ITEMS_PER_THREAD": items_per_thread,
        "WARP_TIME_SLICING": int(warp_time_slicing),
        "BLOCK_DIM_Y": block_dim[1],
        "BLOCK_DIM_Z": block_dim[2],
    }
    if call.uses_ranks:
        template_arguments["OffsetT"] = rank_dtype
    if call.uses_valid_flags:
        template_arguments["ValidFlag"] = valid_flag_dtype

    specialization = Algorithm(
        struct_name="BlockExchange",
        method_name=call.method_name,
        c_name="block_exchange",
        includes=("cub/block/block_exchange.cuh",),
        template_parameters=_TEMPLATE_PARAMETERS,
        parameters=call.parameters,
        template_arguments=template_arguments,
        metadata={
            "scope": "block",
            "primitive": "exchange",
            "mode": call.mode.value,
            "value_form": call.value_form.value,
            "warp_time_slicing": call.warp_time_slicing,
            "ranks": call.uses_ranks,
            "valid_flags": call.uses_valid_flags,
        },
    )
    return BlockExchangeSpecialization(
        specialization=specialization,
        call=call,
        block_dim=block_dim,
    )


__all__ = [
    "BlockExchangeMode",
    "BlockExchangeSemantics",
    "BlockExchangeSpecialization",
    "BlockExchangeValueForm",
    "make_block_exchange_semantics",
    "make_block_exchange_specialization",
]
