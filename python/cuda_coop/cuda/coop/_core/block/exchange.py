# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral CUB BlockExchange semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .._algorithm import Algorithm, AlgorithmSpec
from .._symbols import semantic_token
from .._types import Array, Dependency, TemplateParameter, TempStorageParameter


class BlockExchangeMode(str, Enum):
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
        """Return the matching public CUB ``BlockExchange`` method name."""

        return _CUB_METHOD_NAMES[self]

    @classmethod
    def from_cub_method_name(cls, method_name: str) -> "BlockExchangeMode":
        """Resolve a public CUB method name to its normalized core mode."""

        try:
            return _CUB_METHOD_MODES[method_name]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"unsupported CUB BlockExchange method name: {method_name!r}"
            ) from exc


class BlockExchangeValueForm(str, Enum):
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
_CUB_METHOD_MODES = {name: mode for mode, name in _CUB_METHOD_NAMES.items()}
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
        parameters.append(Array(_VALID_FLAG_T, _ITEMS_PER_THREAD, name="valid_flags"))
    return tuple(parameters)


def _out_of_place_parameters(mode: BlockExchangeMode) -> tuple[Any, ...]:
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
        parameters.append(Array(_VALID_FLAG_T, _ITEMS_PER_THREAD, name="valid_flags"))
    return tuple(parameters)


@dataclass(frozen=True)
class BlockExchangeSemantics:
    """Dimension-independent BlockExchange call contract."""

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
class BlockExchangeSpec:
    """Fully specialized CUB BlockExchange semantics."""

    specialization: AlgorithmSpec
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
    value_form: str | BlockExchangeValueForm = BlockExchangeValueForm.OUT_OF_PLACE,
    warp_time_slicing: bool = False,
    rank_dtype: Any | None = None,
    valid_flag_dtype: Any | None = None,
) -> BlockExchangeSemantics:
    """Build the normalized dimension-independent BlockExchange contract."""

    if dtype is None:
        raise ValueError("dtype must be provided")
    mode = BlockExchangeMode(mode)
    value_form = BlockExchangeValueForm(value_form)
    if (
        not isinstance(items_per_thread, int)
        or isinstance(items_per_thread, bool)
        or items_per_thread < 1
    ):
        raise ValueError("items_per_thread must be a positive integer")
    if not isinstance(warp_time_slicing, bool):
        raise ValueError("warp_time_slicing must be a boolean")
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
        raise ValueError("valid_flag_dtype is required for scatter_to_striped_flagged")
    if not mode.uses_valid_flags and valid_flag_dtype is not None:
        raise ValueError(
            "valid_flag_dtype is only valid for scatter_to_striped_flagged"
        )

    methods: list[tuple[Any, ...]] = []
    if value_form in {BlockExchangeValueForm.IN_PLACE, BlockExchangeValueForm.BOTH}:
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


def make_block_exchange_spec(
    *,
    dtype: Any,
    block_dim: tuple[int, int, int],
    items_per_thread: int,
    mode: str | BlockExchangeMode,
    value_form: str | BlockExchangeValueForm = BlockExchangeValueForm.OUT_OF_PLACE,
    warp_time_slicing: bool = False,
    rank_dtype: Any | None = None,
    valid_flag_dtype: Any | None = None,
) -> BlockExchangeSpec:
    """Build a fully specialized CUB BlockExchange description."""

    block_dim = tuple(block_dim)
    if len(block_dim) != 3 or any(dim < 1 for dim in block_dim):
        raise ValueError("block_dim must contain three positive dimensions")
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
    ).specialize(
        template_arguments,
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
    return BlockExchangeSpec(
        specialization=specialization,
        call=call,
        block_dim=block_dim,
    )
