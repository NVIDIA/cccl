# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral CUB physical and logical WarpExchange semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .._algorithm import Algorithm, AlgorithmSpec
from .._types import Array, Dependency, TemplateParameter, TempStorageParameter
from ..block._common import normalize_positive_int
from ..block.exchange import BlockExchangeSemantics, make_block_exchange_semantics

_SUPPORTED_LOGICAL_WARP_THREADS = frozenset({1, 2, 4, 8, 16, 32})


class WarpExchangeMode(str, Enum):
    STRIPED_TO_BLOCKED = "striped_to_blocked"
    BLOCKED_TO_STRIPED = "blocked_to_striped"
    SCATTER_TO_STRIPED = "scatter_to_striped"


class WarpExchangeValueForm(str, Enum):
    IN_PLACE = "in_place"
    OUT_OF_PLACE = "out_of_place"
    BOTH = "both"


_METHOD_NAMES = {
    WarpExchangeMode.STRIPED_TO_BLOCKED: "StripedToBlocked",
    WarpExchangeMode.BLOCKED_TO_STRIPED: "BlockedToStriped",
    WarpExchangeMode.SCATTER_TO_STRIPED: "ScatterToStriped",
}
_T = Dependency("T")
_ITEMS_PER_THREAD = Dependency("ITEMS_PER_THREAD")
_OFFSET_T = Dependency("OffsetT")


@dataclass(frozen=True)
class WarpExchangeSpec:
    """Fully specialized CUB WarpExchange semantics."""

    specialization: AlgorithmSpec
    call: BlockExchangeSemantics
    mode: WarpExchangeMode
    value_form: WarpExchangeValueForm
    items_per_thread: int
    threads_in_warp: int
    rank_dtype: Any | None

    @property
    def method_name(self) -> str:
        return self.specialization.method_name

    @property
    def uses_ranks(self) -> bool:
        return self.rank_dtype is not None

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.specialization.semantic_key


def _normalize_logical_warp_threads(threads_in_warp: Any) -> int:
    if (
        not isinstance(threads_in_warp, int)
        or isinstance(threads_in_warp, bool)
        or threads_in_warp not in _SUPPORTED_LOGICAL_WARP_THREADS
    ):
        supported = ", ".join(
            str(width) for width in sorted(_SUPPORTED_LOGICAL_WARP_THREADS)
        )
        raise ValueError(
            "WarpExchange requires threads_in_warp in "
            f"{{{supported}}}; got {threads_in_warp!r}"
        )
    return threads_in_warp


def _in_place_parameters() -> tuple[Any, ...]:
    return (
        TempStorageParameter(),
        Array(
            _T,
            _ITEMS_PER_THREAD,
            name="items",
            is_inout=True,
            is_return=False,
        ),
        Array(_OFFSET_T, _ITEMS_PER_THREAD, name="ranks"),
    )


def _out_of_place_parameters(*, uses_ranks: bool) -> tuple[Any, ...]:
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
    if uses_ranks:
        parameters.append(Array(_OFFSET_T, _ITEMS_PER_THREAD, name="ranks"))
    return tuple(parameters)


def make_warp_exchange_spec(
    *,
    dtype: Any,
    items_per_thread: int,
    threads_in_warp: int,
    mode: str | WarpExchangeMode,
    value_form: str | WarpExchangeValueForm = WarpExchangeValueForm.OUT_OF_PLACE,
    rank_dtype: Any | None = None,
) -> WarpExchangeSpec:
    """Build canonical SMEM-backed WarpExchange semantics."""

    if dtype is None:
        raise ValueError("dtype must be provided")
    mode = WarpExchangeMode(mode)
    value_form = WarpExchangeValueForm(value_form)
    items_per_thread = normalize_positive_int(
        "items_per_thread",
        items_per_thread,
    )
    threads_in_warp = _normalize_logical_warp_threads(threads_in_warp)
    uses_ranks = mode is WarpExchangeMode.SCATTER_TO_STRIPED
    if uses_ranks and rank_dtype is None:
        raise ValueError("rank_dtype is required for scatter_to_striped")
    if not uses_ranks and rank_dtype is not None:
        raise ValueError("rank_dtype is only valid for scatter_to_striped")
    if not uses_ranks and value_form is not WarpExchangeValueForm.OUT_OF_PLACE:
        raise ValueError("in-place overloads are only valid for scatter_to_striped")
    call = make_block_exchange_semantics(
        dtype=dtype,
        items_per_thread=items_per_thread,
        mode=mode.value,
        value_form=value_form.value,
        rank_dtype=rank_dtype,
    )

    methods: list[tuple[Any, ...]] = []
    if value_form in {WarpExchangeValueForm.IN_PLACE, WarpExchangeValueForm.BOTH}:
        methods.append(_in_place_parameters())
    if value_form in {
        WarpExchangeValueForm.OUT_OF_PLACE,
        WarpExchangeValueForm.BOTH,
    }:
        methods.append(_out_of_place_parameters(uses_ranks=uses_ranks))

    template_arguments = {
        "T": dtype,
        "ITEMS_PER_THREAD": items_per_thread,
        "LOGICAL_WARP_THREADS": threads_in_warp,
        "WARP_EXCHANGE_ALGORITHM": "::cub::WARP_EXCHANGE_SMEM",
    }
    if uses_ranks:
        template_arguments["OffsetT"] = rank_dtype

    specialization = Algorithm(
        struct_name="WarpExchange",
        method_name=_METHOD_NAMES[mode],
        c_name="warp_exchange",
        includes=("cub/warp/warp_exchange.cuh",),
        template_parameters=(
            TemplateParameter("T"),
            TemplateParameter("ITEMS_PER_THREAD"),
            TemplateParameter("LOGICAL_WARP_THREADS"),
            TemplateParameter("WARP_EXCHANGE_ALGORITHM"),
        ),
        parameters=tuple(methods),
    ).specialize(
        template_arguments,
        metadata={
            "scope": "warp",
            "primitive": "exchange",
            "mode": mode.value,
            "value_form": value_form.value,
            "ranks": uses_ranks,
        },
    )
    return WarpExchangeSpec(
        specialization=specialization,
        call=call,
        mode=mode,
        value_form=value_form,
        items_per_thread=items_per_thread,
        threads_in_warp=threads_in_warp,
        rank_dtype=rank_dtype,
    )


__all__ = [
    "WarpExchangeMode",
    "WarpExchangeSpec",
    "WarpExchangeValueForm",
    "make_warp_exchange_spec",
]
