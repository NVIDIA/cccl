# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import IntEnum, auto
from typing import Optional

import numba

from .._common import normalize_dtype_param
from .._types import (
    Algorithm,
    BasePrimitive,
    Dependency,
    DependentArray,
    Invocable,
    TemplateParameter,
    numba_type_to_wrapper,
)


class WarpExchangeType(IntEnum):
    StripedToBlocked = auto()
    BlockedToStriped = auto()
    ScatterToStriped = auto()


class exchange(BasePrimitive):
    is_one_shot = True

    def __init__(
        self,
        dtype,
        items_per_thread: int,
        threads_in_warp: int = 32,
        warp_exchange_type: WarpExchangeType = WarpExchangeType.StripedToBlocked,
        offset_dtype: Optional[numba.types.Type] = None,
        methods: Optional[dict] = None,
        unique_id=None,
        temp_storage=None,
        node=None,
    ):
        """Create a warp-wide exchange operation."""
        if warp_exchange_type not in WarpExchangeType:
            raise ValueError(
                "warp_exchange_type must be a valid WarpExchangeType value; "
                f"got: {warp_exchange_type!r}"
            )
        if items_per_thread < 1:
            raise ValueError("items_per_thread must be >= 1")

        self.node = node
        self.temp_storage = temp_storage

        dtype = normalize_dtype_param(dtype)
        if offset_dtype is not None:
            offset_dtype = normalize_dtype_param(offset_dtype)

        method_name = {
            WarpExchangeType.StripedToBlocked: "StripedToBlocked",
            WarpExchangeType.BlockedToStriped: "BlockedToStriped",
            WarpExchangeType.ScatterToStriped: "ScatterToStriped",
        }[warp_exchange_type]

        template_parameters = [
            TemplateParameter("T"),
            TemplateParameter("ITEMS_PER_THREAD"),
            TemplateParameter("LOGICAL_WARP_THREADS"),
            TemplateParameter("WARP_EXCHANGE_ALGORITHM"),
        ]

        specialization_kwds = {
            "T": dtype,
            "ITEMS_PER_THREAD": items_per_thread,
            "LOGICAL_WARP_THREADS": threads_in_warp,
            "WARP_EXCHANGE_ALGORITHM": "::cub::WARP_EXCHANGE_SMEM",
        }

        input_items = DependentArray(
            Dependency("T"), Dependency("ITEMS_PER_THREAD"), name="input_items"
        )
        output_items = DependentArray(
            Dependency("T"), Dependency("ITEMS_PER_THREAD"), name="output_items"
        )

        method = [input_items, output_items]

        if warp_exchange_type == WarpExchangeType.ScatterToStriped:
            if offset_dtype is None:
                offset_dtype = numba.int32
            specialization_kwds["OffsetT"] = offset_dtype
            ranks = DependentArray(
                Dependency("OffsetT"), Dependency("ITEMS_PER_THREAD"), name="ranks"
            )
            method.append(ranks)
        elif offset_dtype is not None:
            raise ValueError("offset_dtype is only valid for ScatterToStriped")

        parameters = [method]

        type_definitions = None
        if methods is not None:
            type_definitions = [numba_type_to_wrapper(dtype, methods=methods)]

        template = Algorithm(
            "WarpExchange",
            method_name,
            "warp_exchange",
            ["cub/warp/warp_exchange.cuh"],
            template_parameters,
            parameters,
            self,
            type_definitions=type_definitions,
            threads=threads_in_warp,
            unique_id=unique_id,
        )

        self.algorithm = template
        self.specialization = template.specialize(specialization_kwds)

    @classmethod
    def create(
        cls,
        dtype,
        items_per_thread: int,
        threads_in_warp: int = 32,
        warp_exchange_type: WarpExchangeType = WarpExchangeType.StripedToBlocked,
        offset_dtype: Optional[numba.types.Type] = None,
        methods: Optional[dict] = None,
    ):
        algo = cls(
            dtype=dtype,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            warp_exchange_type=warp_exchange_type,
            offset_dtype=offset_dtype,
            methods=methods,
        )
        specialization = algo.specialization
        return Invocable(
            ltoir_files=specialization.get_lto_ir(threads=threads_in_warp),
            temp_storage_bytes=specialization.temp_storage_bytes,
            temp_storage_alignment=specialization.temp_storage_alignment,
            algorithm=specialization,
        )
