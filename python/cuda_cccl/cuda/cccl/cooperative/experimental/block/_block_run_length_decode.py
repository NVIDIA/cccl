# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba

from .._common import (
    normalize_dim_param,
    normalize_dtype_param,
)
from .._types import (
    Algorithm,
    BasePrimitive,
    TemplateParameter,
)
from .._typing import (
    DimType,
    DtypeType,
)


class BlockRunLength(BasePrimitive):
    struct_name = "BlockRunLengthDecode"
    method_name = "BlockRunLengthDecode"
    c_name = "BlockRunLengthDecode"
    includes = [
        "cub/block/block_run_length_decode.cuh",
    ]

    template_parameters = [
        TemplateParameter("ItemT"),
        TemplateParameter("BLOCK_DIM_X"),
        TemplateParameter("RUNS_PER_THREAD"),
        TemplateParameter("DECODED_ITEMS_PER_THREAD"),
        TemplateParameter("DecodedOffsetT"),
        TemplateParameter("BLOCK_DIM_Y"),
        TemplateParameter("BLOCK_DIM_Z"),
    ]

    # XXX temporary storage location.
    decoder_template_parameters = [
        TemplateParameter("RunLengthT"),
        TemplateParameter("TotalDecodedSizeT"),
        TemplateParameter("UserRunOffsetT"),
    ]

    def _validate_items_per_thread(self, items_per_thread: int, name: str) -> None:
        invalid = not isinstance(items_per_thread, int) or items_per_thread <= 0
        if invalid:
            raise ValueError(
                f"{name} must be a positive integer; got: {items_per_thread}"
            )

        return items_per_thread

    def __init__(
        self,
        item_dtype: DtypeType,
        dim: DimType,
        runs_per_thread: int,
        decoded_items_per_thread: int,
        decoded_offset_dtype: DtypeType = None,
    ) -> None:
        self.item_dtype = normalize_dtype_param(item_dtype)
        self.dim = normalize_dim_param(dim)

        self.runs_per_thread = self._validate_items_per_thread(
            runs_per_thread, "runs_per_thread"
        )

        self.decoded_items_per_thread = self._validate_items_per_thread(
            decoded_items_per_thread, "decoded_items_per_thread"
        )

        if decoded_offset_dtype is None:
            decoded_offset_dtype = numba.uint32
        self.decoded_offset_dtype = normalize_dtype_param(decoded_offset_dtype)

        self.parameters = []

        self.algorithm = Algorithm(
            self.struct_name,
            self.method_name,
            self.c_name,
            self.includes,
            self.template_parameters,
            self.parameters,
        )

        specialization_kwds = {
            "ItemT": self.item_dtype,
            "BLOCK_DIM_X": self.dim[0],
            "RUNS_PER_THREAD": self.runs_per_thread,
            "DECODED_ITEMS_PER_THREAD": self.decoded_items_per_thread,
            "DecodedOffsetT": self.decoded_offset_dtype,
            "BLOCK_DIM_Y": self.dim[1],
            "BLOCK_DIM_Z": self.dim[2],
        }
        self.specialization = self.algorithm.specialize(specialization_kwds)

        # Trigger the LTO IR generation so that the temp storage info is
        # accessible.
        _ = self.specialization.get_lto_ir()

    def decode(
        self,
        struct: "BlockRunLength",
        decoded_items,
        from_decoded_offset=None,
        item_offsets=None,
    ):
        pass


class run_length(BlockRunLength):
    pass
