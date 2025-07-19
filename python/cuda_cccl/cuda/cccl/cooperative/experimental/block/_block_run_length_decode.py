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
    Dependency,
    DependentArray,
    DependentReference,
    TemplateParameter,
)
from .._typing import (
    DimType,
    DtypeType,
)


class BlockRunLengthDecode(BasePrimitive):
    is_child = True

    c_name = "decode"
    method_name = "RunLengthDecode"
    struct_name = "BlockRunLengthDecode"
    includes = [
        "cub/block/block_run_length_decode.cuh",
    ]

    def __init__(
        self,
        parent: "BlockRunLength",
        decoded_items,
        decoded_window_offset,
        relative_offsets=None,
    ) -> None:
        self.parent = parent
        self.decoded_items = decoded_items
        self.decoded_window_offset = decoded_window_offset
        self.relative_offsets = relative_offsets

        # self.template_parameters = list(parent.template_parameters)
        self.template_parameters = []

        self.specialization_kwds = {
            "RunLengthT": parent.item_dtype,
            "TotalDecodedSizeT": parent.decoded_offset_dtype,
        }

        has_relative_offsets = relative_offsets is not None
        if has_relative_offsets:
            self.template_parameters += [
                TemplateParameter("RelativeOffsetsT"),
            ]

        self.parameters = [
            # RunLengthDecode(
            #   ItemT (&decoded_items)[DECODED_ITEMS_PER_THREAD],
            #   DecodedOffsetT decoded_window_offset,
            #   ...
            # ItemT (&decoded_items)[DECODED_ITEMS_PER_THREAD],
            DependentArray(Dependency("ItemT"), Dependency("DECODED_ITEMS_PER_THREAD")),
            DependentReference(Dependency("DecodedOffsetT")),
        ]

        if has_relative_offsets:
            self.template_parameters.append(
                TemplateParameter("RelativeOffsetsT"),
            )

            # RelativeOffsetsT (&relative_offsets)[DECODED_ITEMS_PER_THREAD],
            self.parameters.append(
                DependentArray(
                    Dependency("RelativeOffsetsT"),
                    Dependency("DECODED_ITEMS_PER_THREAD"),
                )
            )

            self.specialization_kwds["RelativeOffsetsT"] = relative_offsets.dtype

        self.algorithm = Algorithm(
            struct_name=self.struct_name,
            method_name=self.method_name,
            c_name=self.c_name,
            includes=[],
            template_parameters=self.template_parameters,
            parameters=self.parameters,
            primitive=self.primitive,
        )

        self.specialization = self.algorithm.specialize(**self.specialization_kwds)


class BlockRunLength(BasePrimitive):
    is_parent = True

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
        total_decoded_size=None,
        temp_storage=None,
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
        decoded_items,
        decoded_window_offset,
        relative_offsets=None,
    ):
        return BlockRunLengthDecode(
            self,
            decoded_items,
            decoded_window_offset,
            relative_offsets=relative_offsets,
        )


class run_length(BlockRunLength):
    pass
