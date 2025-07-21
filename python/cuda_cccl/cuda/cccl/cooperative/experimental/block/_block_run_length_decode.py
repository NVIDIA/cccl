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
    TempStoragePointer,
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
        decoded_items_dtype: DtypeType,
        decoded_window_offset_dtype: DtypeType,
        relative_offsets_dtype: DtypeType = None,
    ) -> None:
        self.parent = parent
        self.decoded_items_dtype = decoded_items_dtype
        self.decoded_window_offset_dtype = decoded_window_offset_dtype
        self.relative_offsets_dtype = relative_offsets_dtype

        # self.template_parameters = list(parent.template_parameters)
        self.template_parameters = [
            TemplateParameter("ItemT"),
            TemplateParameter("DECODED_ITEMS_PER_THREAD"),
            TemplateParameter("DecodedOffsetT"),
        ]

        self.specialization_kwds = {
            "ItemT": parent.item_dtype,
            "DecodedOffsetT": decoded_window_offset_dtype,
            "DECODED_ITEMS_PER_THREAD": parent.decoded_items_per_thread,
        }

        method = [
            DependentArray(
                Dependency("ItemT"),
                Dependency("DECODED_ITEMS_PER_THREAD"),
                name="decoded_items",
            ),
        ]

        if relative_offsets_dtype is not None:
            self.template_parameters.append(
                TemplateParameter("RelativeOffsetT"),
            )
            self.specialization_kwds["RelativeOffsetT"] = relative_offsets_dtype
            method.append(
                DependentArray(
                    Dependency("RelativeOffsetT"),
                    Dependency("DECODED_ITEMS_PER_THREAD"),
                    name="item_offsets",
                )
            )

        method.append(
            DependentReference(
                Dependency("DecodedOffsetT"),
                name="from_decoded_offset",
            )
        )

        self.parameters = [method]

        self.algorithm = Algorithm(
            struct_name=self.struct_name,
            method_name=self.method_name,
            c_name=self.c_name,
            includes=[],
            template_parameters=self.template_parameters,
            parameters=self.parameters,
            primitive=self,
        )

        self.specialization = self.algorithm.specialize(self.specialization_kwds)


class BlockRunLength(BasePrimitive):
    is_parent = True

    struct_name = "BlockRunLengthDecode"
    method_name = "BlockRunLengthDecode"
    c_name = "BlockRunLengthDecode"
    includes = [
        "cub/block/block_run_length_decode.cuh",
    ]

    def _validate_items_per_thread(self, items_per_thread: int, name: str) -> None:
        try:
            items_per_thread = int(items_per_thread)
        except ValueError:
            raise ValueError(f"{name} must be an integer; got: {items_per_thread}")
        if items_per_thread <= 0:
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
        run_values=None,
        run_lengths=None,
        total_decoded_size=None,
        unique_id=None,
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

        template_parameters = [
            TemplateParameter("ItemT"),
            TemplateParameter("BLOCK_DIM_X"),
            TemplateParameter("RUNS_PER_THREAD"),
            TemplateParameter("DECODED_ITEMS_PER_THREAD"),
            TemplateParameter("DecodedOffsetT"),
            TemplateParameter("BLOCK_DIM_Y"),
            TemplateParameter("BLOCK_DIM_Z"),
        ]

        specialization_kwds = {
            "ItemT": self.item_dtype,
            "BLOCK_DIM_X": self.dim[0],
            "RUNS_PER_THREAD": self.runs_per_thread,
            "DECODED_ITEMS_PER_THREAD": self.decoded_items_per_thread,
            "DecodedOffsetT": self.decoded_offset_dtype,
            "BLOCK_DIM_Y": self.dim[1],
            "BLOCK_DIM_Z": self.dim[2],
        }

        method = []

        if run_values is not None:
            method.append(
                DependentArray(
                    Dependency("ItemT"),
                    Dependency("RUNS_PER_THREAD"),
                    name="run_values",
                )
            )

        if run_lengths is not None:
            method.append(
                DependentArray(
                    Dependency("RunLengthT"),
                    Dependency("RUNS_PER_THREAD"),
                    name="run_lengths",
                )
            )
            specialization_kwds["RunLengthT"] = run_lengths.dtype

        if total_decoded_size is not None:
            # template_parameters.append(
            #    TemplateParameter("TotalDecodedSizeT"),
            # )
            specialization_kwds["TotalDecodedSizeT"] = total_decoded_size
            method.append(
                DependentReference(
                    Dependency("TotalDecodedSizeT"),
                    name="total_decoded_size",
                )
            )

        if temp_storage is not None:
            (method.insert(0, TempStoragePointer()),)

        if method:
            self.parameters = [
                method,
            ]
        else:
            self.parameters = []

        self.template_parameters = template_parameters

        self.algorithm = Algorithm(
            self.struct_name,
            self.method_name,
            self.c_name,
            self.includes,
            self.template_parameters,
            self.parameters,
            self,
            unique_id=unique_id,
        )

        self.specialization = self.algorithm.specialize(specialization_kwds)

        # Trigger the LTO IR generation so that the temp storage info is
        # accessible.
        # _ = self.specialization.get_lto_ir()

    def decode(
        self,
        decoded_items_dtype,
        decoded_window_offset_dtype,
        relative_offsets_dtype=None,
    ):
        return BlockRunLengthDecode(
            self,
            decoded_items_dtype,
            decoded_window_offset_dtype,
            relative_offsets_dtype=relative_offsets_dtype,
        )


class run_length(BlockRunLength):
    pass
