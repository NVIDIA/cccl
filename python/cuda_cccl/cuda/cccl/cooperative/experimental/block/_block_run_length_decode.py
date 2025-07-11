# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba

from .._common import (
    normalize_dim_param,
    normalize_dtype_param,
)
from .._enums import (
    BlockLoadAlgorithm,
    BlockStoreAlgorithm,
)
from .._types import (
    Algorithm,
    BasePrimitive,
    Dependency,
    DependentArray,
    DependentPointer,
    Invocable,
    Pointer,
    TemplateParameter,
)
from .._typing import (
    DimType,
    DtypeType,
)

class BlockRunLengthDecode(BasePrimitive):
    template_parameters = [
        TemplateParameter("ItemT"),
        TemplateParameter("BLOCK_DIM_X"),
        TemplateParameter("RUNS_PER_THREAD"),
        TemplateParameter("DECODED_ITEMS_PER_THREAD"),
        TemplateParameter("DecodedOffsetT"),
        TemplateParameter("BLOCK_DIM_Y"),
        TemplateParameter("BLOCK_DIM_Z"),

        TemplateParameter("RunLengthT"),
        TemplateParameter("TotalDecodedSizeT"),

        TemplateParameter("UserRunOffsetT"),

        TemplateParameter("RelativeOffsetsT"),
    ]

    def __init__(self,
                 dtype: DtypeType,
                 dim: DimType,
                 runs_per_thread: int,
                 decoded_items_per_thread: int,
                 decoded_offset_type: DtypeType = None,
                 run_length_type: DtypeType = None,
                 total_decoded_size_type: DtypeType = None,
                 user_run_offset_type: DtypeType = None,
                 relative_offsets_type: DtypeType = None) -> None:

        self.dtype = normalize_dtype_param(dtype)
        self.dim = normalize_dim_param(dim)
        self.runs_per_thread = runs_per_thread


    def __call__(self, run_values, run_lengths, total_decoded_size):
        """
        Create a block run length decoder.

        Parameters
        ----------
        run_values : Pointer to the run values.
        run_lengths : Pointer to the run lengths.
        total_decoded_size : Total size of the decoded items.
        """
        return BlockRunLengthDecodeInvoker(
            self,
            run_values,
            run_lengths,
            total_decoded_size,
        )

def kernel1(run_values,
            run_lengths,
            run_items,
            runs_per_thread,
            decoded_items_per_thread):

    total_decoded_size = 0

    block_rld = cuda.block.run_length_decode(
        run_values,
        run_lengths,
        total_decoded_size,
    )

    stride = cuda.blockDim.x * decoded_items_per_thread
    decoded_window_offset = 0
    while decoded_window_offset < total_decoded_size:
        relative_offsets = cuda.local.array(
            decoded_items_per_thread,
            dtype=run_lengths.dtype,
        )

        decoded_items = cuda.local.array(
            decoded_items_per_thread,
            dtype=run_items.dtype,
        )

        num_valid_items = total_decoded_size - decoded_window_offset

        block_rld.decode(
            decoded_items,
            relative_offsets,
            decoded_window_offset,
        )

        decoded_window_offset += stride

