# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba

from .._common import (
    normalize_dim_param,
    normalize_dtype_param,
)
from .._enums import (
    BlockHistogramAlgorithm,
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


class BlockHistogramInit(BasePrimitive):
    c_name = 'init'
    method_name = 'InitHistogram'

    template_parameters = [
        TemplateParameter("CounterT"),
    ]

    parameters = [
        DependentArray(Dependency("CounterT"), Dependency("BINS")),
    ]

    def __init__(self, struct: "BlockHistogram",
                 histogram_bins,
                 temp_storage=None) -> None:
        c_name = 'init'
        method_name = 'InitHistogram'

        template_parameters = (
            struct.template_parameters +
            self.template_parameters
        )

        algorithm = Algorithm(
            struct.struct_name,
            method_name,
            c_name,
            struct.includes,
            struct.template_parameters,
            parameters,
        ).specialize({
            "T": struct.item_dtype,
            "BLOCK_DIM_X": struct.dim[0],
            "ITEMS_PER_THREAD": struct.items_per_thread,
            "ALGORITHM": str(struct.algorithm_cpp_string),
            "BLOCK_DIM_Y": self.dim[1],
            "BLOCK_DIM_Z": self.dim[2],
            "CounterT": struct.counter_dtype,
            "BINS": struct.bins,
        })
        algorithm.temp_storage = temp_storage

        return algorithm

class BlockHistogramComposite(BasePrimitive):
    c_name = 'composite'
    method_name = 'Composite'

    parameters = [
        # `T (&items)[ITEMS_PER_THREAD]`
        DependentArray(Dependency("T"), Dependency("ITEMS_PER_THREAD")),
        # `CounterT (&bins)[BINS]`
        DependentArray(Dependency("CounterT"), Dependency("BINS")),
    ]

    def __init__(self, struct: "BlockHistogram",
                 thread_samples,
                 histogram_bins,
                 temp_storage=None) -> None:
        c_name = 'init'
        method_name = 'InitHistogram'

        algorithm = Algorithm(
            struct.struct_name,
            method_name,
            c_name,
            struct.includes,
            struct.template_parameters,
            parameters,
        ).specialize({
            "T": self.item_dtype,
            "BLOCK_DIM_X": self.dim[0],
            "ITEMS_PER_THREAD": self.items_per_thread,
            "ALGORITHM": str(self.algorithm),
            "BLOCK_DIM_Y": self.dim[1],
            "BLOCK_DIM_Z": self.dim[2],
            "CounterT": self.counter_dtype,
            "BINS": self.bins,
        })
        algorithm.temp_storage = temp_storage

        return algorithm


class BlockHistogram:
    default_algorithm = BlockHistogramAlgorithm.BLOCK_HISTO_ATOMIC
    struct_name = "BlockHistogram"
    includes = [
        "cub/block/block_histogram.cuh",
    ]

    template_parameters = [
        TemplateParameter("T"),
        TemplateParameter("BLOCK_DIM_X"),
        TemplateParameter("ITEMS_PER_THREAD"),
        TemplateParameter("ALGORITHM"),
        TemplateParameter("BLOCK_DIM_Y"),
        TemplateParameter("BLOCK_DIM_Z"),
        TemplateParameter("CounterT"),
        TemplateParameter("BINS"),
    ]

    def __init__(self,
                 item_dtype: DtypeType,
                 counter_dtype: DtypeType,
                 dim: DimType,
                 items_per_thread: int,
                 algorithm: BlockHistogramAlgorithm = None,
                 bins: int = 256) -> None:
        self.item_dtype = normalize_dtype_param(item_dtype)
        self.counter_dtype = normalize_dtype_param(counter_dtype)
        self.dim = normalize_dim_param(dim)
        self.items_per_thread = items_per_thread
        self.algorithm = algorithm or self.default_algorithm
        self.bins = bins

    def init(self, histogram_bins, temp_storage=None):
        return BlockHistogramInit(
            self,
            histogram_bins,
            temp_storage=temp_storage,
        )

    def composite(self, thread_data, histogram_bins, temp_storage=None):
        return BlockHistogramComposite(
            self,
            thread_samples,
            histogram_bins,
            temp_storage=temp_storage,
        )
