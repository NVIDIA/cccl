# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


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
    Invocable,
    TemplateParameter,
)
from .._typing import (
    DimType,
    DtypeType,
)


class BlockHistogramInit(BasePrimitive):
    c_name = "init"
    method_name = "InitHistogram"

    template_parameters = [
        TemplateParameter("CounterT"),
    ]

    parameters = [
        DependentArray(Dependency("CounterT"), Dependency("BINS")),
    ]

    def __init__(
        self, struct: "BlockHistogram", histogram_bins, temp_storage=None
    ) -> None:
        c_name = "init"
        method_name = "InitHistogram"

        template_parameters = struct.template_parameters + self.template_parameters

        algorithm = Algorithm(
            struct.struct_name,
            method_name,
            c_name,
            struct.includes,
            template_parameters,
            self.parameters,
        ).specialize(
            {
                "T": struct.item_dtype,
                "BLOCK_DIM_X": struct.dim[0],
                "ITEMS_PER_THREAD": struct.items_per_thread,
                "ALGORITHM": str(struct.algorithm_cpp_string),
                "BLOCK_DIM_Y": self.dim[1],
                "BLOCK_DIM_Z": self.dim[2],
                "CounterT": struct.counter_dtype,
                "BINS": struct.bins,
            }
        )
        algorithm.temp_storage = temp_storage

        return algorithm


class BlockHistogramComposite(BasePrimitive):
    c_name = "composite"
    method_name = "Composite"

    template_parameters = [
        TemplateParameter("CounterT"),
    ]

    parameters = [
        # `T (&items)[ITEMS_PER_THREAD]`
        DependentArray(Dependency("T"), Dependency("ITEMS_PER_THREAD")),
        # `CounterT (&bins)[BINS]`
        DependentArray(Dependency("CounterT"), Dependency("BINS")),
    ]

    def __init__(
        self,
        struct: "BlockHistogram",
        thread_samples,
        histogram_bins,
        temp_storage=None,
    ) -> None:
        c_name = "init"
        method_name = "InitHistogram"

        template_parameters = struct.template_parameters + self.template_parameters

        algorithm = Algorithm(
            struct.struct_name,
            method_name,
            c_name,
            struct.includes,
            template_parameters,
            self.parameters,
        ).specialize(
            {
                "T": self.item_dtype,
                "BLOCK_DIM_X": self.dim[0],
                "ITEMS_PER_THREAD": self.items_per_thread,
                "ALGORITHM": str(self.algorithm),
                "BLOCK_DIM_Y": self.dim[1],
                "BLOCK_DIM_Z": self.dim[2],
                "CounterT": self.counter_dtype,
                "BINS": self.bins,
            }
        )
        algorithm.temp_storage = temp_storage

        return algorithm


class BlockHistogram(BasePrimitive):
    default_algorithm = BlockHistogramAlgorithm.ATOMIC
    struct_name = "BlockHistogram"
    method_name = "BlockHistogram"
    c_name = "BlockHistogram"
    includes = [
        "cub/block/block_histogram.cuh",
    ]

    template_parameters = [
        TemplateParameter("T"),
        TemplateParameter("BLOCK_DIM_X"),
        TemplateParameter("ITEMS_PER_THREAD"),
        TemplateParameter("BINS"),
        TemplateParameter("ALGORITHM"),
        TemplateParameter("BLOCK_DIM_Y"),
        TemplateParameter("BLOCK_DIM_Z"),
    ]

    def __init__(
        self,
        item_dtype: DtypeType,
        # counter_dtype: DtypeType,
        dim: DimType,
        items_per_thread: int,
        algorithm: BlockHistogramAlgorithm = None,
        bins: int = 256,
        temp_storage=None,
    ) -> None:
        self.item_dtype = normalize_dtype_param(item_dtype)
        # self.counter_dtype = normalize_dtype_param(counter_dtype)
        self.dim = normalize_dim_param(dim)
        self.items_per_thread = items_per_thread
        self.bins = bins
        self.temp_storage = temp_storage

        (algorithm_cub, algorithm_enum) = self.resolve_cub_algorithm(
            algorithm,
        )

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
            "T": self.item_dtype,
            "BLOCK_DIM_X": self.dim[0],
            "ITEMS_PER_THREAD": items_per_thread,
            "BINS": bins,
            "ALGORITHM": algorithm_cub,
            "BLOCK_DIM_Y": self.dim[1],
            "BLOCK_DIM_Z": self.dim[2],
        }
        self.specialization = self.algorithm.specialize(specialization_kwds)

        # Unlike primitives such as load/store/scan--which are effectively
        # "one-shot" stateless operations--the BlockHistogram struct needs to
        # have persistent state, such that the subsequent `init()` and
        # `composite()` methods can be called (with the latter being called
        # possibly many times).
        #
        # We can still safely trigger LTO IR generation at this point.  As we
        # have not furnished any parameters, it'll expand to something along
        # the lines of:
        #
        #   #include <cuda/std/cstdint>
        #   #include <cub/block/block_histogram.cuh>
        #   using algorithm_t = cub::BlockHistogram<
        #       ::cuda::std::uint8_t, 128, 4, 256,
        #       ::cub::BLOCK_HISTO_SORT, 1, 1>;
        #   using temp_storage_t = typename algorithm_t::TempStorage;
        #
        # Which is all we need at this stage in order for our temp storage
        # bytes and alignment properties to function correctly.
        _ = self.specialization.get_lto_ir()

    def init(self, histogram_bins, temp_storage=None):
        return BlockHistogramInit(
            self,
            histogram_bins,
            temp_storage=temp_storage,
        )

    def composite(self, thread_data, histogram_bins, temp_storage=None):
        return BlockHistogramComposite(
            self,
            thread_data,
            histogram_bins,
            temp_storage=temp_storage,
        )

    @classmethod
    def create(
        cls,
        item_dtype: DtypeType,
        dim: DimType,
        items_per_thread: int,
        algorithm: BlockHistogramAlgorithm = None,
        bins: int = 256,
        temp_storage=None,
    ) -> "BlockHistogram":
        """
        Create a BlockHistogram instance with the specified parameters.
        """
        algo = cls(
            item_dtype=item_dtype,
            dim=dim,
            items_per_thread=items_per_thread,
            algorithm=algorithm,
            bins=bins,
            temp_storage=temp_storage,
        )
        specialization = algo.specialization

        ltoir_files = specialization.get_lto_ir()
        temp_storage_bytes = specialization.temp_storage_bytes
        temp_storage_alignment = specialization.temp_storage_alignment
        algorithm = specialization

        return Invocable(
            ltoir_files=ltoir_files,
            temp_storage_bytes=temp_storage_bytes,
            temp_storage_alignment=temp_storage_alignment,
            algorithm=algorithm,
        )


class histogram(BlockHistogram):
    pass
