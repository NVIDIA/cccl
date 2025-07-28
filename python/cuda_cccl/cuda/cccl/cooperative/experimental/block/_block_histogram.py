# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import TYPE_CHECKING, Callable

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
    Invocable,
    TemplateParameter,
)
from .._typing import (
    DimType,
    DtypeType,
)

STRUCT_NAME = "BlockHistogram"
INCLUDES = ("cub/block/block_histogram.cuh",)

if TYPE_CHECKING:
    from ._rewrite import CoopNode


class BlockHistogramInit(BasePrimitive):
    is_child = True
    c_name = "init"
    method_name = "InitHistogram"
    struct_name = STRUCT_NAME
    includes = INCLUDES

    template_parameters = (
        TemplateParameter("CounterT"),
        TemplateParameter("BINS"),
    )

    parameters = (
        (
            DependentArray(
                Dependency("CounterT"),
                Dependency("BINS"),
                name="histogram",
            ),
        ),
    )

    def __init__(
        self,
        parent: "BlockHistogram",
        node: "CoopNode",
    ) -> None:
        self.node = node
        self.parent = parent
        parent_node = self.parent_node = parent.node
        self.histogram = parent_node.histogram
        self.bins = parent_node.bins
        self.counter_dtype = parent_node.counter_dtype

        self.algorithm = Algorithm(
            self.struct_name,
            self.method_name,
            self.c_name,
            self.includes,
            self.template_parameters,
            self.parameters,
            primitive=self,
            unique_id=parent_node.unique_id,
        )

        specialization_kwds = {
            "CounterT": self.counter_dtype,
            "BINS": self.bins,
        }

        self.specialization = self.algorithm.specialize(specialization_kwds)


class BlockHistogramComposite(BasePrimitive):
    is_child = True
    c_name = "composite"
    method_name = "Composite"
    struct_name = STRUCT_NAME
    includes = INCLUDES

    c_name = "composite"
    method_name = "Composite"

    template_parameters = (
        TemplateParameter("T"),
        TemplateParameter("CounterT"),
        TemplateParameter("BINS"),
        TemplateParameter("ITEMS_PER_THREAD"),
    )

    parameters = (
        (
            # `T (&items)[ITEMS_PER_THREAD]`
            DependentArray(
                value_dtype=Dependency("T"),
                size=Dependency("ITEMS_PER_THREAD"),
                name="items",
            ),
            # `CounterT (&histogram)[BINS]`
            DependentArray(
                value_dtype=Dependency("CounterT"),
                size=Dependency("BINS"),
                name="histogram",
            ),
        ),
    )

    def __init__(
        self,
        parent: "BlockHistogram",
        node: "CoopNode",
        items: numba.types.Array,
    ) -> None:
        self.node = node
        self.parent = parent
        parent_node = self.parent_node = parent.node
        self.items = items
        self.items_per_thread = parent_node.items_per_thread
        self.item_dtype = parent_node.item_dtype
        self.histogram = parent_node.histogram
        self.bins = parent_node.bins
        self.counter_dtype = parent_node.counter_dtype

        self.algorithm = Algorithm(
            self.struct_name,
            self.method_name,
            self.c_name,
            self.includes,
            self.template_parameters,
            self.parameters,
            primitive=self,
            unique_id=parent_node.unique_id,
        )

        specialization_kwds = {
            "T": self.item_dtype,
            "ITEMS_PER_THREAD": self.items_per_thread,
            "CounterT": self.counter_dtype,
            "BINS": self.bins,
        }

        self.specialization = self.algorithm.specialize(specialization_kwds)


class BlockHistogram(BasePrimitive):
    is_parent = True
    c_name = "BlockHistogram"
    method_name = "BlockHistogram"
    default_algorithm = BlockHistogramAlgorithm.ATOMIC
    struct_name = STRUCT_NAME
    includes = INCLUDES

    template_parameters = (
        TemplateParameter("T"),
        TemplateParameter("BLOCK_DIM_X"),
        TemplateParameter("ITEMS_PER_THREAD"),
        TemplateParameter("BINS"),
        TemplateParameter("ALGORITHM"),
        TemplateParameter("BLOCK_DIM_Y"),
        TemplateParameter("BLOCK_DIM_Z"),
    )

    def __init__(
        self,
        item_dtype: DtypeType,
        counter_dtype: DtypeType,
        dim: DimType,
        items_per_thread: int,
        bins: int,
        algorithm: BlockHistogramAlgorithm = None,
        unique_id=None,
        node: "CoopNode" = None,
        temp_storage=None,
    ) -> Callable:
        """
        Creates a block-wide histogram primitive based on the CUB library's
        BlockHistogram functionality.

        :param item_dtype: Supplies the data type of the items being sampled.
            Note that this is almost always an 8-bit integer type, e.g.
            ``np.uint8``.

        :param counter_dtype: Supplies the data type of the histogram counters.

        :param dim: Supplies the dimensions of the CUDA thread block.

        :param items_per_thread: Supplies the number of items each thread will
            sample.  A good default to start with is usually 4.

        :param bins: Supplies the number of bins in the histogram.  This should
            represent the number of unique values that can be sampled by the
            ``item_dtype``, i.e. if ``item_dtype`` is ``np.uint8``, then ``bins``
            should be 256.

        :param algorithm: Supplies the algorithm to use for the histogram.
            Defaults to ``BlockHistogramAlgorithm.ATOMIC``, which uses atomic
            operations to build the histogram.

        :param unique_id: Optionally supplies a unique identifier for the
            primitive that can be used to create unique names in translation
            units.

        :param temp_storage: Optionally supplies temporary storage.  Not
            yet implemented.  Will raise ``NotImplementedError`` if a non-None
            value is provided.

        :raises ValueError: If ``items_per_thread`` is less than 1.

        :raises ValueError: If ``algorithm`` is not one of the
            :py:class:`BlockHistogramAlgorithm` enum values.

        :raises ValueError: If ``bins`` is not a positive integer.
        """

        if items_per_thread < 1:
            raise ValueError("items_per_thread must be greater than or equal to 1")

        if algorithm is None:
            algorithm = BlockHistogramAlgorithm.ATOMIC

        if algorithm not in BlockHistogramAlgorithm:
            raise ValueError(
                f"Invalid algorithm: {algorithm!r}. "
                f"Must be one of {list(BlockHistogramAlgorithm)}."
            )

        if not isinstance(bins, int) or bins <= 0:
            raise ValueError(f"Invalid bins: {bins!r}. Must be a positive integer.")

        if temp_storage is not None:
            raise NotImplementedError(
                "Temporary storage is not yet supported for BlockRunLengthDecode."
            )

        # Validation complete, continue with algorithm creation.
        self.node = node
        dim = normalize_dim_param(dim)
        item_dtype = normalize_dtype_param(item_dtype)
        counter_dtype = normalize_dtype_param(counter_dtype)

        parameters = []

        specialization_kwds = {
            "T": item_dtype,
            "BLOCK_DIM_X": dim[0],
            "ITEMS_PER_THREAD": items_per_thread,
            "BINS": bins,
            "ALGORITHM": str(algorithm),
            "BLOCK_DIM_Y": dim[1],
            "BLOCK_DIM_Z": dim[2],
            "CounterT": counter_dtype,
        }

        struct_name = "BlockHistogram"
        method_name = "Histogram"
        c_name = "block_histogram"
        includes = ["cub/block/block_histogram.cuh"]

        self.algorithm = Algorithm(
            struct_name,
            method_name,
            c_name,
            includes,
            self.template_parameters,
            parameters,
            primitive=self,
            unique_id=unique_id,
        )

        # Save attributes our children will need.
        self.item_dtype = item_dtype
        self.counter_dtype = counter_dtype
        self.items_per_thread = items_per_thread
        self.bins = bins

        self.specialization = self.algorithm.specialize(specialization_kwds)

    def init(self, node: "CoopNode") -> BlockHistogramInit:
        return BlockHistogramInit(self, node)

    def composite(self, node: "CoopNode", items) -> BlockHistogramComposite:
        return BlockHistogramComposite(self, node, items)

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
        raise NotImplementedError
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
