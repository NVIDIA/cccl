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
    Array,
    BasePrimitive,
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


class histogram_init(BasePrimitive):
    is_child = True
    c_name = "init"
    method_name = "InitHistogram"
    struct_name = STRUCT_NAME
    includes = INCLUDES

    def __init__(
        self,
        parent: "histogram",
        node: "CoopNode",
    ) -> None:
        self.node = node
        self.parent = parent
        self.struct_name = parent.struct_name
        self.includes = parent.includes
        parent_node = self.parent_node = parent.node
        self.histogram = parent_node.histogram
        self.bins = parent_node.bins
        self.counter_dtype = parent_node.counter_dtype

        template_parameters = parent.template_parameters
        parameters = (
            (
                Array(
                    self.counter_dtype,
                    self.bins,
                    name="histogram",
                ),
            ),
        )

        self.algorithm = Algorithm(
            self.struct_name,
            self.method_name,
            self.c_name,
            self.includes,
            template_parameters,
            parameters,
            primitive=self,
            unique_id=parent_node.unique_id,
        )

        dim = parent_node.threads_per_block
        algorithm = parent_node.algorithm or parent.default_algorithm
        specialization_kwds = {
            "T": parent_node.item_dtype,
            "BLOCK_DIM_X": dim[0],
            "ITEMS_PER_THREAD": parent_node.items_per_thread,
            "BINS": parent_node.bins,
            "ALGORITHM": str(algorithm),
            "BLOCK_DIM_Y": dim[1],
            "BLOCK_DIM_Z": dim[2],
        }

        self.specialization = self.algorithm.specialize(specialization_kwds)


class histogram_composite(BasePrimitive):
    is_child = True
    c_name = "composite"
    method_name = "Composite"
    struct_name = STRUCT_NAME
    includes = INCLUDES

    c_name = "composite"
    method_name = "Composite"

    def __init__(
        self,
        parent: "histogram",
        node: "CoopNode",
        items: numba.types.Array,
    ) -> None:
        self.node = node
        self.parent = parent
        self.struct_name = parent.struct_name
        self.includes = parent.includes
        parent_node = self.parent_node = parent.node
        self.items = items
        self.items_per_thread = parent_node.items_per_thread
        self.item_dtype = parent_node.item_dtype
        self.histogram = parent_node.histogram
        self.bins = parent_node.bins
        self.counter_dtype = parent_node.counter_dtype

        template_parameters = parent.template_parameters
        parameters = (
            (
                Array(self.item_dtype, self.items_per_thread, name="items"),
                Array(
                    self.counter_dtype,
                    self.bins,
                    name="histogram",
                ),
            ),
        )

        self.algorithm = Algorithm(
            self.struct_name,
            self.method_name,
            self.c_name,
            self.includes,
            template_parameters,
            parameters,
            primitive=self,
            unique_id=parent_node.unique_id,
        )

        dim = parent_node.threads_per_block
        algorithm = parent_node.algorithm or parent.default_algorithm
        specialization_kwds = {
            "T": parent_node.item_dtype,
            "BLOCK_DIM_X": dim[0],
            "ITEMS_PER_THREAD": parent_node.items_per_thread,
            "BINS": parent_node.bins,
            "ALGORITHM": str(algorithm),
            "BLOCK_DIM_Y": dim[1],
            "BLOCK_DIM_Z": dim[2],
        }

        self.specialization = self.algorithm.specialize(specialization_kwds)


class histogram(BasePrimitive):
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

        Example:
            The snippet below demonstrates a block histogram with explicit
            init and composite calls.

            .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_histogram.py
                :language: python
                :dedent:
                :start-after: example-begin imports
                :end-before: example-end imports

            .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_histogram.py
                :language: python
                :dedent:
                :start-after: example-begin histogram
                :end-before: example-end histogram

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
                "Temporary storage is not yet supported for histogram."
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
        self.struct_name = struct_name
        self.includes = includes
        self.specialization = self.algorithm.specialize(specialization_kwds)

    def init(self, node: "CoopNode") -> histogram_init:
        return histogram_init(self, node)

    def composite(self, node: "CoopNode", items) -> histogram_composite:
        return histogram_composite(self, node, items)

    @classmethod
    def create(
        cls,
        item_dtype: DtypeType,
        counter_dtype: DtypeType,
        dim: DimType,
        items_per_thread: int,
        algorithm: BlockHistogramAlgorithm = None,
        bins: int = 256,
        temp_storage=None,
    ) -> "histogram":
        """
        Create a histogram instance with the specified parameters.
        """
        algo = cls(
            item_dtype=item_dtype,
            counter_dtype=counter_dtype,
            dim=dim,
            items_per_thread=items_per_thread,
            algorithm=algorithm,
            bins=bins,
            temp_storage=temp_storage,
        )
        return algo


def _build_histogram_spec(
    item_dtype,
    counter_dtype,
    threads_per_block=None,
    items_per_thread=1,
    **kwargs,
):
    kw = dict(kwargs)
    if threads_per_block is None:
        threads_per_block = kw.pop("dim", None)
    spec = {
        "item_dtype": item_dtype,
        "counter_dtype": counter_dtype,
        "dim": threads_per_block,
        "items_per_thread": items_per_thread,
    }
    spec.update(kw)
    return spec


def _make_histogram_two_phase(
    item_dtype,
    counter_dtype,
    threads_per_block=None,
    items_per_thread=1,
    **kwargs,
):
    spec = _build_histogram_spec(
        item_dtype=item_dtype,
        counter_dtype=counter_dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        **kwargs,
    )
    return histogram.create(**spec)
