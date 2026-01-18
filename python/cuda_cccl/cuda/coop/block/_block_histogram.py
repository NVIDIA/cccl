# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from types import SimpleNamespace
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


ATOMIC_WRAPPER_CODE = """
namespace cub {
template <typename T,
          int BLOCK_DIM_X,
          int ITEMS_PER_THREAD,
          int BINS,
          ::cub::BlockHistogramAlgorithm ALGORITHM,
          int BLOCK_DIM_Y,
          int BLOCK_DIM_Z>
struct BlockHistogramAtomicWrapper
{
  static constexpr int BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;

  struct TempStorage
  {};

  template <typename U, int Size = sizeof(U)>
  struct IndexLoader;

  template <typename U>
  struct IndexLoader<U, 1>
  {
    __device__ __forceinline__ static unsigned int load(const U* ptr)
    {
      unsigned int idx;
      asm volatile("ld.u8 %0, [%1];" : "=r"(idx) : "l"(ptr));
      return idx & 0xFFu;
    }
  };

  template <typename U>
  struct IndexLoader<U, 2>
  {
    __device__ __forceinline__ static unsigned int load(const U* ptr)
    {
      unsigned int idx;
      asm volatile("ld.u16 %0, [%1];" : "=r"(idx) : "l"(ptr));
      return idx & 0xFFFFu;
    }
  };

  template <typename U>
  struct IndexLoader<U, 4>
  {
    __device__ __forceinline__ static unsigned int load(const U* ptr)
    {
      unsigned int idx;
      asm volatile("ld.u32 %0, [%1];" : "=r"(idx) : "l"(ptr));
      return idx;
    }
  };

  __device__ __forceinline__ BlockHistogramAtomicWrapper() {}

  template <typename CounterT>
  __device__ __forceinline__ void InitHistogram(CounterT (&histogram)[BINS])
  {
    unsigned int linear_tid =
      (threadIdx.z * BLOCK_DIM_Y + threadIdx.y) * BLOCK_DIM_X + threadIdx.x;

    int histo_offset = 0;
    #pragma unroll
    for (; histo_offset + BLOCK_THREADS <= BINS; histo_offset += BLOCK_THREADS)
    {
      histogram[histo_offset + linear_tid] = 0;
    }

    if ((BINS % BLOCK_THREADS != 0) && (histo_offset + linear_tid < BINS))
    {
      histogram[histo_offset + linear_tid] = 0;
    }
  }

  template <typename CounterT>
  __device__ __forceinline__ void Composite(T (&items)[ITEMS_PER_THREAD],
                                            CounterT (&histogram)[BINS])
  {
    constexpr unsigned int bin_mask = static_cast<unsigned int>(BINS - 1);
    constexpr bool bins_power_of_two = (BINS & (BINS - 1)) == 0;
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i)
    {
      unsigned int idx = IndexLoader<T>::load(&items[i]);
      if (bins_power_of_two)
      {
        idx &= bin_mask;
      }
      if (idx < BINS)
      {
        atomicAdd(&histogram[idx], 1);
      }
    }
  }
};
} // namespace cub
"""


class BlockHistogramInit(BasePrimitive):
    is_child = True
    c_name = "init"
    method_name = "InitHistogram"
    struct_name = STRUCT_NAME
    includes = INCLUDES

    def __init__(
        self,
        parent: "BlockHistogram",
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


class BlockHistogramComposite(BasePrimitive):
    is_child = True
    c_name = "composite"
    method_name = "Composite"
    struct_name = STRUCT_NAME
    includes = INCLUDES

    c_name = "composite"
    method_name = "Composite"

    def __init__(
        self,
        parent: "BlockHistogram",
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
        type_definitions = None

        if algorithm == BlockHistogramAlgorithm.ATOMIC:
            struct_name = "BlockHistogramAtomicWrapper"
            type_definitions = [SimpleNamespace(code=ATOMIC_WRAPPER_CODE, lto_irs=[])]

        self.algorithm = Algorithm(
            struct_name,
            method_name,
            c_name,
            includes,
            self.template_parameters,
            parameters,
            primitive=self,
            unique_id=unique_id,
            type_definitions=type_definitions,
        )

        # Save attributes our children will need.
        self.item_dtype = item_dtype
        self.counter_dtype = counter_dtype
        self.items_per_thread = items_per_thread
        self.bins = bins
        self.struct_name = struct_name
        self.includes = includes
        self.type_definitions = type_definitions

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
