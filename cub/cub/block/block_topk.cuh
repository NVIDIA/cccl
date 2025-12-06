// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! The @c cub::BlockTopK class provides a :ref:`collective <collective-primitives>` method for selecting the top-k
//! elements from a set of items within a CUDA thread block.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/block_radix_sort.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

CUB_NAMESPACE_BEGIN

namespace detail
{
template <typename KeyT, int BLOCK_DIM_X, int items_per_thread, typename ValueT = cub::NullType>
class BlockTopK
{
private:
  // Internal CUB primitive
  using BlockRadixSortT = cub::BlockRadixSort<KeyT, BLOCK_DIM_X, items_per_thread, ValueT>;

public:
  // Expose TempStorage requirements
  struct TempStorage
  {
    typename BlockRadixSortT::TempStorage sort_storage;
  };

private:
  TempStorage& temp_storage;
  int linear_tid;

public:
  __device__ __forceinline__ BlockTopK(TempStorage& temp_storage)
      : temp_storage(temp_storage)
      , linear_tid(threadIdx.x)
  {}

  __device__ __forceinline__ void Select(
    KeyT (&keys)[items_per_thread],
    ValueT (&values)[items_per_thread],
    int k,
    bool is_descending = true,
    int valid_items    = BLOCK_DIM_X * items_per_thread,
    int begin_bit      = 0,
    int end_bit        = sizeof(KeyT) * 8)
  {
    // Delegate to CUB BlockRadixSort
    // Note: BlockRadixSort produces a BLOCKED arrangement.
    // Thread 0 has items [0 .. IPT-1], Thread 1 has [IPT .. 2*IPT-1], etc.

    if (is_descending)
    {
      // Sort Descending: Largest items move to Rank 0 (Thread 0)
      BlockRadixSortT(temp_storage.sort_storage).SortDescending(keys, values, begin_bit, end_bit);
    }
    else
    {
      // Sort Ascending: Smallest items move to Rank 0 (Thread 0)
      BlockRadixSortT(temp_storage.sort_storage).Sort(keys, values, begin_bit, end_bit);
    }
  }

  // Overload for Keys only
  __device__ __forceinline__ void Select(
    KeyT (&keys)[items_per_thread],
    int k,
    bool is_descending = true,
    int valid_items    = BLOCK_DIM_X * items_per_thread,
    int begin_bit      = 0,
    int end_bit        = sizeof(KeyT) * 8)
  {
    if (is_descending)
    {
      BlockRadixSortT(temp_storage.sort_storage).SortDescending(keys, begin_bit, end_bit);
    }
    else
    {
      BlockRadixSortT(temp_storage.sort_storage).Sort(keys, begin_bit, end_bit);
    }
  }
};
} // namespace detail

CUB_NAMESPACE_END
