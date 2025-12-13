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
template <typename KeyT, int BlockDimX, int ItemsPerThread, typename ValueT = NullType>
class BlockTopK
{
private:
  using BlockRadixSortT = BlockRadixSort<KeyT, BlockDimX, ItemsPerThread, ValueT>;

public:
  struct TempStorage
  {
    typename BlockRadixSortT::TempStorage sort_storage;
  };

private:
  TempStorage& temp_storage;
  int linear_tid;

public:
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockTopK(TempStorage& temp_storage)
      : temp_storage(temp_storage)
      , linear_tid(threadIdx.x)
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE void
  Max(KeyT (&keys)[ItemsPerThread],
      ValueT (&values)[ItemsPerThread],
      int k,
      int begin_bit = 0,
      int end_bit   = sizeof(KeyT) * 8)
  {
    BlockRadixSortT(temp_storage.sort_storage).SortDescending(keys, values, begin_bit, end_bit);
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void
  Max(KeyT (&keys)[ItemsPerThread], int k, int begin_bit = 0, int end_bit = sizeof(KeyT) * 8)
  {
    BlockRadixSortT(temp_storage.sort_storage).SortDescending(keys, begin_bit, end_bit);
  }
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  Min(KeyT (&keys)[ItemsPerThread],
      ValueT (&values)[ItemsPerThread],
      int k,
      int begin_bit = 0,
      int end_bit   = sizeof(KeyT) * 8)
  {
    BlockRadixSortT(temp_storage.sort_storage).Sort(keys, values, begin_bit, end_bit);
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void
  Min(KeyT (&keys)[ItemsPerThread], int k, int begin_bit = 0, int end_bit = sizeof(KeyT) * 8)
  {
    BlockRadixSortT(temp_storage.sort_storage).Sort(keys, begin_bit, end_bit);
  }
};
} // namespace detail

CUB_NAMESPACE_END
