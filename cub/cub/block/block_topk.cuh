// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! The @c cub::detail::block_topk class provides a :ref:`collective <collective-primitives>` method for selecting the
//! top-k elements from a set of items within a CUDA thread block.

#pragma once

#include <cub/config.cuh>

#include <cstdint>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/block_exchange.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/specializations/block_topk_air.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

CUB_NAMESPACE_BEGIN

namespace detail
{
enum class block_topk_algorithm
{
  air_top_k
};

// TODO (elstehle): Add documentation
template <typename KeyT, int BlockDimX, int ItemsPerThread, typename ValueT = NullType>
class block_topk
{
private:
  using internal_block_topk_t = block_topk_air<KeyT, BlockDimX, ItemsPerThread, ValueT>;

public:
  struct TempStorage
  {
    typename internal_block_topk_t::TempStorage topk_storage;
  };

private:
  TempStorage& storage;

public:
  _CCCL_DEVICE _CCCL_FORCEINLINE block_topk(TempStorage& storage)
      : storage(storage)
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE void max_pairs(
    KeyT (&keys)[ItemsPerThread],
    ValueT (&values)[ItemsPerThread],
    int k,
    int begin_bit = 0,
    int end_bit   = sizeof(KeyT) * 8)
  {
    internal_block_topk_t(storage.topk_storage)
      .template select_pairs<detail::topk::select::max>(keys, values, k, begin_bit, end_bit);
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void
  max_keys(KeyT (&keys)[ItemsPerThread], int k, int begin_bit = 0, int end_bit = sizeof(KeyT) * 8)
  {
    internal_block_topk_t(storage.topk_storage)
      .template select_keys<detail::topk::select::max>(keys, k, begin_bit, end_bit);
  }
  _CCCL_DEVICE _CCCL_FORCEINLINE void min_pairs(
    KeyT (&keys)[ItemsPerThread],
    ValueT (&values)[ItemsPerThread],
    int k,
    int begin_bit = 0,
    int end_bit   = sizeof(KeyT) * 8)
  {
    internal_block_topk_t(storage.topk_storage)
      .template select_pairs<detail::topk::select::min>(keys, values, k, begin_bit, end_bit);
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void
  min_keys(KeyT (&keys)[ItemsPerThread], int k, int begin_bit = 0, int end_bit = sizeof(KeyT) * 8)
  {
    internal_block_topk_t(storage.topk_storage)
      .template select_keys<detail::topk::select::min>(keys, k, begin_bit, end_bit);
  }
};
} // namespace detail

CUB_NAMESPACE_END
