//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO___STATIC_MAP_KERNELS_CUH
#define _CUDAX___CUCO___STATIC_MAP_KERNELS_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/block_reduce.cuh>

#include <cuda/atomic>
#include <cuda/std/cstdint>
#include <cuda/std/iterator>

#include <cuda/experimental/__cuco/__detail/bitwise_compare.cuh>
#include <cuda/experimental/__cuco/__detail/types.cuh>
#include <cuda/experimental/__cuco/__detail/utils.hpp>

#include <cooperative_groups.h>

#include <cooperative_groups/reduce.h>
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::__static_map
{
template <int _CgSize, int _BlockSize, class _InputIt, class _Ref>
CCCL_DETAIL_KERNEL_ATTRIBUTES void
__insert_or_assign(_InputIt __first, ::cuda::experimental::cuco::__detail::__index_type __n, _Ref __ref)
{
  const auto __loop_stride = ::cuda::experimental::cuco::__detail::__grid_stride() / _CgSize;
  auto __idx               = ::cuda::experimental::cuco::__detail::__global_thread_id() / _CgSize;

  while (__idx < __n)
  {
    typename ::cuda::std::iterator_traits<_InputIt>::value_type const __insert_pair = *(__first + __idx);
    if constexpr (_CgSize == 1)
    {
      __ref.insert_or_assign(__insert_pair);
    }
    else
    {
      const auto __tile = ::cooperative_groups::tiled_partition<_CgSize, ::cooperative_groups::thread_block>(
        ::cooperative_groups::this_thread_block());
      __ref.insert_or_assign(__tile, __insert_pair);
    }
    __idx += __loop_stride;
  }
}

template <bool _HasInit, int _CgSize, int _BlockSize, class _InputIt, class _Init, class _Op, class _Ref>
CCCL_DETAIL_KERNEL_ATTRIBUTES void __insert_or_apply(
  _InputIt __first,
  ::cuda::experimental::cuco::__detail::__index_type __n,
  [[maybe_unused]] _Init __init,
  _Op __op,
  _Ref __ref)
{
  const auto __loop_stride = ::cuda::experimental::cuco::__detail::__grid_stride() / _CgSize;
  auto __idx               = ::cuda::experimental::cuco::__detail::__global_thread_id() / _CgSize;

  while (__idx < __n)
  {
    using __value_type               = typename ::cuda::std::iterator_traits<_InputIt>::value_type;
    __value_type const __insert_pair = *(__first + __idx);
    if constexpr (_CgSize == 1)
    {
      if constexpr (_HasInit)
      {
        __ref.insert_or_apply(__insert_pair, __init, __op);
      }
      else
      {
        __ref.insert_or_apply(__insert_pair, __op);
      }
    }
    else
    {
      const auto __tile = ::cooperative_groups::tiled_partition<_CgSize, ::cooperative_groups::thread_block>(
        ::cooperative_groups::this_thread_block());
      if constexpr (_HasInit)
      {
        __ref.insert_or_apply(__tile, __insert_pair, __init, __op);
      }
      else
      {
        __ref.insert_or_apply(__tile, __insert_pair, __op);
      }
    }
    __idx += __loop_stride;
  }
}

//! @brief Shared-memory accelerated __insert_or_apply kernel.
//!
//! @tparam _NumBuckets Compile-time number of buckets for the shared memory map
//! @tparam _SharedMapRefType Ref type for the block-scoped shared memory map
template <bool _HasInit,
          int _CgSize,
          int _BlockSize,
          int _NumBuckets,
          class _SharedMapRefType,
          class _InputIt,
          class _Init,
          class _Op,
          class _Ref>
CCCL_DETAIL_KERNEL_ATTRIBUTES void __insert_or_apply_shmem(
  _InputIt __first,
  ::cuda::experimental::cuco::__detail::__index_type __n,
  [[maybe_unused]] _Init __init,
  _Op __op,
  _Ref __ref)
{
  static_assert(_CgSize == 1, "use shared_memory kernel only if cg_size == 1");
  namespace cg             = ::cooperative_groups;
  using __key_type         = typename _Ref::key_type;
  using __value_type       = typename _Ref::mapped_type;
  using __input_value_type = typename ::cuda::std::iterator_traits<_InputIt>::value_type;

  const auto __block       = cg::this_thread_block();
  const auto __thread_idx  = __block.thread_rank();
  const auto __loop_stride = ::cuda::experimental::cuco::__detail::__grid_stride() / _CgSize;
  auto __idx               = ::cuda::experimental::cuco::__detail::__global_thread_id() / _CgSize;

  auto __warp                  = cg::tiled_partition<32, cg::thread_block>(__block);
  const auto __warp_thread_idx = __warp.thread_rank();

  // Allocate shared memory slots with compile-time known size
  constexpr int __num_slots = _NumBuckets * _SharedMapRefType::bucket_size;
  __shared__ typename _SharedMapRefType::value_type __slots[__num_slots];

  // Create the storage ref for the shared memory slots
  using __storage_ref_type = typename _SharedMapRefType::storage_ref_type;
  auto __storage           = __storage_ref_type{__slots, _NumBuckets};

  using __atomic_type = ::cuda::atomic<::cuda::std::int32_t, ::cuda::thread_scope_block>;
  __shared__ __atomic_type __block_cardinality;
  if (__thread_idx == 0)
  {
    new (&__block_cardinality) __atomic_type{};
  }
  __block.sync();

  // Construct the shared-memory map ref directly (no operator rebinding needed)
  auto __shared_map = _SharedMapRefType{
    typename _SharedMapRefType::empty_key{__ref.empty_key_sentinel()},
    typename _SharedMapRefType::empty_value{__ref.empty_value_sentinel()},
    __ref.key_eq(),
    __ref.probing_scheme(),
    __storage};
  __shared_map.initialize(__block);
  __block.sync();

  while ((__idx - __thread_idx / _CgSize) < __n)
  {
    ::cuda::std::int32_t __inserted         = 0;
    ::cuda::std::int32_t __warp_cardinality = 0;
    if (__idx < __n)
    {
      __input_value_type const __insert_pair = *(__first + __idx);
      if constexpr (_HasInit)
      {
        __inserted = __shared_map.insert_or_apply(__insert_pair, __init, __op);
      }
      else
      {
        __inserted = __shared_map.insert_or_apply(__insert_pair, __op);
      }
    }
    if (__idx - __warp_thread_idx < __n)
    {
      __warp_cardinality = cg::reduce(__warp, __inserted, cg::plus<::cuda::std::int32_t>());
    }
    if (__warp_thread_idx == 0)
    {
      __block_cardinality.fetch_add(__warp_cardinality, ::cuda::memory_order_relaxed);
    }
    __block.sync();
    if (__block_cardinality > _BlockSize)
    {
      break;
    }
    __idx += __loop_stride;
  }

  // Flush shared memory map into the global map
  auto __bucket_idx = __thread_idx;
  while (__bucket_idx < _NumBuckets)
  {
    const auto __slot = __storage[__bucket_idx][0];
    if (!::cuda::experimental::cuco::__detail::__bitwise_compare(__slot.first, __ref.empty_key_sentinel()))
    {
      if constexpr (_HasInit)
      {
        __ref.insert_or_apply(__slot, __init, __op);
      }
      else
      {
        __ref.insert_or_apply(__slot, __op);
      }
    }
    __bucket_idx += _BlockSize;
  }

  // Handle overflow: if shared map got too full, process remaining elements directly
  if (__block_cardinality > _BlockSize)
  {
    __idx += __loop_stride;
    while (__idx < __n)
    {
      __input_value_type const __insert_pair = *(__first + __idx);
      if constexpr (_HasInit)
      {
        __ref.insert_or_apply(__insert_pair, __init, __op);
      }
      else
      {
        __ref.insert_or_apply(__insert_pair, __op);
      }
      __idx += __loop_stride;
    }
  }
}
} // namespace cuda::experimental::cuco::__static_map

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO___STATIC_MAP_KERNELS_CUH
