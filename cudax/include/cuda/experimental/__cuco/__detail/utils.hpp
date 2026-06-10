//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO___DETAIL_UTILS_HPP
#define _CUDAX___CUCO___DETAIL_UTILS_HPP

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__runtime/api_wrapper.h>
#include <cuda/cmath>
#include <cuda/hierarchy>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/distance.h>
#include <cuda/std/cstdint>

#include <cooperative_groups.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::__detail
{
using __index_type = ::cuda::std::int64_t;

[[nodiscard]] _CCCL_DEVICE inline __index_type __global_thread_id() noexcept
{
  return ::cuda::gpu_thread.rank_as<__index_type>(::cuda::grid);
}

[[nodiscard]] _CCCL_DEVICE inline __index_type __grid_stride() noexcept
{
  return ::cuda::gpu_thread.count_as<__index_type>(::cuda::grid);
}

constexpr _CCCL_HOST_DEVICE ::cuda::std::int32_t __default_block_size() noexcept
{
  return 128;
}

constexpr _CCCL_HOST_DEVICE ::cuda::std::int32_t __default_stride() noexcept
{
  return 1;
}

constexpr _CCCL_HOST_DEVICE ::cuda::std::int32_t __warp_size() noexcept
{
  return 32;
}

template <class _Tile>
struct __tile_size;

template <::cuda::std::uint32_t _Size, class _ParentCG>
struct __tile_size<::cooperative_groups::thread_block_tile<_Size, _ParentCG>>
{
  static constexpr ::cuda::std::uint32_t __value = _Size;
};

template <class _Tile>
inline constexpr ::cuda::std::uint32_t __tile_size_v = __tile_size<_Tile>::__value;

constexpr _CCCL_HOST_DEVICE __index_type __grid_size(
  __index_type __num,
  ::cuda::std::int32_t __cg_size    = 1,
  ::cuda::std::int32_t __stride     = __default_stride(),
  ::cuda::std::int32_t __block_size = __default_block_size()) noexcept
{
  return ::cuda::ceil_div(__cg_size * __num, __stride * __block_size);
}

#if !_CCCL_COMPILER(NVRTC)
template <class _Kernel>
constexpr auto __max_occupancy_grid_size(
  ::cuda::std::int32_t __block_size, _Kernel __kernel, ::cuda::std::size_t __dynamic_shm_size = 0)
{
  int __device = 0;
  _CCCL_TRY_CUDA_API(cudaGetDevice, "Failed to get current device", &__device);

  int __num_multiprocessors = -1;
  _CCCL_TRY_CUDA_API(
    cudaDeviceGetAttribute,
    "Failed to get multiprocessor count",
    &__num_multiprocessors,
    cudaDevAttrMultiProcessorCount,
    __device);

  int __max_active_blocks_per_multiprocessor{};
  _CCCL_TRY_CUDA_API(
    cudaOccupancyMaxActiveBlocksPerMultiprocessor,
    "Failed to get max active blocks per multiprocessor",
    &__max_active_blocks_per_multiprocessor,
    __kernel,
    __block_size,
    __dynamic_shm_size);

  return __max_active_blocks_per_multiprocessor * __num_multiprocessors;
}
#endif // !_CCCL_COMPILER(NVRTC)
//! @brief Distance helper requiring random access iterators.

template <class _Iterator>
[[nodiscard]] _CCCL_HOST_DEVICE constexpr __index_type __distance(_Iterator __begin, _Iterator __end)
{
  static_assert(::cuda::std::random_access_iterator<_Iterator>, "Input iterator should be a random access iterator.");
  return __index_type{::cuda::std::distance(__begin, __end)};
}
} // namespace cuda::experimental::cuco::__detail

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO___DETAIL_UTILS_HPP
