//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO_DETAIL_UTILITY_CUDA_CUH
#define _CUDAX___CUCO_DETAIL_UTILITY_CUDA_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__hierarchy/hierarchy_levels.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/distance.h>
#include <cuda/std/cstdint>

#include <cooperative_groups.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::detail
{
using __index_type = ::cuda::std::int64_t;

#if _CCCL_CUDA_COMPILATION()
[[nodiscard]] _CCCL_DEVICE inline __index_type __global_thread_id() noexcept
{
  return ::cuda::gpu_thread.rank_as<__index_type>(::cuda::grid);
}

[[nodiscard]] _CCCL_DEVICE inline __index_type __grid_stride() noexcept
{
  return ::cuda::gpu_thread.count_as<__index_type>(::cuda::grid);
}
#endif // _CCCL_CUDA_COMPILATION()

inline constexpr int __default_block_size = 128;

inline constexpr int __default_stride = 1;

inline constexpr int __warp_size = 32;

template <class _Tile>
struct __tile_size;

template <::cuda::std::uint32_t _Size, class _ParentCG>
struct __tile_size<::cooperative_groups::thread_block_tile<_Size, _ParentCG>>
{
  static constexpr int __value = _Size;
};

template <class _Tile>
inline constexpr int __tile_size_v = __tile_size<_Tile>::__value;

constexpr _CCCL_HOST_DEVICE __index_type __grid_size(
  __index_type __num,
  int __cg_size    = 1,
  int __stride     = __default_stride,
  int __block_size = __default_block_size) noexcept
{
  return ::cuda::ceil_div(__cg_size * __num, __stride * __block_size);
}

//! @brief Distance helper requiring random access iterators.
template <class _Iterator>
[[nodiscard]] _CCCL_HOST_DEVICE constexpr __index_type __distance(_Iterator __begin, _Iterator __end)
{
  static_assert(::cuda::std::random_access_iterator<_Iterator>, "Input iterator should be a random access iterator.");
  return __index_type{::cuda::std::distance(__begin, __end)};
}
} // namespace cuda::experimental::cuco::detail

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_DETAIL_UTILITY_CUDA_CUH
