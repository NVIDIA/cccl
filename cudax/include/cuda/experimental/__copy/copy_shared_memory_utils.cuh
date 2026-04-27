//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__COPY_COPY_SHARED_MEMORY_UTILS_H
#define _CUDAX__COPY_COPY_SHARED_MEMORY_UTILS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/__cmath/ceil_div.h>
#  include <cuda/__cmath/round_up.h>
#  include <cuda/__device/all_devices.h>
#  include <cuda/__device/attributes.h>
#  include <cuda/__device/device_ref.h>
#  include <cuda/__driver/driver_api.h>
#  include <cuda/std/__algorithm/min.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/array>

#  include <cuda/experimental/__copy_bytes/types.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! Maximum tensor rank for which the shared-memory transpose kernel is instantiated. Higher ranks cause excessive
//! register pressure (many rank-sized arrays and fully-unrolled loops).
inline constexpr ::cuda::std::size_t __max_shared_mem_kernel_rank = 8;

//! A tile size is always representable by an unsigned integer.
using __tile_extent_t = unsigned;

//! @brief Count the number of leading contiguous dimensions in a raw tensor.
//!
//! Starting from dimension 0, counts consecutive dimensions where `stride[0] == 1` and `stride[i] == stride[i-1] *
//! extent[i-1]` for each subsequent dimension.
//!
//! @param[in] __tensor Raw tensor descriptor
//! @return Number of leading contiguous dimensions (0 if stride[0] != 1 or rank is 0)
template <typename _ExtentT, typename _StrideT, typename _Tp, ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API ::cuda::std::size_t
__num_contiguous_dimensions(const __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>& __tensor) noexcept
{
  using __rank_t = typename __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>::__rank_t;
  if (__tensor.__rank == 0 || __tensor.__strides[0] != 1)
  {
    return 0;
  }
  __rank_t __count       = 1;
  auto __expected_stride = static_cast<_StrideT>(__tensor.__extents[0]);
  for (__rank_t __i = 1; __i < __tensor.__rank; ++__i)
  {
    if (__tensor.__strides[__i] != __expected_stride)
    {
      break;
    }
    __expected_stride *= static_cast<_StrideT>(__tensor.__extents[__i]);
    ++__count;
  }
  return __count;
}

//! @brief Return a device_ref for the current CUDA device.
//!
//! @return Device reference for the active CUDA context's device
[[nodiscard]] _CCCL_HOST_API inline ::cuda::device_ref __current_device() noexcept
{
  const auto __dev_id = ::cuda::__driver::__cudevice_to_ordinal(::cuda::__driver::__ctxGetDevice());
  return ::cuda::devices[__dev_id];
}

//! Maximum extent of a single tile dimension, set to the warp size so that the innermost tile dimension maps to a
//! full warp of coalesced accesses.
inline constexpr size_t __max_tile_size = 32;

//! @brief Decide whether the shared-memory tiled transpose kernel is profitable.
//!
//! Returns true when the destination has stride-1 in mode 0, the source does not, there are at least two contiguous
//! destination dimensions, the resulting tile is large enough to amortize the shared-memory overhead, and the total
//! number of tiles is sufficient to utilize the GPU (at least one full wave across all SMs).
//!
//! @param[in] __src Source raw tensor descriptor
//! @param[in] __dst Destination raw tensor descriptor
//! @return true if the shared-memory kernel should be used
template <typename _ExtentT,
          typename _StrideTIn,
          typename _StrideTOut,
          typename _TpIn,
          typename _TpOut,
          ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API bool
__use_shared_mem_kernel(const __raw_tensor<_ExtentT, _StrideTIn, _TpIn, _MaxRank>& __src,
                        const __raw_tensor<_ExtentT, _StrideTOut, _TpOut, _MaxRank>& __dst) noexcept
{
  using ::cuda::std::size_t;
  using __rank_t                    = typename __raw_tensor<_ExtentT, _StrideTOut, _TpOut, _MaxRank>::__rank_t;
  const size_t __num_contiguous_dst = ::cuda::experimental::__num_contiguous_dimensions(__dst);
  // * destination is contiguous (in dimension 0) -> coalesced destination writes
  // * source is not contiguous (already excluded by vectorized/contiguous copy)
  // * there are at least two contiguous destination dimensions -> otherwise, direct copy is better
  if (__src.__strides[0] == 1 || __dst.__strides[0] != 1 || __num_contiguous_dst < 2)
  {
    return false;
  }

  // * the tile is large enough to benefits from coalesced memory accesses.
  // algorithm:
  // - accumulate the product of the tile sizes until the shared memory limit is reached
  // - the tile size (which is at block level) is capped at the warp size to get coalesced accesses
  const auto __current_dev            = ::cuda::experimental::__current_device();
  const size_t __max_shared_mem_bytes = __current_dev.attribute<::cudaDevAttrMaxSharedMemoryPerBlock>();
  size_t __size_product               = 1;
  int __tile_rank                     = 0;
  for (size_t __r = 0; __r < __num_contiguous_dst; ++__r, ++__tile_rank)
  {
    const auto __tile_size_r = ::cuda::std::min(static_cast<size_t>(__dst.__extents[__r]), __max_tile_size);
    if (__size_product * __tile_size_r * sizeof(_TpIn) > __max_shared_mem_bytes)
    {
      break;
    }
    __size_product *= __tile_size_r;
  }
  // if the final tile size is too small, exit
  // 8 means each warp performs (block size / warp size) iterations >= 8
  if (__tile_rank < 2 || __size_product < __max_tile_size * 8)
  {
    return false;
  }

  // * there are enough tiles to keep the GPU busy (at least one full wave across all SMs)
  size_t __num_tiles = 1; // __num_tiles == number of blocks
  for (__rank_t __r = 0; __r < __dst.__rank; ++__r)
  {
    const auto __extent    = static_cast<size_t>(__dst.__extents[__r]);
    const auto __tile_size = (__r < __tile_rank) ? ::cuda::std::min(__extent, __max_tile_size) : size_t{1};
    __num_tiles *= ::cuda::ceil_div(__extent, __tile_size);
  }
  const size_t __num_sms = __current_dev.attribute<::cudaDevAttrMultiProcessorCount>();
  return __num_tiles >= __num_sms;
}

//! @brief Compute the shared-memory tile sizes for a destination tensor.
//!
//! Greedily expands the tile across contiguous dimensions up to the warp-size cap per dimension and the device
//! shared-memory limit. Dimensions beyond the tile rank are set to extent 1.
//!
//! @param[in]  __tensor          Destination raw tensor descriptor
//! @param[out] __tile_total_size Total number of elements in one tile (output)
//! @return Per-dimension tile sizes (unused dimensions are 1)
template <typename _TpIn, typename _ExtentT, typename _StrideT, typename _TpOut, ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API ::cuda::std::array<__tile_extent_t, _MaxRank> __find_shared_mem_tiling(
  const __raw_tensor<_ExtentT, _StrideT, _TpOut, _MaxRank>& __tensor, ::cuda::std::size_t& __tile_total_size) noexcept
{
  using ::cuda::std::size_t;
  const auto __current_dev            = ::cuda::experimental::__current_device();
  const size_t __max_shared_mem_bytes = __current_dev.attribute<::cudaDevAttrMaxSharedMemoryPerBlock>();
  const size_t __num_contiguous_dst   = ::cuda::experimental::__num_contiguous_dimensions(__tensor);

  ::cuda::std::array<__tile_extent_t, _MaxRank> __tile_sizes{};
  __tile_total_size  = 1;
  size_t __tile_rank = 0;
  for (size_t __r = 0; __r < __num_contiguous_dst; ++__r, ++__tile_rank)
  {
    const auto __tile_size_r = ::cuda::std::min(static_cast<size_t>(__tensor.__extents[__r]), __max_tile_size);
    if (__tile_total_size * __tile_size_r * sizeof(_TpIn) > __max_shared_mem_bytes)
    {
      break;
    }
    __tile_sizes[__r] = static_cast<__tile_extent_t>(__tile_size_r);
    __tile_total_size *= __tile_size_r;
  }
  for (size_t __r = __tile_rank; __r < _MaxRank; ++__r)
  {
    __tile_sizes[__r] = 1;
  }
  return __tile_sizes;
}

//! @brief Compute the thread block size for the shared-memory kernel.
//!
//! Balances occupancy by dividing the SM threads across as many blocks as the shared memory allows, then caps at
//! the device maximum.
//!
//! @param[in] __tile_total_bytes Shared memory required for one tile in bytes
//! @return Thread block size
[[nodiscard]] _CCCL_HOST_API inline int __find_thread_block_size(::cuda::std::size_t __tile_total_bytes) noexcept
{
  using ::cuda::std::size_t;
  const auto __dev                      = ::cuda::experimental::__current_device();
  const size_t __total_sm_threads       = __dev.attribute<::cudaDevAttrMaxThreadsPerMultiProcessor>();
  const size_t __max_thread_block_size  = __dev.attribute<::cudaDevAttrMaxThreadsPerBlock>();
  const size_t __total_shared_mem_bytes = __dev.attribute<::cudaDevAttrMaxSharedMemoryPerMultiprocessor>();
  const auto __num_blocks_per_sm        = __total_shared_mem_bytes / __tile_total_bytes;
  const auto __thread_block_size = ::cuda::std::min(__total_sm_threads / __num_blocks_per_sm, __max_thread_block_size);
  const auto __thread_block_size32 = ::cuda::round_up(__thread_block_size, /*warp size=*/size_t{32});
  return static_cast<int>(__thread_block_size32);
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // _CUDAX__COPY_COPY_SHARED_MEMORY_UTILS_H
