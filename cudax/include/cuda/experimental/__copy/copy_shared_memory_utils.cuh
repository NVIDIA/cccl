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

#  include <cuda/experimental/__copy_bytes/tensor_query.cuh>
#  include <cuda/experimental/__copy_bytes/types.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! Maximum tensor rank for which the shared-memory transpose kernel is instantiated. Higher ranks cause excessive
//! register pressure (many rank-sized arrays and fully-unrolled loops).
inline constexpr ::cuda::std::size_t __max_shared_mem_kernel_rank = 8;

//! A tile size is always representable by an unsigned integer.
using __tile_extent_t = unsigned;

//! @brief Copy a raw tensor descriptor into one with a narrower static maximum rank.
//!
//! @param[in] __tensor Raw tensor descriptor with dynamic rank matching _RankOut
//! @return Raw tensor descriptor with _RankOut as its static maximum rank
template <::cuda::std::size_t _RankOut, typename _ExtentT, typename _StrideT, typename _Tp, ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API __raw_tensor<_ExtentT, _StrideT, _Tp, _RankOut>
__narrow_raw_tensor_rank(const __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>& __tensor) noexcept
{
  _CCCL_ASSERT(__tensor.__rank == _RankOut, "tensor rank must match the narrowed static rank");
  __raw_tensor<_ExtentT, _StrideT, _Tp, _RankOut> __result{__tensor.__data, _RankOut, {}, {}};
  for (::cuda::std::size_t __i = 0; __i < _RankOut; ++__i)
  {
    __result.__extents[__i] = __tensor.__extents[__i];
    __result.__strides[__i] = __tensor.__strides[__i];
  }
  return __result;
}

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

// The structure holds the tiling information to optimize the transpose (shared-memory) kernel.
// - __tile_sizes: the size of each tile dimension in shared-memory
// - __src_perm: the permutation of the source dimensions (copy to shared-memory)
// - __dst_perm: the permutation of the destination dimensions (copy from shared-memory)
// - __tile_total_size: the total size of the tile in shared-memory
// - __active_tile_dims: the number of dimensions covered by the tile
// - __active_32_dims: the number of tile dimensions with extent __max_tile_size
// - __is_valid: true if the tiling is valid
// - __use_xor_swizzle: true if the XOR swizzle is used
template <::cuda::std::size_t _MaxRank>
struct __shared_mem_tiling_result
{
  ::cuda::std::array<__tile_extent_t, _MaxRank> __tile_sizes{};
  ::cuda::std::array<::cuda::std::size_t, _MaxRank> __src_perm{};
  ::cuda::std::array<::cuda::std::size_t, _MaxRank> __dst_perm{};
  ::cuda::std::size_t __tile_total_size  = 1;
  ::cuda::std::size_t __active_tile_dims = 0;
  ::cuda::std::size_t __active_32_dims   = 0;
  bool __is_valid                        = false;
  bool __use_xor_swizzle                 = false;
};

//! @brief Adds a contiguous stride-1 run from one tensor to the shared-memory tile.
//!
//! @param[in]     __tensor                 Raw tensor descriptor used to find coalesced modes
//! @param[in]     __perm                   Mode order to scan
//! @param[in,out] __result                 Shared-memory tiling result updated with selected tile sizes
//! @param[in]     __max_shared_mem_bytes   Maximum shared-memory capacity for one tile
//! @return Number of coalesced elements covered by this tensor's selected tile run
template <typename _SmemTp, typename _ExtentT, typename _StrideT, typename _Tp, ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API ::cuda::std::size_t __add_coalesced_tile_run(
  const __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>& __tensor,
  const ::cuda::std::array<::cuda::std::size_t, _MaxRank>& __perm,
  __shared_mem_tiling_result<_MaxRank>& __result,
  ::cuda::std::size_t __max_shared_mem_bytes) noexcept
{
  using ::cuda::std::size_t;
  size_t __coalesced_tile_size = 1;
  _StrideT __expected_stride   = 1;

  for (size_t __i = 0; __i < __tensor.__rank; ++__i)
  {
    const auto __perm_i = __perm[__i];
    const auto __extent = static_cast<size_t>(__tensor.__extents[__perm_i]);
    const auto __stride = ::cuda::experimental::__abs_integer(__tensor.__strides[__perm_i]);
    if (__stride != __expected_stride) // input tensor not contiguous
    {
      break;
    }

    if (__result.__tile_sizes[__perm_i] == 1) // first time we see this dimension
    {
      const auto __tile_size             = ::cuda::std::min(__extent, __max_tile_size);
      const auto __tile_total_size_bytes = __result.__tile_total_size * __tile_size * sizeof(_SmemTp);
      if (__tile_total_size_bytes > __max_shared_mem_bytes)
      {
        break;
      }
      // if the tile fits in shared-memory, update the result
      __result.__tile_sizes[__perm_i] = static_cast<__tile_extent_t>(__tile_size);
      __result.__tile_total_size *= __tile_size;
      ++__result.__active_tile_dims;
      if (__tile_size == __max_tile_size)
      {
        ++__result.__active_32_dims;
      }
    }
    __coalesced_tile_size *= __result.__tile_sizes[__perm_i];
    __expected_stride *= static_cast<_StrideT>(__extent);
  }
  return __coalesced_tile_size;
}

//! @brief Compute a source/destination-aware shared-memory tile.
//!
//! The selected tile spans coalesced dimensions from both layouts. This keeps the load phase ordered by source stride
//! and the store phase ordered by destination stride, without requiring either coalesced dimension to be mode 0.
//!
//! @param[in] __src Source raw tensor descriptor
//! @param[in] __dst Destination raw tensor descriptor
//! @return Shared-memory tiling decision and layout permutations
template <typename _TpIn,
          typename _ExtentT,
          typename _StrideTIn,
          typename _TpSrc,
          typename _StrideTOut,
          typename _TpDst,
          ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API __shared_mem_tiling_result<_MaxRank>
__find_shared_mem_tiling(const __raw_tensor<_ExtentT, _StrideTIn, _TpSrc, _MaxRank>& __src,
                         const __raw_tensor<_ExtentT, _StrideTOut, _TpDst, _MaxRank>& __dst) noexcept
{
  using ::cuda::std::size_t;
  __shared_mem_tiling_result<_MaxRank> __result{};
  // initialize the source and destination permutations and sort them by stride
  for (size_t __i = 0; __i < _MaxRank; ++__i)
  {
    __result.__tile_sizes[__i] = 1;
    __result.__src_perm[__i]   = __i;
    __result.__dst_perm[__i]   = __i;
  }
  __result.__src_perm = ::cuda::experimental::__stride_order(__src);
  __result.__dst_perm = ::cuda::experimental::__stride_order(__dst);

  const auto __current_dev            = ::cuda::experimental::__current_device();
  const size_t __max_shared_mem_bytes = __current_dev.attribute<::cudaDevAttrMaxSharedMemoryPerBlock>();
  const auto __src_coalesced_tile_size =
    ::cuda::experimental::__add_coalesced_tile_run<_TpIn>(__src, __result.__src_perm, __result, __max_shared_mem_bytes);
  const auto __dst_coalesced_tile_size =
    ::cuda::experimental::__add_coalesced_tile_run<_TpIn>(__dst, __result.__dst_perm, __result, __max_shared_mem_bytes);

  // If the tile total size is too small, or the coalescing is not useful on both sides, or the number of active tile
  // dimensions is less than 2, return the result.
  if (__result.__tile_total_size < __max_tile_size * 8 || __src_coalesced_tile_size < 2 || __dst_coalesced_tile_size < 2
      || __result.__active_tile_dims < 2)
  {
    return __result;
  }

  // There must be enough blocks to keep the GPU busy (at least one full wave across all SMs).
  const size_t __num_sms = __current_dev.attribute<::cudaDevAttrMultiProcessorCount>();
  size_t __num_tiles     = 1;
  for (size_t __r = 0; __r < __dst.__rank; ++__r)
  {
    const auto __extent    = static_cast<size_t>(__dst.__extents[__r]);
    const auto __tile_size = static_cast<size_t>(__result.__tile_sizes[__r]);
    __num_tiles *= ::cuda::ceil_div(__extent, __tile_size);
  }
  if (__num_tiles < __num_sms)
  {
    return __result;
  }

  __result.__is_valid = true;
  // Shared memory swizzle makes sense only for 32-bit and 64-bit types.
  __result.__use_xor_swizzle = (sizeof(_TpIn) == 4 || sizeof(_TpIn) == 8) //
                            && __result.__active_32_dims == 2;
  return __result;
}

//! @brief Decide whether the shared-memory tiled transpose kernel is profitable.
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
  return ::cuda::experimental::__find_shared_mem_tiling<_TpIn>(__src, __dst).__is_valid;
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
