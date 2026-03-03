//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__COPY_BYTES_COPY_BYTES_SHARED_MEM
#define _CUDAX__COPY_BYTES_COPY_BYTES_SHARED_MEM

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__launch/configuration.h>
#include <cuda/__launch/launch.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/devices>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__algorithm/stable_sort.h>
#include <cuda/std/__cstdlib/abs.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/array>
#include <cuda/std/cstdint>

#include <cuda/experimental/__copy/types.cuh>
#include <cuda/experimental/__copy/utils.cuh>
#include <cuda/experimental/__copy_bytes/layout_optimization.cuh>
//
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief Decompose a linear index into a strided offset via modulo/divide chain.
//!
//! For each mode k in [0, __rank), extracts coordinate `__linear_idx % __sizes[k]`,
//! multiplies by `__strides[k]`, and accumulates the result. Mode 0 is the
//! fastest-varying (innermost) dimension.
//!
//! @param[in] __linear_idx  Linear index to decompose
//! @param[in] __sizes       Per-mode extents for the decomposition
//! @param[in] __strides     Per-mode strides to dot with extracted coordinates
//! @param[in] __rank        Number of active modes
//! @return The accumulated offset
template <typename _Ep, typename _Sp, ::cuda::std::size_t _MaxRank>
_CCCL_DEVICE auto __linear_to_offset(
  int __linear_idx,
  const ::cuda::std::array<_Ep, _MaxRank>& __sizes,
  const ::cuda::std::array<_Sp, _MaxRank>& __strides,
  int __rank)
{
  _Sp __offset = 0;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int __i = 0; __i < int{_MaxRank}; ++__i)
  {
    if (__i < __rank)
    {
      const auto __coord = __linear_idx % __sizes[__i];
      __linear_idx /= __sizes[__i];
      __offset += __coord * __strides[__i];
    }
  }
  return __offset;
}

//! @brief Shared-memory tiled transpose kernel for arbitrary-rank tensors.
//!
//! Each block processes one R-dimensional tile. Threads cooperatively iterate
//! over tile elements with a stride loop (blockDim.x may be smaller than the tile).
//! Phase 1 loads from global source into shared memory using a src-coalesced thread mapping.
//! Phase 2 stores from shared memory to global destination using a dst-coalesced thread mapping.
//!
//! @tparam _MaxRank  Maximum supported tensor rank
//! @tparam _Tp       Element type
template <int _MaxRank, typename _Tp, typename _Ep, typename _Sp>
__global__ void __copy_bytes_shared_mem_kernel(
  const _Tp* const _CCCL_RESTRICT __src_ptr,
  _Tp* const _CCCL_RESTRICT __dst_ptr,
  const ::cuda::std::array<_Ep, _MaxRank> __grid_tile_sizes,
  const ::cuda::std::array<_Sp, _MaxRank> __grid_tile_src_strides,
  const ::cuda::std::array<_Sp, _MaxRank> __grid_tile_dst_strides,
  const ::cuda::std::array<unsigned, _MaxRank> __src_perm_sizes,
  const ::cuda::std::array<_Sp, _MaxRank> __src_perm_src_strides,
  const ::cuda::std::array<int, _MaxRank> __src_perm_smem_strides,
  const ::cuda::std::array<unsigned, _MaxRank> __tile_sizes,
  const ::cuda::std::array<_Sp, _MaxRank> __dst_strides,
  const int __rank,
  const int __tile_total_size)
{
  namespace cudax = ::cuda::experimental;
  extern __shared__ char __smem_bytes[];
  _Tp* __smem = reinterpret_cast<_Tp*>(__smem_bytes);

  const int __tid    = static_cast<int>(threadIdx.x);
  const int __stride = static_cast<int>(blockDim.x);

  const auto __src_base =
    cudax::__linear_to_offset(static_cast<int>(blockIdx.x), __grid_tile_sizes, __grid_tile_src_strides, __rank);
  const auto __dst_base =
    cudax::__linear_to_offset(static_cast<int>(blockIdx.x), __grid_tile_sizes, __grid_tile_dst_strides, __rank);

  for (int __i = __tid; __i < __tile_total_size; __i += __stride)
  {
    const auto __src_elem = cudax::__linear_to_offset(__i, __src_perm_sizes, __src_perm_src_strides, __rank);
    const auto __smem_idx = cudax::__linear_to_offset(__i, __src_perm_sizes, __src_perm_smem_strides, __rank);
    __smem[__smem_idx]    = __src_ptr[__src_base + __src_elem];
  }

  __syncthreads();

  for (int __i = __tid; __i < __tile_total_size; __i += __stride)
  {
    const auto __dst_elem              = cudax::__linear_to_offset(__i, __tile_sizes, __dst_strides, __rank);
    __dst_ptr[__dst_base + __dst_elem] = __smem[__i];
  }
}

#if !_CCCL_COMPILER(NVRTC)

_CCCL_HOST_API inline device_ref __current_device()
{
  const auto __dev_id = ::cuda::__driver::__cudevice_to_ordinal(::cuda::__driver::__ctxGetDevice());
  return ::cuda::devices[__dev_id];
}

template <typename _Ep, typename _Sp, typename _TpSrc, typename _TpDst, ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API bool __use_shared_mem_kernel(
  const __raw_tensor<_Ep, _Sp, _TpSrc, _MaxRank>& __src, const __raw_tensor<_Ep, _Sp, _TpDst, _MaxRank>& __dst)
{
  const int __num_contiguous_dimensions = ::cuda::experimental::__num_contiguous_dimensions(__dst);
  if (__src.__strides[0] == 1 || __dst.__strides[0] != 1 || __num_contiguous_dimensions < 2)
  {
    return false;
  }
  constexpr int __warp_size           = 32;
  constexpr size_t __max_tile_size    = __warp_size; // warp size
  const size_t __max_shared_mem_bytes = __current_device().attribute<::cudaDevAttrMaxSharedMemoryPerBlockOptin>();
  size_t __size_product               = 1;
  int __tile_rank                     = 0;
  for (size_t __r = 0; __r < __num_contiguous_dimensions; ++__r, ++__tile_rank)
  {
    auto __tile_size_r = ::cuda::std::min(static_cast<size_t>(__dst.__extents[__r]), __max_tile_size);
    if (__size_product * __tile_size_r * sizeof(_TpDst) > __max_shared_mem_bytes)
    {
      break;
    }
    __size_product *= __tile_size_r;
  }
  return (__tile_rank >= 2 && __size_product >= __warp_size * 8); // potential tuning
}

template <typename _Ep, typename _Sp, typename _Tp, ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API __raw_tensor<int, int, _Tp, _MaxRank>
__find_shared_mem_tiling(const __raw_tensor<_Ep, _Sp, _Tp, _MaxRank>& __tensor, size_t& __tile_total_size)
{
  constexpr int __warp_size           = 32;
  constexpr size_t __max_tile_size    = __warp_size; // warp size
  const size_t __max_shared_mem_bytes = __current_device().attribute<::cudaDevAttrMaxSharedMemoryPerBlockOptin>();
  __raw_tensor<int, int, _Tp, _MaxRank> __tiling{};
  auto& __tile_rank    = __tiling.__rank;
  auto& __tile_extents = __tiling.__extents;
  auto& __tile_strides = __tiling.__strides;

  const int __num_contiguous_dimensions = ::cuda::experimental::__num_contiguous_dimensions(__tensor);
  int __size_product                    = 1;
  __tile_rank                           = 0;
  for (int __r = 0; __r < __num_contiguous_dimensions; ++__r, ++__tile_rank)
  {
    const auto __tile_size_r = ::cuda::std::min(static_cast<size_t>(__tensor.__extents[__r]), __max_tile_size);
    // tile does not fit in shared memory
    if (__size_product * __tile_size_r * sizeof(_Tp) > __max_shared_mem_bytes)
    {
      break;
    }
    __tile_extents[__r] = __tile_size_r;
    __tile_strides[__r] = __size_product; // contiguous stride
    __size_product *= __tile_size_r;
  }
  for (int __r = __tile_rank; __r < __tensor.__rank; ++__r)
  {
    __tile_extents[__r] = 1;
    __tile_strides[__r] = __size_product;
  }
  __tile_total_size = __size_product;
  return __tiling;
}

[[nodiscard]] _CCCL_HOST_API int __find_thread_block_size(size_t __tile_total_bytes)
{
  const auto __dev                      = __current_device();
  const size_t __total_sm_threads       = __dev.attribute<::cudaDevAttrMaxThreadsPerMultiProcessor>();
  const size_t __max_thread_block_size  = __dev.attribute<::cudaDevAttrMaxThreadsPerBlock>();
  const size_t __total_shared_mem_bytes = __dev.attribute<::cudaDevAttrMaxSharedMemoryPerBlock>();

  const auto __num_blocks = ::cuda::ceil_div(__total_shared_mem_bytes, __tile_total_bytes);
  const auto __thread_block_size =
    ::cuda::std::min(::cuda::ceil_div(__total_sm_threads, __num_blocks), __max_thread_block_size);
  return __thread_block_size;
}

//! @brief Launch the shared-memory tiled transpose kernel for the "transpose case"
//!        (dst stride-1 in mode 0, src stride-1 elsewhere).
//!
//! Precomputes the reordered arrays for src-coalesced thread decomposition, then
//! launches one block per tile. Threads iterate over tile elements with a stride loop.
//!
//! @pre `__src.__rank >= 2`
//! @pre `__dst.__strides[0] == 1`
//! @pre `__src.__strides[0] != 1`
//! @pre Tile sizes evenly divide tensor shapes (assumed for this prototype)
template <typename _Ep, typename _Sp, typename _TpSrc, typename _TpDst, ::cuda::std::size_t _MaxRank>
_CCCL_HOST_API void copy_bytes_shared_mem(
  const __raw_tensor<_Ep, _Sp, _TpSrc, _MaxRank>& __src,
  const __raw_tensor<_Ep, _Sp, _TpDst, _MaxRank>& __dst,
  ::cuda::stream_ref __stream)
{
  namespace cudax = ::cuda::experimental;
  using ::cuda::std::int64_t;
  using ::cuda::std::size_t;
  using __value_type = ::cuda::std::remove_cv_t<_TpSrc>;
  _CCCL_ASSERT(__src.__rank >= 2, "Rank must be at least 2 for shared memory transpose");
  _CCCL_ASSERT(__src.__rank == __dst.__rank, "Source and destination ranks must be equal");
  _CCCL_ASSERT(__src.__extents == __dst.__extents, "Source and destination extents must be identical");
  const size_t __rank = __src.__rank;

  size_t __tile_total_size      = 0;
  const auto __tiling           = cudax::__find_shared_mem_tiling(__dst, __tile_total_size);
  const int __thread_block_size = cudax::__find_thread_block_size(__tile_total_size * sizeof(__value_type));

  cuda::std::array<unsigned, _MaxRank> __tile_sizes = __tiling.__extents;

  //--------------------------------------------------------------------------------------------------------------------
  // Grid sizes and strides for block index decomposition
  ::cuda::std::array<_Ep, _MaxRank> __grid_tile_sizes{};
  ::cuda::std::array<_Sp, _MaxRank> __grid_tile_src_strides{};
  ::cuda::std::array<_Sp, _MaxRank> __grid_tile_dst_strides{};
  size_t __grid_size = 1;
  for (size_t __i = 0; __i < __rank; ++__i)
  {
    __grid_tile_sizes[__i]       = __src.__extents[__i] / __tile_sizes[__i];
    __grid_tile_src_strides[__i] = static_cast<_Sp>(__tile_sizes[__i]) * __src.__strides[__i];
    __grid_tile_dst_strides[__i] = static_cast<_Sp>(__tile_sizes[__i]) * __dst.__strides[__i];
    __grid_size *= __grid_tile_sizes[__i];
  }
  //--------------------------------------------------------------------------------------------------------------------
  // Src-coalesced permutation: sort modes by ascending |src_stride|
  ::cuda::std::array<int, _MaxRank> __src_perm{};
  for (size_t __i = 0; __i < _MaxRank; ++__i)
  {
    __src_perm[__i] = __i;
  }
  ::cuda::std::stable_sort(__src_perm.begin(), __src_perm.begin() + __rank, [&](size_t __a, size_t __b) {
    return ::cuda::std::abs(__src.__strides[__a]) < ::cuda::std::abs(__src.__strides[__b]);
  });
  //--------------------------------------------------------------------------------------------------------------------
  // Reordered arrays for loading src to shared memory (src-coalesced thread decomposition)
  ::cuda::std::array<unsigned, _MaxRank> __src_perm_sizes{};
  ::cuda::std::array<_Sp, _MaxRank> __src_perm_src_strides{};
  ::cuda::std::array<int, _MaxRank> __src_perm_smem_strides{};
  ::cuda::std::array<int, _MaxRank> __canonical_strides{};
  __canonical_strides[0] = 1;
  for (size_t __i = 1; __i < __rank; ++__i)
  {
    __canonical_strides[__i] = __canonical_strides[__i - 1] * __tile_sizes[__i - 1];
  }
  for (size_t __i = 0; __i < __rank; ++__i)
  {
    const auto __p               = __src_perm[__i];
    __src_perm_sizes[__i]        = __tile_sizes[__p];
    __src_perm_src_strides[__i]  = __src.__strides[__p];
    __src_perm_smem_strides[__i] = __canonical_strides[__p];
  }
  //--------------------------------------------------------------------------------------------------------------------
  // Launch the kernel
  const auto __config = ::cuda::make_config(
    ::cuda::block_dims(static_cast<unsigned>(__thread_block_size)),
    ::cuda::grid_dims(__grid_size),
    ::cuda::dynamic_shared_memory<__value_type[]>(__tile_total_size));
  const auto __kernel = cudax::__copy_bytes_shared_mem_kernel<static_cast<int>(_MaxRank), __value_type, _Ep, _Sp>;
  ::cuda::launch(
    __stream,
    __config,
    __kernel,
    static_cast<const __value_type*>(__src.__data),
    static_cast<__value_type*>(__dst.__data),
    __grid_tile_sizes,
    __grid_tile_src_strides,
    __grid_tile_dst_strides,
    __src_perm_sizes,
    __src_perm_src_strides,
    __src_perm_smem_strides,
    __tile_sizes,
    __dst.__strides,
    static_cast<int>(__rank),
    static_cast<int>(__tile_total_size));
}

#endif // !_CCCL_COMPILER(NVRTC)
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__COPY_BYTES_COPY_BYTES_SHARED_MEM
