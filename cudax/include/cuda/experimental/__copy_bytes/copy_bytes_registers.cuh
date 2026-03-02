//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__COPY_BYTES_COPY_BYTES_REGISTERS
#define _CUDAX__COPY_BYTES_COPY_BYTES_REGISTERS

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/device_copy.cuh>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__driver/driver_api.h>
#include <cuda/__launch/configuration.h>
#include <cuda/__launch/launch.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/devices>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__exception/cuda_error.h>
#include <cuda/std/__host_stdlib/stdexcept>
#include <cuda/std/__mdspan/extents.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/remove_pointer.h>

#include <cuda/experimental/__copy/utils.cuh>
#include <cuda/experimental/__copy_bytes/copy_bytes_naive.cuh>
#include <cuda/experimental/__copy_bytes/cute_utils.cuh>
#include <cuda/experimental/__copy_bytes/layout_optimization.cuh>

#include <cute/algorithm/copy.hpp>
#include <cute/tensor_impl.hpp>
//
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief Tiled copy kernel.
//!
//! Each block processes one tile of _TileSize elements along the innermost dimension.
//! Blocks are distributed over (tiles_per_row * outer_size) tiles.
//!
//! When both tensors have compile-time stride Int<1> in mode 0 (vectorized path), each thread copies a contiguous chunk
//! of elements via cute::copy.
template <typename _Config,
          typename _IndexT,
          typename _SrcPtr,
          typename _SrcLayout,
          typename _DstPtr,
          typename _DstLayout,
          int _TileSize,
          int _VectorBits>
__global__ void __copy_bytes_kernel(
  _CCCL_GRID_CONSTANT const _Config __config,
  _CCCL_GRID_CONSTANT const _SrcPtr* const _CCCL_RESTRICT __src_ptr,
  _CCCL_GRID_CONSTANT const _SrcLayout __src_layout,
  _CCCL_GRID_CONSTANT _DstPtr* const _CCCL_RESTRICT __dst_ptr,
  _CCCL_GRID_CONSTANT const _DstLayout __dst_layout,
  _CCCL_GRID_CONSTANT const _IndexT __inner_size,
  _CCCL_GRID_CONSTANT const unsigned __tiles_per_row)
{
  const auto __src            = ::cuda::experimental::__make_gmem_tensor(__src_ptr, __src_layout);
  auto __dst                  = ::cuda::experimental::__make_gmem_tensor(__dst_ptr, __dst_layout);
  constexpr int __num_threads = ::cuda::gpu_thread.count(::cuda::block, __config);
  const auto __thread_id      = ::cuda::gpu_thread.rank_as<_IndexT>(::cuda::block, __config);
  const auto __block_id       = ::cuda::block.rank_as<int>(::cuda::grid, __config);
  const int __inner_tile      = __block_id % __tiles_per_row;
  const int __outer_idx       = __block_id / __tiles_per_row;
  const int __inner_offset    = __inner_tile * _TileSize;
  const int __flat_offset     = __inner_offset + __outer_idx * __inner_size;
  const int __remaining       = __inner_size - __inner_offset;

  if (__remaining >= _TileSize)
  {
    constexpr int __elems_per_thread = _TileSize / __num_threads;
    const auto __thr_layout          = ::cute::make_layout(::cute::Int<__elems_per_thread>{});
    const auto __thr_offset          = __flat_offset + __thread_id * __elems_per_thread;
    const auto __thr_src             = ::cuda::experimental::__make_gmem_tensor(&__src(__thr_offset), __thr_layout);
    auto __thr_dst                   = ::cuda::experimental::__make_gmem_tensor(&__dst(__thr_offset), __thr_layout);
    ::cute::copy(::cute::AutoVectorizingCopyWithAssumedAlignment<_VectorBits>{}, __thr_src, __thr_dst);
  }
  else
  {
    for (auto __i = __thread_id; __i < __remaining; __i += __num_threads)
    {
      __dst(__flat_offset + __i) = __src(__flat_offset + __i);
    }
  }
}

#if !_CCCL_COMPILER(NVRTC)

inline constexpr int __block_size = 256;

//! @brief Launch the tiled copy kernel with pre-built (recast) tensors.
//!
//! Computes tile size, inner/outer dimensions from the tensor and _VectorBits, then decomposes each CuTe tensor into
//! its raw pointer and layout.
template <int _VectorBits, typename _SrcTensor, typename _DstTensor>
_CCCL_HOST_API void
__launch_copy_bytes_kernel(const _SrcTensor& __src_tensor, const _DstTensor& __dst_tensor, ::cuda::stream_ref __stream)
{
  constexpr int __max_vector_bytes = 32;
  using ::cuda::std::size_t;
  const auto __src_ptr    = ::cute::raw_pointer_cast(__src_tensor.data());
  const auto __dst_ptr    = ::cute::raw_pointer_cast(__dst_tensor.data());
  const auto __src_layout = __src_tensor.layout();
  const auto __dst_layout = __dst_tensor.layout();

  constexpr size_t __vec_bytes   = _VectorBits / CHAR_BIT;
  constexpr size_t __tile_size   = __block_size * (__max_vector_bytes / __vec_bytes);
  const auto __inner_size        = static_cast<size_t>(::cute::size<0>(__src_tensor));
  const auto __outer_size        = static_cast<size_t>(::cute::size(__src_tensor)) / __inner_size;
  const unsigned __tiles_per_row = ::cuda::ceil_div(__inner_size, __tile_size);
  const auto __grid_size         = __tiles_per_row * __outer_size;
  const auto __config = ::cuda::make_config(::cuda::block_dims<__block_size>(), ::cuda::grid_dims(__grid_size));

  using __src_t        = ::cuda::std::remove_cv_t<::cuda::std::remove_pointer_t<decltype(__src_ptr)>>;
  using __dst_t        = ::cuda::std::remove_pointer_t<decltype(__dst_ptr)>;
  using __src_layout_t = decltype(__src_layout);
  using __dst_layout_t = decltype(__dst_layout);
  const auto __kernel  = ::cuda::experimental::
    __copy_bytes_kernel<decltype(__config), int, __src_t, __src_layout_t, __dst_t, __dst_layout_t, __tile_size, _VectorBits>;

  ::cuda::launch(
    __stream, __config, __kernel, __src_ptr, __src_layout, __dst_ptr, __dst_layout, __inner_size, __tiles_per_row);
}

template <::cuda::std::size_t _VectorBytes>
struct __vector_access
{
  using type = ::cuda::std::__make_nbit_uint_t<_VectorBytes * CHAR_BIT>;
};

// NOT SUPPORTED BY CUTE
// template <>
// struct __vector_access<32>
//{
//  using type = ::ulonglong4_32a;
//};

template <::cuda::std::size_t _VectorBytes>
using __vector_access_t = typename __vector_access<_VectorBytes>::type;

//! @brief Dispatch a vectorized copy kernel based on the common vector size in bytes.
//!
//! Recasts the source and destination tensors to the appropriate vector type
//! and launches the unified tiled copy kernel.
//!
//! @param[in] __stream              CUDA stream to launch on
//! @param[in] __src_ptr                 Source CuTe tensor
//! @param[in] __dst_ptr                 Destination CuTe tensor
//! @param[in] __common_vector_bytes Common vectorization width in bytes (1, 2, 4, 8, or 16)
template <typename _SrcTensor, typename _DstTensor>
_CCCL_HOST_API void __dispatch_vectorized_copy(
  const _SrcTensor& __src_ptr, const _DstTensor& __dst_ptr, int __common_vector_bytes, ::cuda::stream_ref __stream)
{
  auto __launch = [&](auto __vec_c) {
    constexpr int __vec_bytes    = decltype(__vec_c)::value;
    constexpr int __vec_bits_int = __vec_bytes * CHAR_BIT;
    using __vec_type             = __vector_access_t<__vec_bytes>;

    const auto __src_recast = ::cute::recast<__vec_type>(__src_ptr);
    const auto __dst_recast = ::cute::recast<__vec_type>(__dst_ptr);
    ::cuda::experimental::__launch_copy_bytes_kernel<__vec_bits_int>(__src_recast, __dst_recast, __stream);
  };
  switch (__common_vector_bytes)
  {
    // case 32:
    //   __launch(::cuda::std::integral_constant<int, 32>{});
    //   break;
    case 16:
      __launch(::cuda::std::integral_constant<int, 16>{});
      break;
    case 8:
      __launch(::cuda::std::integral_constant<int, 8>{});
      break;
    case 4:
      __launch(::cuda::std::integral_constant<int, 4>{});
      break;
    case 2:
      __launch(::cuda::std::integral_constant<int, 2>{});
      break;
    default:
      __launch(::cuda::std::integral_constant<int, 1>{});
      break;
  }
}

struct __vectorized_dispatch_tag
{
  static constexpr bool __vectorized = true;
  ::cuda::std::size_t __common_vector_bytes;
};

struct __naive_dispatch_tag
{
  static constexpr bool __vectorized = false;
};

//! @brief Recursively dispatch a rank-dependent copy on two raw tensors.
//!
//! Tries ranks `_Np`, `_Np+1`, ... up to `min(_MaxRank, 5)`. When the runtime
//! rank matches, builds CuTe layouts and dispatches via the tag-selected path:
//! - @ref __vectorized_dispatch_tag : contiguous layouts + vectorized kernel
//! - @ref __naive_dispatch_tag      : general layouts + naive kernel
//!
//! @tparam _Np   Starting candidate rank
//! @param[in] __tag    Dispatch tag (carries extra parameters for the vectorized path)
//! @param[in] __src    Source raw tensor
//! @param[in] __dst    Destination raw tensor
//! @param[in] __stream CUDA stream
//! @return `true` if the rank matched and the copy was dispatched, `false` if rank exceeded the limit
template <::cuda::std::size_t _Np,
          typename _Tag,
          typename _Ep,
          typename _Sp,
          typename _TpSrc,
          typename _TpDst,
          ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API bool __dispatch_by_rank(
  const _Tag& __tag,
  const __raw_tensor<_Ep, _Sp, _TpSrc, _MaxRank>& __src,
  const __raw_tensor<_Ep, _Sp, _TpDst, _MaxRank>& __dst,
  ::cuda::stream_ref __stream)
{
  constexpr auto __max_dispatch_rank = ::cuda::std::min(_MaxRank, ::cuda::std::size_t{5});
  if constexpr (_Np > __max_dispatch_rank)
  {
    return false;
  }
  else
  {
    if (__src.__rank == _Np)
    {
      constexpr auto __seq = ::cuda::std::make_index_sequence<_Np - 1>{};
      if constexpr (_Tag::__vectorized)
      {
        const auto __src_layout = ::cuda::experimental::__to_cute_layout_contiguous(__src, __seq);
        const auto __dst_layout = ::cuda::experimental::__to_cute_layout_contiguous(__dst, __seq);
        ::cuda::experimental::__dispatch_vectorized_copy(
          ::cuda::experimental::__make_gmem_tensor(__src.__data, __src_layout),
          ::cuda::experimental::__make_gmem_tensor(__dst.__data, __dst_layout),
          __tag.__common_vector_bytes,
          __stream);
      }
      else
      {
        const auto __src_layout = ::cuda::experimental::__to_cute_layout(__src, __seq);
        const auto __dst_layout = ::cuda::experimental::__to_cute_layout(__dst, __seq);
        ::cuda::experimental::copy_bytes_naive(__src.__data, __src_layout, __dst.__data, __dst_layout, __stream);
      }
      return true;
    }
    return ::cuda::experimental::__dispatch_by_rank<_Np + 1>(__tag, __src, __dst, __stream);
  }
}

//! @brief Register-based copy using cute::copy.
//!
//! Preprocesses both layouts to determine the optimal copy strategy:
//!
//! 1. Sort both layouts by dst's ascending stride (common permutation).
//! 2. If both have stride-1 in mode 0 (vectorized path):
//!    - Compute the contiguous extent for each tensor, use the minimum as inner_size to avoid crossing mode boundaries.
//!    - Compute the maximum compatible vectorization width and recast.
//!    - Launch the unified kernel (cute::copy auto-vectorizes).
//! 3. Fallback: copy_bytes_naive for non-vectorizable or unsupported configurations.
template <typename _Tp, typename _SrcLayout, typename _DstLayout>
_CCCL_HOST_API void copy_bytes_registers(
  const _Tp* __src_ptr,
  const _SrcLayout& __src_layout,
  _Tp* __dst_ptr,
  const _DstLayout& __dst_layout,
  ::cuda::stream_ref __stream)
{
  namespace cudax          = cuda::experimental;
  constexpr int __src_rank = decltype(::cute::rank(__src_layout))::value;
  constexpr int __dst_rank = decltype(::cute::rank(__dst_layout))::value;
  static_assert(__src_rank == __dst_rank, "Source and destination layouts must have the same rank");
  auto __src_raw = cudax::__to_raw_tensor<__src_rank>(__src_ptr, __src_layout, cudax::__remove_extent1_mode);
  auto __dst_raw = cudax::__to_raw_tensor<__dst_rank>(__dst_ptr, __dst_layout, cudax::__remove_extent1_mode);
  _CCCL_ASSERT(__src_raw.__rank == __dst_raw.__rank, "Source and destination layouts must have the same rank");
  if (__src_raw.__extents != __dst_raw.__extents)
  {
    _CCCL_THROW(::std::invalid_argument, "Source and destination layouts must have the same extents");
  }
  const auto __total_size = static_cast<::cuda::std::size_t>(::cute::size(__src_layout));
  if (__total_size == 0)
  {
    return;
  }
  if (__total_size == 1)
  {
    ::cuda::__driver::__memcpyAsync(__dst_ptr, __src_ptr, sizeof(_Tp), __stream.get());
    return;
  }
  // NOTE: source and destination extents are identical
  //       Sort both by dst's ascending absolute stride (common permutation).
  //       After this, dst has stride-1 in mode 0 (if any mode is stride-1).
  //       Shapes are kept in sync (both tensors share the same shape because they are ordered by the same permutation).
  cudax::__sort_by_stride_paired(__src_raw, __dst_raw);
  // Flip modes where both strides are negative to positive, enabling coalescing and vectorization.
  cudax::__flip_negative_strides_paired(__src_raw, __dst_raw);
  // Merge adjacent modes that are contiguous in both tensors, reducing effective rank.
  cudax::__coalesce_paired(__src_raw, __dst_raw);

  const bool __are_both_contiguous = (__src_raw.__strides[0] == 1) && (__dst_raw.__strides[0] == 1);
  if (__are_both_contiguous /*&& ::std::is_same_v<_Tp, _Up>*/)
  {
    if (__src_raw.__rank == 1)
    {
      using __extents_t       = ::cuda::std::dims<1>;
      const auto __src_mdspan = ::cuda::std::mdspan<const _Tp, __extents_t>(__src_raw.__data, __src_raw.__extents[0]);
      const auto __dst_mdspan = ::cuda::std::mdspan<_Tp, __extents_t>(__dst_raw.__data, __dst_raw.__extents[0]);
      const auto __status     = ::cub::detail::copy_mdspan::copy(__src_mdspan, __dst_mdspan, __stream.get());
      if (__status != ::cudaSuccess)
      {
        _CCCL_THROW(::cuda::cuda_error, __status, "cub::DeviceCopy::Copy failed");
      }
      return;
    }
    const auto __src_vector_bytes        = cudax::__max_vector_size_bytes(__src_raw);
    const auto __dst_vector_bytes        = cudax::__max_vector_size_bytes(__dst_raw);
    const auto __dev_id                  = ::cuda::__driver::__cudevice_to_ordinal(::cuda::__driver::__ctxGetDevice());
    const auto __dev                     = ::cuda::devices[__dev_id];
    const auto __major                   = __dev.attribute<::cudaDevAttrComputeCapabilityMajor>();
    const size_t __max_access_size_bytes = (__major >= 9) ? 32 : 16;
    const auto __common_vector_bytes =
      ::cuda::std::min({__src_vector_bytes, __dst_vector_bytes, __max_access_size_bytes});
    const auto __vector_tag = cudax::__vectorized_dispatch_tag{__common_vector_bytes};
    if (__common_vector_bytes > sizeof(_Tp)
        && cudax::__dispatch_by_rank<2>(__vector_tag, __src_raw, __dst_raw, __stream))
    {
      return;
    }
  }
  // transpose case
  // else if (__dst_raw.__strides[0] == 1)
  //{
  //  cudax::copy_bytes_shared_mem(__src_raw, __dst_raw, __stream);
  //  return;
  //}
  if (!cudax::__dispatch_by_rank<1>(cudax::__naive_dispatch_tag{}, __src_raw, __dst_raw, __stream))
  {
    cudax::copy_bytes_naive(__src_ptr, __src_layout, __dst_ptr, __dst_layout, __stream);
  }
}

#endif // !_CCCL_COMPILER(NVRTC)
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__COPY_BYTES_COPY_BYTES_REGISTERS
