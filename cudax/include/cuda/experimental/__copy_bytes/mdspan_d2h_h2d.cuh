//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_COPY_MDSPAN_D2H_H2D_H
#define __CUDAX_COPY_MDSPAN_D2H_H2D_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/__driver/driver_api.h>
#  include <cuda/__mdspan/host_device_mdspan.h>
#  include <cuda/__mdspan/traits.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__algorithm/max.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__host_stdlib/stdexcept>
#  include <cuda/std/__mdspan/default_accessor.h>
#  include <cuda/std/__mdspan/mdspan.h>
#  include <cuda/std/__memory/is_sufficiently_aligned.h>
#  include <cuda/std/__type_traits/common_type.h>
#  include <cuda/std/__type_traits/is_const.h>
#  include <cuda/std/__type_traits/is_convertible.h>
#  include <cuda/std/__type_traits/is_same.h>
#  include <cuda/std/__type_traits/is_trivially_copyable.h>
#  include <cuda/std/__type_traits/remove_cv.h>

#  include <cuda/experimental/__copy_bytes/memcpy_batch_tiles.cuh>
#  include <cuda/experimental/__copy_bytes/simplify_paired.cuh>
#  include <cuda/experimental/__copy_bytes/tensor_query.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief Internal implementation of @ref copy_bytes for host/device mdspan transfers.
//!
//! Validates preconditions, converts mdspans to raw tensor descriptors, simplifies the paired layout
//! (sort, flip negative strides, coalesce), then dispatches a batched asynchronous memcpy.
//!
//! @param[in]  __src       Source mdspan
//! @param[out] __dst       Destination mdspan
//! @param[in]  __direction Copy direction (host-to-device or device-to-host)
//! @param[in]  __stream    CUDA stream for the asynchronous transfer
template <typename _TpIn,
          typename _ExtentsIn,
          typename _LayoutPolicyIn,
          typename _AccessorPolicyIn,
          typename _TpOut,
          typename _ExtentsOut,
          typename _LayoutPolicyOut,
          typename _AccessorPolicyOut>
_CCCL_HOST_API void __copy_bytes_impl(
  ::cuda::std::mdspan<_TpIn, _ExtentsIn, _LayoutPolicyIn, _AccessorPolicyIn> __src,
  ::cuda::std::mdspan<_TpOut, _ExtentsOut, _LayoutPolicyOut, _AccessorPolicyOut> __dst,
  [[maybe_unused]] __copy_direction __direction,
  ::cuda::stream_ref __stream)
{
  namespace cudax = ::cuda::experimental;
  static_assert(::cuda::std::is_same_v<::cuda::std::remove_cv_t<_TpIn>, ::cuda::std::remove_cv_t<_TpOut>>,
                "cudax::copy_bytes: TpIn and TpOut must be the same type");
  static_assert(::cuda::std::is_trivially_copyable_v<_TpIn>, "TpIn must be trivially copyable");
  static_assert(!::cuda::std::is_const_v<_TpOut>, "TpOut must not be const");
  static_assert(::cuda::__is_cuda_mdspan_layout_v<_LayoutPolicyIn>,
                "cudax::copy_bytes: LayoutPolicyIn must be a predefined layout policy");
  static_assert(::cuda::__is_cuda_mdspan_layout_v<_LayoutPolicyOut>,
                "cudax::copy_bytes: LayoutPolicyOut must be a predefined layout policy");
  using __default_accessor_in  = ::cuda::std::default_accessor<_TpIn>;
  using __default_accessor_out = ::cuda::std::default_accessor<_TpOut>;
  static_assert(::cuda::std::is_convertible_v<_AccessorPolicyIn, __default_accessor_in>,
                "cudax::copy_bytes: AccessorPolicyIn must be convertible to cuda::std::default_accessor");
  static_assert(::cuda::std::is_convertible_v<_AccessorPolicyOut, __default_accessor_out>,
                "cudax::copy_bytes: AccessorPolicyOut must be convertible to cuda::std::default_accessor");
  if (__stream.get() == nullptr)
  {
    _CCCL_THROW(::std::invalid_argument, "cudax::copy_bytes: stream must not be nullptr");
  }
  if (__src.size() != __dst.size())
  {
    _CCCL_THROW(::std::invalid_argument, "cudax::copy_bytes: mdspans must have the same size");
  }

  const auto __tensor_size = __src.size();
  if (__tensor_size == 0)
  {
    return;
  }
  if (__src.data_handle() == nullptr || __dst.data_handle() == nullptr)
  {
    _CCCL_THROW(::std::invalid_argument, "cudax::copy_bytes: mdspan data handle must not be nullptr");
  }
  if (!::cuda::std::is_sufficiently_aligned<alignof(_TpIn)>(__src.data_handle()))
  {
    _CCCL_THROW(::std::invalid_argument, "cudax::copy_bytes: source mdspan must be sufficiently aligned");
  }
  if (!::cuda::std::is_sufficiently_aligned<alignof(_TpOut)>(__dst.data_handle()))
  {
    _CCCL_THROW(::std::invalid_argument, "cudax::copy_bytes: destination mdspan must be sufficiently aligned");
  }
  if (cudax::__has_interleaved_stride_order(__dst))
  {
    _CCCL_THROW(::std::invalid_argument,
                "cudax::copy_bytes: destination mdspan must not have interleaved stride order");
  }

  if (__tensor_size == 1) // rank == 0 also falls into this case
  {
    auto __src_ptr = __src.data_handle();
    auto __dst_ptr = __dst.data_handle();
    if constexpr (::cuda::__is_layout_stride_relaxed_v<_LayoutPolicyIn>)
    {
      __src_ptr += __src.mapping().offset();
    }
    if constexpr (::cuda::__is_layout_stride_relaxed_v<_LayoutPolicyOut>)
    {
      __dst_ptr += __dst.mapping().offset();
    }
    ::cuda::__driver::__memcpyAsync(__dst_ptr, __src_ptr, sizeof(_TpIn), __stream.get());
    return;
  }
  if constexpr (_ExtentsIn::rank() > 0 && _ExtentsOut::rank() > 0)
  {
    using __extent_t = ::cuda::std::common_type_t<typename _ExtentsIn::index_type, typename _ExtentsOut::index_type>;
    using __stride_t =
      ::cuda::std::common_type_t<cudax::__mdspan_stride_t<_LayoutPolicyIn, decltype(__src.mapping())>,
                                 cudax::__mdspan_stride_t<_LayoutPolicyOut, decltype(__dst.mapping())>>;
    constexpr auto __max_rank = ::cuda::std::max(_ExtentsIn::rank(), _ExtentsOut::rank());
    const auto __src_raw      = cudax::__to_raw_tensor<__extent_t, __stride_t, __max_rank>(__src);
    const auto __dst_raw      = cudax::__to_raw_tensor<__extent_t, __stride_t, __max_rank>(__dst);
    if (!cudax::__same_extents(__src_raw, __dst_raw))
    {
      _CCCL_THROW(::std::invalid_argument,
                  "cudax::copy_bytes: mdspans must have the same extents (after removing singleton dimensions)");
    }

    auto __src_simplified = __src_raw;
    auto __dst_simplified = __dst_raw;
    cudax::__sort_by_stride_paired(__src_simplified, __dst_simplified);
    cudax::__flip_negative_strides_paired(__src_simplified, __dst_simplified);
    cudax::__coalesce_paired(__src_simplified, __dst_simplified);

    const bool __both_stride1    = (__src_simplified.__strides[0] == 1) && (__dst_simplified.__strides[0] == 1);
    const __extent_t __tile_size = __both_stride1 ? __src_simplified.__extents[0] : __extent_t{1};
    const auto __src_iter        = (__tile_size > 1) ? __src_simplified : cudax::__reverse_modes(__src_raw);
    const auto __dst_iter        = (__tile_size > 1) ? __dst_simplified : cudax::__reverse_modes(__dst_raw);

    const auto __num_tiles  = __tensor_size / __tile_size;
    const auto __copy_bytes = __tile_size * sizeof(_TpIn);
    _CCCL_ASSERT(__tensor_size % __tile_size == 0, "cudax::copy_bytes: tensor size must be divisible by tile size");
    __tile_iterator_linearized<__extent_t, __stride_t, _TpIn, __max_rank> __src_tiles_iterator(__src_iter, __tile_size);
    __tile_iterator_linearized<__extent_t, __stride_t, _TpOut, __max_rank> __dst_tiles_iterator(__dst_iter, __tile_size);

    cudax::__memcpy_batch_tiles(
      __src_tiles_iterator,
      __dst_tiles_iterator,
      __num_tiles,
      __copy_bytes,
      __direction,
      __src.data_handle(),
      __dst.data_handle(),
      __stream);
  }
}

/***********************************************************************************************************************
 * Public API
 **********************************************************************************************************************/

//! @rst
//! .. _cudax-copy-bytes:
//!
//! Asynchronous byte-wise mdspan copy
//! ------------------------------------
//!
//! ``copy_bytes`` asynchronously copies elements between a host ``mdspan`` and a device ``mdspan`` on the given
//! CUDA stream. Two overloads are provided: host-to-device and device-to-host.
//!
//! - Source and destination must have the same total number of elements and identical extents
//!   (after removing extent-1 dimensions).
//! - The implementation supports any stride value independently for source and destination mdspans.
//! - Element types must be trivially copyable and (ignoring cv-qualification) the same type.
//! - Layout policies must be one of the predefined ``cuda::std`` layout policies
//!   (``layout_right``, ``layout_left``, ``layout_stride``) or ``cuda::layout_stride_relaxed``.
//! - Accessor policies must be convertible to ``cuda::std::default_accessor``.
//! - The destination must not have an interleaved stride order.
//!
//! The implementation is optimized to maximize the contiguous memory regions to copy and relies on batched asynchronous
//! memcpy.
//!
//! .. code-block:: c++
//!
//!    #include <cuda/experimental/copy_bytes.cuh>
//!
//!      using extents_t = cuda::std::dims<2>;
//!      cuda::host_mdspan<const float, extents_t> src(src_ptr, extents);
//!      cuda::device_mdspan<float, extents_t>     dst(dst_ptr, extents);
//!      cuda::experimental::copy_bytes(src, dst, stream);
//!
//! @endrst
//! @param[in] __src Source mdspan
//! @param[out] __dst Destination mdspan
//! @param[in] __stream CUDA stream for the asynchronous transfer
template <typename _TpIn,
          typename _ExtentsIn,
          typename _LayoutPolicyIn,
          typename _AccessorPolicyIn,
          typename _TpOut,
          typename _ExtentsOut,
          typename _LayoutPolicyOut,
          typename _AccessorPolicyOut>
_CCCL_HOST_API void copy_bytes(::cuda::host_mdspan<_TpIn, _ExtentsIn, _LayoutPolicyIn, _AccessorPolicyIn> __src,
                               ::cuda::device_mdspan<_TpOut, _ExtentsOut, _LayoutPolicyOut, _AccessorPolicyOut> __dst,
                               ::cuda::stream_ref __stream)
{
  using __src_type = ::cuda::std::mdspan<_TpIn, _ExtentsIn, _LayoutPolicyIn, _AccessorPolicyIn>;
  using __dst_type = ::cuda::std::mdspan<_TpOut, _ExtentsOut, _LayoutPolicyOut, _AccessorPolicyOut>;
  ::cuda::experimental::__copy_bytes_impl(
    static_cast<__src_type>(__src), static_cast<__dst_type>(__dst), __copy_direction::__host_to_device, __stream);
}

//! @brief Asynchronously copies bytes from a device mdspan to a host mdspan.
//!
//! @param[in] __src Source device mdspan
//! @param[out] __dst Destination host mdspan
//! @param[in] __stream CUDA stream for the asynchronous transfer
template <typename _TpIn,
          typename _ExtentsIn,
          typename _LayoutPolicyIn,
          typename _AccessorPolicyIn,
          typename _TpOut,
          typename _ExtentsOut,
          typename _LayoutPolicyOut,
          typename _AccessorPolicyOut>
_CCCL_HOST_API void copy_bytes(::cuda::device_mdspan<_TpIn, _ExtentsIn, _LayoutPolicyIn, _AccessorPolicyIn> __src,
                               ::cuda::host_mdspan<_TpOut, _ExtentsOut, _LayoutPolicyOut, _AccessorPolicyOut> __dst,
                               ::cuda::stream_ref __stream)
{
  using __src_type = ::cuda::std::mdspan<_TpIn, _ExtentsIn, _LayoutPolicyIn, _AccessorPolicyIn>;
  using __dst_type = ::cuda::std::mdspan<_TpOut, _ExtentsOut, _LayoutPolicyOut, _AccessorPolicyOut>;
  ::cuda::experimental::__copy_bytes_impl(
    static_cast<__src_type>(__src), static_cast<__dst_type>(__dst), __copy_direction::__device_to_host, __stream);
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // __CUDAX_COPY_MDSPAN_D2H_H2D_H
