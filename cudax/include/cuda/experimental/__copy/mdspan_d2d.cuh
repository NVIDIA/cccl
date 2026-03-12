//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_COPY_MDSPAN_D2D_H
#define __CUDAX_COPY_MDSPAN_D2D_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cub/device/device_transform.cuh>

#  include <cuda/__driver/driver_api.h>
#  include <cuda/__functional/address_stability.h>
#  include <cuda/__mdspan/host_device_mdspan.h>
#  include <cuda/__mdspan/traits.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__algorithm/max.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__host_stdlib/stdexcept>
#  include <cuda/std/__memory/is_sufficiently_aligned.h>
#  include <cuda/std/__type_traits/common_type.h>
#  include <cuda/std/__type_traits/is_convertible.h>
#  include <cuda/std/__type_traits/is_same.h>

#  include <cuda/experimental/__copy/copy_optimized.cuh>
#  include <cuda/experimental/__copy/tensor_copy_utils.cuh>
#  include <cuda/experimental/__copy/vector_access.cuh>
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
_CCCL_HOST_API void copy(::cuda::device_mdspan<_TpIn, _ExtentsIn, _LayoutPolicyIn, _AccessorPolicyIn> __src,
                         ::cuda::device_mdspan<_TpOut, _ExtentsOut, _LayoutPolicyOut, _AccessorPolicyOut> __dst,
                         ::cuda::stream_ref __stream)
{
  namespace cudax = ::cuda::experimental;
  static_assert(::cuda::std::is_convertible_v<_TpIn, _TpOut>, "TpIn must be convertible to TpOut");
  static_assert(!::cuda::std::is_const_v<_TpOut>, "TpOut must not be const");
  static_assert(::cuda::__is_cuda_mdspan_layout_v<_LayoutPolicyIn>,
                "LayoutPolicyIn must be a predefined layout policy");
  static_assert(::cuda::__is_cuda_mdspan_layout_v<_LayoutPolicyOut>,
                "LayoutPolicyOut must be a predefined layout policy");

  if (__src.size() != __dst.size())
  {
    _CCCL_THROW(::std::invalid_argument, "mdspans must have the same size");
  }
  const auto __tensor_size = __src.size();
  if (__tensor_size == 0)
  {
    return;
  }
  if (__src.data_handle() == nullptr || __dst.data_handle() == nullptr)
  {
    _CCCL_THROW(::std::invalid_argument, "mdspan data handle must not be nullptr");
  }
  if (!::cuda::std::is_sufficiently_aligned<alignof(_TpIn)>(__src.data_handle()))
  {
    _CCCL_THROW(::std::invalid_argument, "source mdspan must be sufficiently aligned");
  }
  if (!::cuda::std::is_sufficiently_aligned<alignof(_TpOut)>(__dst.data_handle()))
  {
    _CCCL_THROW(::std::invalid_argument, "destination mdspan must be sufficiently aligned");
  }
  if (cudax::__has_interleaved_stride_order(__dst))
  {
    _CCCL_THROW(::std::invalid_argument, "destination mdspan must not have interleaved stride order");
  }
  if (cudax::__may_overlap(__src, __dst))
  {
    _CCCL_THROW(::std::invalid_argument, "mdspans must not overlap in memory");
  }
  if (__tensor_size == 1)
  {
    ::cuda::__driver::__memcpyAsync(__dst.data_handle(), __src.data_handle(), sizeof(_TpIn), __stream.get());
    return;
  }
  if constexpr (_ExtentsIn::rank() > 0 && _ExtentsOut::rank() > 0)
  {
    constexpr auto __are_accessors_default_convertible =
      ::cuda::std::is_convertible_v<_AccessorPolicyIn, ::cuda::std::default_accessor<_TpIn>>
      && ::cuda::std::is_convertible_v<_AccessorPolicyOut, ::cuda::std::default_accessor<_TpOut>>;
    constexpr auto __have_same_type =
      ::cuda::std::is_same_v<::cuda::std::remove_cvref_t<_TpIn>, ::cuda::std::remove_cvref_t<_TpOut>>;

    using __extent_t = ::cuda::std::common_type_t<typename _ExtentsIn::index_type, typename _ExtentsOut::index_type>;
    using __stride_t = ::cuda::std::common_type_t<cudax::__mdspan_stride_t<_ExtentsIn, _LayoutPolicyIn>,
                                                  cudax::__mdspan_stride_t<_ExtentsOut, _LayoutPolicyOut>>;
    constexpr auto __max_rank = ::cuda::std::max(_ExtentsIn::rank(), _ExtentsOut::rank());
    const auto __src_raw      = cudax::__to_raw_tensor<__extent_t, __stride_t, __max_rank>(__src);
    const auto __dst_raw      = cudax::__to_raw_tensor<__extent_t, __stride_t, __max_rank>(__dst);
    if (!cudax::__same_extents(__src_raw, __dst_raw))
    {
      _CCCL_THROW(::std::invalid_argument, "mdspans must have the same extents (after removing singleton dimensions)");
    }
    auto __src_simplified = __src_raw;
    auto __dst_simplified = __dst_raw;
    cudax::__sort_by_stride_paired(__src_simplified, __dst_simplified);
    cudax::__flip_negative_strides_paired(__src_simplified, __dst_simplified);
    cudax::__coalesce_paired(__src_simplified, __dst_simplified);
    using __unsigned_extent_t = typename decltype(__src_simplified)::__unsigned_extent_t;
    const bool __both_stride1 = (__src_simplified.__strides[0] == 1) && (__dst_simplified.__strides[0] == 1);
    const __unsigned_extent_t __tile_size = __both_stride1 ? __src_simplified.__extents[0] : __unsigned_extent_t{1};
    const auto __src_normalized           = (__tile_size > 1) ? __src_simplified : cudax::__reverse_modes(__src_raw);
    const auto __dst_normalized           = (__tile_size > 1) ? __dst_simplified : cudax::__reverse_modes(__dst_raw);

    _CCCL_ASSERT(__tensor_size % __tile_size == 0, "tensor size must be divisible by tile size");
    // (1) contiguous case
    if (__tile_size == __tensor_size)
    {
      _CCCL_TRY_CUDA_API(
        cub::DeviceTransform::Transform,
        "cub::DeviceTransform::Transform failed",
        __src.data_handle(),
        __dst.data_handle(),
        __tensor_size,
        ::cuda::proclaim_copyable_arguments(::cuda::std::identity{}),
        __stream.get());
    }
    // (2) vectorized case
    if constexpr (__have_same_type && sizeof(_TpIn) <= __max_vector_access && __are_accessors_default_convertible)
    {
      if (__both_stride1)
      {
        using ::cuda::std::size_t;
        const auto __src_alignment            = cudax::__max_alignment(__src_normalized);
        const auto __dst_alignment            = cudax::__max_alignment(__dst_normalized);
        const auto __max_gpu_arch_vector_size = cudax::__max_gpu_arch_vector_size();
        const auto __vector_size_bytes =
          ::cuda::std::min({__src_alignment, __dst_alignment, __max_gpu_arch_vector_size});

        const auto __inner_extent_bytes  = static_cast<size_t>(__src_normalized.__extents[0]) * sizeof(_TpIn);
        const auto __inner_extent_vector = __inner_extent_bytes / __vector_size_bytes;

        if (__inner_extent_vector >= cudax::__tile_size(__vector_size_bytes))
        {
          const auto __vector_tag = cudax::__vectorized_dispatch_tag{__common_alignment};
          if (cudax::__dispatch_by_rank<2>(__vector_tag, __src_normalized, __dst_normalized, __stream))
          {
            return;
          }
        }
        else
        {
          cudax::__copy_vectorized_dispatch(__src_normalized, __dst_normalized, __vector_size_bytes, __stream);
        }
      }
    }
    // (3) transpose case
    if (cudax::__use_shared_mem_kernel(__src_normalized, __dst_normalized))
    {
      cudax::copy_bytes_shared_mem(__src_normalized, __dst_normalized, __stream);
    }
    // (4) fallback case
    else
    {
      cudax::__copy_optimized(__src_normalized, __dst_normalized, __stream, __src.accessor(), __dst.accessor());
    }
  }
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // __CUDAX_COPY_MDSPAN_D2D_H
