//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__COPY_MDSPAN_D2D_H
#define _CUDAX__COPY_MDSPAN_D2D_H

#include <cuda/std/detail/__config>

#include <cuda/std/__type_traits/remove_cv.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cub/device/device_transform.cuh>

#  include <cuda/__cmath/pow2.h>
#  include <cuda/__driver/driver_api.h>
#  include <cuda/__functional/address_stability.h>
#  include <cuda/__mdspan/host_device_mdspan.h>
#  include <cuda/__mdspan/traits.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/__type_traits/is_trivially_copyable.h>
#  include <cuda/std/__algorithm/max.h>
#  include <cuda/std/__functional/identity.h>
#  include <cuda/std/__host_stdlib/stdexcept>
#  include <cuda/std/__mdspan/default_accessor.h>
#  include <cuda/std/__memory/is_sufficiently_aligned.h>
#  include <cuda/std/__type_traits/common_type.h>
#  include <cuda/std/__type_traits/conditional.h>
#  include <cuda/std/__type_traits/is_const.h>
#  include <cuda/std/__type_traits/is_convertible.h>
#  include <cuda/std/__type_traits/is_same.h>
#  include <cuda/std/__type_traits/is_signed.h>

#  include <cuda/experimental/__copy/copy_contiguous.cuh>
#  include <cuda/experimental/__copy/copy_dst_contiguous.cuh>
#  include <cuda/experimental/__copy/copy_optimized.cuh>
#  include <cuda/experimental/__copy/copy_shared_memory.cuh>
#  include <cuda/experimental/__copy/dispatch_by_vector.cuh>
#  include <cuda/experimental/__copy/tensor_copy_utils.cuh>
#  include <cuda/experimental/__copy/vector_access.cuh>
#  include <cuda/experimental/__copy_bytes/simplify_paired.cuh>
#  include <cuda/experimental/__copy_bytes/tensor_query.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief Launch a copy using the simplified tensor pair's exact static rank.
//!
//! The shared-memory path is reconsidered with the narrower descriptors because the original mdspan rank may exceed
//! the shared-memory kernel rank limit.
//!
//! @param[in]  __src          Simplified source raw tensor
//! @param[out] __dst          Simplified destination raw tensor
//! @param[in]  __stream       CUDA stream for the asynchronous transfer
//! @param[in]  __src_accessor Accessor for reading source elements
//! @param[in]  __dst_accessor Accessor for writing destination elements
template <::cuda::std::size_t _RankOut,
          typename _ExtentT,
          typename _StrideTIn,
          typename _StrideTOut,
          typename _TpIn,
          typename _TpOut,
          ::cuda::std::size_t _MaxRank,
          typename _SrcAccessor,
          typename _DstAccessor>
_CCCL_HOST_API void __copy_simplified_rank(
  const __raw_tensor<_ExtentT, _StrideTIn, _TpIn, _MaxRank>& __src,
  const __raw_tensor<_ExtentT, _StrideTOut, _TpOut, _MaxRank>& __dst,
  ::cuda::stream_ref __stream,
  const _SrcAccessor& __src_accessor,
  const _DstAccessor& __dst_accessor) noexcept
{
  // create a copy of the tensors with compile-time rank
  const auto __src_narrow = ::cuda::experimental::__narrow_raw_tensor_rank<_RankOut>(__src);
  const auto __dst_narrow = ::cuda::experimental::__narrow_raw_tensor_rank<_RankOut>(__dst);
  if constexpr (_RankOut >= 2)
  {
    // if source and destination tensors are not both contiguous, try to use the destination tensor as contiguous
    if constexpr (sizeof(_TpIn) <= 2 && sizeof(_TpOut) <= 2)
    {
      auto __src_dst_ordered = __src_narrow;
      auto __dst_dst_ordered = __dst_narrow;
      ::cuda::experimental::__sort_by_stride_paired(__dst_dst_ordered, __src_dst_ordered);
      ::cuda::experimental::__flip_negative_strides_paired(__dst_dst_ordered, __src_dst_ordered);
      // destination tensor is contiguous
      if (::cuda::experimental::__num_contiguous_dimensions(__dst_dst_ordered) == __dst_dst_ordered.__rank)
      {
        ::cuda::experimental::__copy_dst_contiguous(
          __src_dst_ordered,
          __dst_dst_ordered,
          ::cuda::experimental::__total_size(__src_dst_ordered),
          __stream,
          __src_accessor,
          __dst_accessor);
        return;
      }
    }

    if (::cuda::experimental::__use_shared_mem_kernel(__src_narrow, __dst_narrow))
    {
      ::cuda::experimental::__launch_copy_shared_mem_kernel(
        __src_narrow, __dst_narrow, __stream, __src_accessor, __dst_accessor);
      return;
    }
  }
  ::cuda::experimental::__copy_optimized(
    __src_narrow,
    __dst_narrow,
    ::cuda::experimental::__total_size(__src_narrow),
    __stream,
    __src_accessor,
    __dst_accessor);
}

//! @brief Copy elements between two device mdspans.
//!
//! Validates preconditions, converts mdspans to raw tensor descriptors, simplifies the paired layout
//! (sort, flip negative strides, coalesce), then dispatches either a vectorized contiguous kernel or a
//! strided element-wise kernel.
//!
//! @param[in]  __src    Source device mdspan
//! @param[out] __dst    Destination device mdspan
//! @param[in]  __stream CUDA stream for the asynchronous transfer
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

  using __default_accessor_in  = ::cuda::std::default_accessor<_TpIn>;
  using __default_accessor_out = ::cuda::std::default_accessor<_TpOut>;
  constexpr bool __have_default_accessors =
    ::cuda::std::is_convertible_v<_AccessorPolicyIn, __default_accessor_in>
    && ::cuda::std::is_convertible_v<_AccessorPolicyOut, __default_accessor_out>;
  constexpr bool __are_byte_copyable =
    ::cuda::std::is_same_v<::cuda::std::remove_cv_t<_TpIn>, ::cuda::std::remove_cv_t<_TpOut>>
    && ::cuda::is_trivially_copyable_v<_TpIn> //
    && __have_default_accessors;

  if (__tensor_size == 1 && __are_byte_copyable)
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

  // rank == 0 for both tensors is already handled above -> their size is exactly 1
  if constexpr (_ExtentsIn::rank() > 0 && _ExtentsOut::rank() > 0)
  {
    // use the most efficient type for device code
    using __src_extent_t = ::cuda::std::common_type_t<typename _ExtentsIn::index_type, int>;
    using __dst_extent_t = ::cuda::std::common_type_t<typename _ExtentsOut::index_type, int>;
    using __common_extent_t =
      ::cuda::std::conditional_t<(sizeof(__src_extent_t) < sizeof(__dst_extent_t)), __src_extent_t, __dst_extent_t>;
    using __src_stride_t =
      ::cuda::std::common_type_t<cudax::__mdspan_stride_t<_LayoutPolicyIn, decltype(__src.mapping())>, int>;
    using __dst_stride_t =
      ::cuda::std::common_type_t<cudax::__mdspan_stride_t<_LayoutPolicyOut, decltype(__dst.mapping())>, int>;

    constexpr auto __max_rank = ::cuda::std::max(_ExtentsIn::rank(), _ExtentsOut::rank());
    const auto __src_raw      = cudax::__to_raw_tensor<__common_extent_t, __src_stride_t, __max_rank>(__src);
    const auto __dst_raw      = cudax::__to_raw_tensor<__common_extent_t, __dst_stride_t, __max_rank>(__dst);
    if (!cudax::__same_extents(__src_raw, __dst_raw))
    {
      _CCCL_THROW(::std::invalid_argument, "mdspans must have the same extents (after removing singleton dimensions)");
    }

    auto __src_simplified = __src_raw;
    auto __dst_simplified = __dst_raw;
    cudax::__sort_by_stride_paired(__src_simplified, __dst_simplified);
    cudax::__flip_negative_strides_paired(__src_simplified, __dst_simplified);
    cudax::__coalesce_paired(__src_simplified, __dst_simplified);
    const bool __both_stride1 = (__src_simplified.__strides[0] == 1) && (__dst_simplified.__strides[0] == 1);
    const auto __tile_size    = __both_stride1 ? __src_simplified.__extents[0] : 1;
    auto __src_normalized     = (__tile_size > 1) ? __src_simplified : cudax::__reverse_modes(__src_raw);
    auto __dst_normalized     = (__tile_size > 1) ? __dst_simplified : cudax::__reverse_modes(__dst_raw);

    // If the source tensor is not contiguous, check if the destination tensor is contiguous and use it instead.
    // Prefer the destination tensor in this case to improve coalescing.
    if (__src_simplified.__strides[0] != 1)
    {
      auto __src_dst_ordered = __src_raw;
      auto __dst_dst_ordered = __dst_raw;
      cudax::__sort_by_stride_paired(__dst_dst_ordered, __src_dst_ordered);
      cudax::__flip_negative_strides_paired(__dst_dst_ordered, __src_dst_ordered);
      cudax::__coalesce_paired(__dst_dst_ordered, __src_dst_ordered);
      if (__dst_dst_ordered.__strides[0] == 1)
      {
        __src_normalized = __src_dst_ordered;
        __dst_normalized = __dst_dst_ordered;
      }
    }

    _CCCL_ASSERT(__tensor_size % __tile_size == 0, "tensor size must be divisible by tile size");
    const auto __inner_extent_bytes = __src_normalized.__extents[0] * sizeof(_TpIn);

    // check the preconditions for the vectorized case
    constexpr bool __are_vectorizable_copy =
      sizeof(_TpIn) <= __max_vector_access && ::cuda::is_power_of_two(sizeof(_TpIn)) && __are_byte_copyable;

    // (1) contiguous case
    if constexpr (__have_default_accessors)
    {
      if (static_cast<::cuda::std::size_t>(__tile_size) == __tensor_size)
      {
        _CCCL_TRY_CUDA_API(
          CUB_NS_QUALIFIER::DeviceTransform::Transform,
          "cub::DeviceTransform::Transform failed",
          __src_simplified.__data,
          __dst_simplified.__data,
          __tensor_size,
          ::cuda::proclaim_copyable_arguments(::cuda::std::identity{}),
          __stream.get());
        return;
      }
    }

    // (2) both contiguous and inner size is large
    if (__both_stride1 && __inner_extent_bytes >= cudax::__bytes_in_flight())
    {
      // (2a) vectorized case
      if constexpr (__are_vectorizable_copy)
      {
        const auto __op = [__stream](const auto& __src, const auto& __dst) {
          cudax::__launch_copy_contiguous_kernel(__src, __dst, __stream);
        };
        cudax::__dispatch_by_vector_size(__src_normalized, __dst_normalized, __op);
      }
      // (2b) non-vectorized case but inner size is large enough to use the contiguous kernel
      else
      {
        cudax::__launch_copy_contiguous_kernel(
          __src_normalized, __dst_normalized, __stream, __src.accessor(), __dst.accessor());
      }
      return;
    }

    const auto __try_shared_mem_copy = [&]() {
      if constexpr (__max_rank >= 2 && __max_rank <= cudax::__max_shared_mem_kernel_rank)
      {
        if (__src_simplified.__rank == 2) // Optimize when the actual rank is 2
        {
          const auto __src_rank2 = cudax::__narrow_raw_tensor_rank<2>(__src_simplified);
          const auto __dst_rank2 = cudax::__narrow_raw_tensor_rank<2>(__dst_simplified);
          if (cudax::__use_shared_mem_kernel(__src_rank2, __dst_rank2))
          {
            cudax::__launch_copy_shared_mem_kernel(
              __src_rank2, __dst_rank2, __stream, __src.accessor(), __dst.accessor());
            return true;
          }
        }
        if (cudax::__use_shared_mem_kernel(__src_simplified, __dst_simplified))
        {
          cudax::__launch_copy_shared_mem_kernel(
            __src_simplified, __dst_simplified, __stream, __src.accessor(), __dst.accessor());
          return true;
        }
      }
      return false;
    };

    // (3) both non-contiguous and inner size is not large -> try vectorized case
    if constexpr (__are_vectorizable_copy)
    {
      if (__both_stride1)
      {
        // transpose cases can have the innermost mode == 1. These should not fall in the vectorizable_copy and
        // prefer the shared-memory transpose when possible.
        const auto __src_stride_order = cudax::__stride_order(__src_simplified);
        const auto __dst_stride_order = cudax::__stride_order(__dst_simplified);
        bool __same_stride_order      = true;
        for (::cuda::std::size_t __i = 0; __i < __src_simplified.__rank; ++__i)
        {
          __same_stride_order = __same_stride_order && __src_stride_order[__i] == __dst_stride_order[__i];
        }
        if (!__same_stride_order && __try_shared_mem_copy())
        {
          return;
        }

        const auto __op = [__stream](const auto& __src, const auto& __dst) {
          cudax::__copy_optimized(__src, __dst, cudax::__total_size(__src), __stream);
        };
        cudax::__dispatch_by_vector_size(__src_normalized, __dst_normalized, __op);
        return;
      }
    }

    // (4) If only the source tensor has a negative stride (on a given mode) and the destination is contiguous, perform
    // the copy following the destination order.
    if constexpr (::cuda::std::is_signed_v<__src_stride_t>)
    {
      bool __has_negative_src_stride = false;
      for (::cuda::std::size_t __i = 0; __i < __src_normalized.__rank; ++__i)
      {
        __has_negative_src_stride = __has_negative_src_stride || __src_normalized.__strides[__i] < 0;
      }
      if (__has_negative_src_stride && cudax::__num_contiguous_dimensions(__dst_normalized) == __dst_normalized.__rank)
      {
        cudax::__copy_dst_contiguous(
          __src_normalized,
          __dst_normalized,
          cudax::__total_size(__src_normalized),
          __stream,
          __src.accessor(),
          __dst.accessor());
        return;
      }
    }

    // (5) transpose case (non-vectorizable)
    if (__try_shared_mem_copy())
    {
      return;
    }

    // (6) use the simplified rank in device code for common low-rank layouts
    if constexpr (__max_rank > 1)
    {
      if (__src_simplified.__rank == 1)
      {
        cudax::__copy_simplified_rank<1>(
          __src_simplified, __dst_simplified, __stream, __src.accessor(), __dst.accessor());
        return;
      }
    }
    if constexpr (__max_rank > 2)
    {
      if (__src_simplified.__rank == 2)
      {
        cudax::__copy_simplified_rank<2>(
          __src_simplified, __dst_simplified, __stream, __src.accessor(), __dst.accessor());
        return;
      }
    }
    if constexpr (__max_rank > 3)
    {
      if (__src_simplified.__rank == 3)
      {
        cudax::__copy_simplified_rank<3>(
          __src_simplified, __dst_simplified, __stream, __src.accessor(), __dst.accessor());
        return;
      }
    }
    if constexpr (__max_rank > 4)
    {
      if (__src_simplified.__rank == 4)
      {
        cudax::__copy_simplified_rank<4>(
          __src_simplified, __dst_simplified, __stream, __src.accessor(), __dst.accessor());
        return;
      }
    }

    // (7) generic case (fallback)
    cudax::__copy_optimized(
      __src_normalized,
      __dst_normalized,
      cudax::__total_size(__src_normalized),
      __stream,
      __src.accessor(),
      __dst.accessor());
  }
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // _CUDAX__COPY_MDSPAN_D2D_H
