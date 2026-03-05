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
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__cstdlib/abs.h>
#  include <cuda/std/__type_traits/is_convertible.h>
#  include <cuda/std/__type_traits/is_same.h>

#  include <cuda/experimental/__copy/max_common_layout.cuh>
#  include <cuda/experimental/__copy/memcpy_batch_tiles.cuh>
#  include <cuda/experimental/__copy/utils.cuh>
#  include <cuda/experimental/__copy_bytes/layout_optimization.cuh>

#  include <stdexcept>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
template <typename _Ep, typename _Sp, typename _Tp, ::cuda::std::size_t _MaxRank>
struct __tile_pointer_iterator
{
  using _Ep1 = ::cuda::std::make_unsigned_t<_Ep>;

  _Tp* __data;
  ::cuda::std::size_t __outer_rank;
  ::cuda::std::array<_Ep1, _MaxRank> __outer_extents;
  ::cuda::std::array<_Sp, _MaxRank> __outer_strides;

  [[nodiscard]] _CCCL_HOST_API _Tp* operator()(::cuda::std::size_t __tile_idx) const noexcept
  {
    _Sp __offset = 0;
    for (::cuda::std::size_t __i = 0; __i < __outer_rank; ++__i)
    {
      __offset += static_cast<_Sp>(__tile_idx % __outer_extents[__i]) * __outer_strides[__i];
      __tile_idx /= __outer_extents[__i];
    }
    return __data + __offset;
  }
};

template <typename _Ep, typename _Sp, typename _Tp, ::cuda::std::size_t _Rank>
struct __tile_pointer_iterator_independent
{
  const __raw_tensor<_Ep, _Sp, _Tp, _Rank> __tensor_original_;
  const __raw_tensor_ordered<_Ep, _Sp, _Tp, _Rank> __tensor_sorted_;
  const ::cuda::std::size_t __contiguous_size_;
  const bool __use_stride_order_;

  _CCCL_HOST_API explicit __tile_pointer_iterator_independent(
    const __raw_tensor<_Ep, _Sp, _Tp, _Rank>& __tensor_original,
    const __raw_tensor_ordered<_Ep, _Sp, _Tp, _Rank>& __tensor_sorted,
    ::cuda::std::size_t __contiguous_size,
    bool __use_stride_order) noexcept
      : __tensor_original_{__tensor_original}
      , __tensor_sorted_{__tensor_sorted}
      , __contiguous_size_{__contiguous_size}
      , __use_stride_order_{__use_stride_order}
  {}

  [[nodiscard]] _CCCL_HOST_API _Tp* operator()(::cuda::std::size_t __tile_idx) const noexcept
  {
    const auto& __extents        = __use_stride_order_ ? __tensor_sorted_.__extents : __tensor_original_.__extents;
    const auto& __strides        = __use_stride_order_ ? __tensor_sorted_.__strides : __tensor_original_.__strides;
    const auto __rank            = __tensor_sorted_.__rank;
    auto __index                 = __tile_idx * __contiguous_size_;
    ::cuda::std::size_t __offset = 0;
    for (::cuda::std::size_t __i = 0; __i < __rank; ++__i)
    {
      __offset += (__index % __extents[__i]) * __strides[__i];
      __index /= __extents[__i];
    }
    return __tensor_sorted_.__data + __offset;
  }
};

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
  static_assert(::cuda::std::is_trivially_copyable_v<_TpIn>, "TpIn must be trivially copyable");
  static_assert(::cuda::std::is_trivially_copyable_v<_TpOut>, "TpOut must be trivially copyable");
  static_assert(!::cuda::std::is_const_v<_TpOut>, "TpOut must not be const");
  static_assert(::cuda::std::is_same_v<::cuda::std::remove_cv_t<_TpIn>, ::cuda::std::remove_cv_t<_TpOut>>,
                "TpIn and TpOut must be the same type");
  static_assert(::cuda::__is_cuda_mdspan_layout_v<_LayoutPolicyIn>,
                "LayoutPolicyIn must be a predefined layout policy");
  static_assert(::cuda::__is_cuda_mdspan_layout_v<_LayoutPolicyOut>,
                "LayoutPolicyOut must be a predefined layout policy");
  using __default_accessor_in  = ::cuda::std::default_accessor<_TpIn>;
  using __default_accessor_out = ::cuda::std::default_accessor<_TpOut>;
  static_assert(::cuda::std::is_convertible_v<_AccessorPolicyIn, __default_accessor_in>,
                "AccessorPolicyIn must be convertible to cuda::std::default_accessor");
  static_assert(::cuda::std::is_convertible_v<_AccessorPolicyOut, __default_accessor_out>,
                "AccessorPolicyOut must be convertible to cuda::std::default_accessor");
  if (__src.size() != __dst.size())
  {
    _CCCL_THROW(::std::invalid_argument, "mdspans must have the same size");
  }
  if (__src.size() == 0)
  {
    return;
  }
  if (__src.data_handle() == nullptr || __dst.data_handle() == nullptr)
  {
    _CCCL_THROW(::std::invalid_argument, "mdspan data handle must not be nullptr");
  }
  if (::cuda::experimental::__is_not_unique(__dst))
  {
    _CCCL_THROW(::std::invalid_argument, "destination mdspan must have unique layout");
  }
  if (__src.size() == 1)
  {
    ::cuda::__driver::__memcpyAsync(__dst.data_handle(), __src.data_handle(), sizeof(_TpIn), __stream.get());
    return;
  }
  if constexpr (_ExtentsIn::rank() > 0 && _ExtentsOut::rank() > 0)
  {
    using ::cuda::std::size_t;
    using __extent_t          = ::cuda::std::size_t;
    using __stride_t          = ::cuda::std::int64_t;
    constexpr auto __max_rank = ::cuda::std::max(_ExtentsIn::rank(), _ExtentsOut::rank());

    // Simplify each tensor independently:
    //   1. Remove extent-1 modes (safe: extent-1 modes don't affect data layout)
    //   2. Widen to a common _MaxRank so the paired functions accept both
    // NOTE: independent coalescing is NOT applied here because it would merge modes
    // that correspond to different logical orderings, breaking the paired pipeline.
    // The paired pipeline's own __coalesce_paired handles joint coalescing correctly.
    auto __src_raw = cudax::__widen<__max_rank>(cudax::__to_raw_tensor(__src, ::cuda::std::true_type{}));
    auto __dst_raw = cudax::__widen<__max_rank>(cudax::__to_raw_tensor(__dst, ::cuda::std::true_type{}));

    // Compare simplified shapes
    bool __same_shape = (__src_raw.__rank == __dst_raw.__rank);
    for (size_t __i = 0; __same_shape && __i < __src_raw.__rank; ++__i)
    {
      __same_shape = (__src_raw.__extents[__i] == __dst_raw.__extents[__i]);
    }

    if (__same_shape)
    {
      // ---- Paired pipeline: simplified tensors have the same shape ----
      // Extents already match and no extent-1 modes remain.
      // Sort both by dst's ascending absolute stride (same permutation to both)
      cudax::__sort_by_stride_paired(__src_raw, __dst_raw);
      cudax::__flip_negative_strides_paired(__src_raw, __dst_raw);
      cudax::__coalesce_paired(__src_raw, __dst_raw);

      const bool __both_stride1  = (__src_raw.__strides[0] == 1) && (__dst_raw.__strides[0] == 1);
      const auto __tile_size     = __both_stride1 ? static_cast<size_t>(__src_raw.__extents[0]) : size_t{1};
      const auto __num_tiles     = static_cast<size_t>(__src.size() / __tile_size);
      const auto __copy_bytes    = __tile_size * sizeof(_TpIn);
      const size_t __outer_start = __both_stride1 ? size_t{1} : size_t{0};
      const size_t __outer_rank  = __src_raw.__rank - __outer_start;

      __tile_pointer_iterator<__extent_t, __stride_t, _TpIn, __max_rank> __src_iter{};
      __src_iter.__data       = __src_raw.__data;
      __src_iter.__outer_rank = __outer_rank;
      __tile_pointer_iterator<__extent_t, __stride_t, _TpOut, __max_rank> __dst_iter{};
      __dst_iter.__data       = __dst_raw.__data;
      __dst_iter.__outer_rank = __outer_rank;
      for (size_t __i = 0; __i < __outer_rank; ++__i)
      {
        __src_iter.__outer_extents[__i] = __src_raw.__extents[__i + __outer_start];
        __src_iter.__outer_strides[__i] = __src_raw.__strides[__i + __outer_start];
        __dst_iter.__outer_extents[__i] = __dst_raw.__extents[__i + __outer_start];
        __dst_iter.__outer_strides[__i] = __dst_raw.__strides[__i + __outer_start];
      }

      ::cuda::experimental::__memcpy_batch_tiles(
        __src_iter,
        __dst_iter,
        __num_tiles,
        __copy_bytes,
        __direction,
        __src.data_handle(),
        __dst.data_handle(),
        __stream);
    }
    else
    {
      // ---- Independent pipeline: simplified shapes differ ----
      const auto __src_orig         = ::cuda::experimental::__to_raw_tensor(__src);
      const auto __dst_orig         = ::cuda::experimental::__to_raw_tensor(__dst);
      const auto __src_sorted       = ::cuda::experimental::__sort_by_stride(__src_orig);
      const auto __dst_sorted       = ::cuda::experimental::__sort_by_stride(__dst_orig);
      const auto __tile_size        = ::cuda::experimental::__max_common_contiguous_size(__src_sorted, __dst_sorted);
      const auto __num_tiles        = static_cast<size_t>(__src.size() / __tile_size);
      const auto __copy_bytes       = __tile_size * sizeof(_TpIn);
      const auto __use_stride_order = ::cuda::experimental::__same_stride_order(__src_sorted, __dst_sorted);
      __tile_pointer_iterator_independent<__extent_t, __stride_t, _TpIn, _ExtentsIn::rank()> __src_tiles_iterator(
        __src_orig, __src_sorted, __tile_size, __use_stride_order);
      __tile_pointer_iterator_independent<__extent_t, __stride_t, _TpOut, _ExtentsOut::rank()> __dst_tiles_iterator(
        __dst_orig, __dst_sorted, __tile_size, __use_stride_order);
      ::cuda::experimental::__memcpy_batch_tiles(
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
}

/***********************************************************************************************************************
 * Public API
 **********************************************************************************************************************/

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
