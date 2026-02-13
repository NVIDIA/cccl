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
#  include <cuda/std/__type_traits/is_convertible.h>
#  include <cuda/std/__type_traits/is_same.h>

#  include <cuda/experimental/__copy/coalesce_right.cuh>
#  include <cuda/experimental/__copy/max_common_layout.cuh>
#  include <cuda/experimental/__copy/memcpy_batch_tiles.cuh>
#  include <cuda/experimental/__copy/utils.cuh>

#  include <stdexcept>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
template <typename _Tp, int _Rank>
struct __tile_pointer_iterator
{
  const __raw_tensor<_Tp, _Rank> __tensor_original_;
  const __raw_tensor_ordered<_Tp, _Rank> __tensor_sorted_;
  const ::cuda::std::size_t __contiguous_size_;
  const bool __use_stride_order_;

  _CCCL_HOST_API explicit __tile_pointer_iterator(
    const __raw_tensor<_Tp, _Rank>& __tensor_original,
    const __raw_tensor_ordered<_Tp, _Rank>& __tensor_sorted,
    ::cuda::std::size_t __contiguous_size,
    bool __use_stride_order) noexcept
      : __tensor_original_{__tensor_original}
      , __tensor_sorted_{__tensor_sorted}
      , __contiguous_size_{__contiguous_size}
      , __use_stride_order_{__use_stride_order}
  {}

  [[nodiscard]] _CCCL_HOST_API _Tp* operator()(::cuda::std::size_t __tile_idx) const noexcept
  {
    // TODO: potential optimizations: use fast_div_mod in tile iterator
    const auto& __shapes         = __use_stride_order_ ? __tensor_sorted_.__shapes : __tensor_original_.__shapes;
    const auto& __strides        = __use_stride_order_ ? __tensor_sorted_.__strides : __tensor_original_.__strides;
    const auto __rank            = static_cast<int>(__tensor_sorted_.__rank);
    auto __index                 = __tile_idx * __contiguous_size_;
    ::cuda::std::size_t __offset = 0;
    for (int __i = __rank - 1; __i >= 0; --__i)
    {
      __offset += (__index % __shapes[__i]) * __strides[__i];
      __index /= __shapes[__i];
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
    _CCCL_THROW(std::invalid_argument, "mdspans must have the same size");
  }
  if (__src.size() == 0)
  {
    return;
  }
  if (__src.data_handle() == nullptr || __dst.data_handle() == nullptr)
  {
    _CCCL_THROW(std::invalid_argument, "mdspan data handle must not be nullptr");
  }
  if constexpr (_ExtentsIn::rank() == 0) // scalar case
  {
    ::cuda::__driver::__memcpyAsync(__dst.data_handle(), __src.data_handle(), sizeof(_TpIn), __stream.get());
  }
  else
  {
    // __coalesce_right is not strictly needed. It is an optimization to reduce the number of modes.
    using ::cuda::std::size_t;
    const auto __src1       = ::cuda::experimental::__to_raw_tensor(__src);
    const auto __dst1       = ::cuda::experimental::__to_raw_tensor(__dst);
    const auto __src2       = ::cuda::experimental::__coalesce_right(__src1);
    const auto __dst2       = ::cuda::experimental::__coalesce_right(__dst1);
    const auto __src_sorted = ::cuda::experimental::__sort_by_stride_desc(__src1);
    const auto __dst_sorted = ::cuda::experimental::__sort_by_stride_desc(__dst1);
    if (::cuda::experimental::__is_not_unique(__dst_sorted))
    {
      _CCCL_THROW(std::invalid_argument, "destination mdspan must have unique layout");
    }
    const auto __tile_size        = ::cuda::experimental::__max_common_contiguous_size(__src_sorted, __dst_sorted);
    const auto __num_tiles        = static_cast<size_t>(__src.size() / __tile_size);
    const auto __copy_bytes       = __tile_size * sizeof(_TpIn);
    const auto __use_stride_order = ::cuda::experimental::__same_stride_order(__src_sorted, __dst_sorted);
    __tile_pointer_iterator<_TpIn, _ExtentsIn::rank()> __src_tiles_iterator(
      __src1, __src_sorted, __tile_size, __use_stride_order);
    __tile_pointer_iterator<_TpOut, _ExtentsOut::rank()> __dst_tiles_iterator(
      __dst1, __dst_sorted, __tile_size, __use_stride_order);
#  if _CCCL_CTK_AT_LEAST(13, 0)
    // Use the memcpy batch API to copy all tiles in one call
    ::cuda::experimental::__memcpy_batch_tiles(
      __src_tiles_iterator,
      __dst_tiles_iterator,
      __num_tiles,
      __copy_bytes,
      __direction,
      __src.data_handle(),
      __dst.data_handle(),
      __stream);
#  else
    // Use individual memcpy calls for each tile
    for (size_t __tile_idx = 0; __tile_idx < __num_tiles; ++__tile_idx)
    {
      const auto __src_ptr = static_cast<const void*>(__src_tiles_iterator(__tile_idx));
      const auto __dst_ptr = static_cast<void*>(__dst_tiles_iterator(__tile_idx));
      ::cuda::__driver::__memcpyAsync(__dst_ptr, __src_ptr, __copy_bytes, __stream.get());
    }
#  endif // _CCCL_CTK_AT_LEAST(13, 0)
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
