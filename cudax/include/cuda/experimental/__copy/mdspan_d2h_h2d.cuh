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

#  include <cuda/experimental/__copy/cute/max_common_layout.cuh>
#  include <cuda/experimental/__copy/mdspan_to_cute.cuh>

#  include <stdexcept>
#  include <vector>

#  include <cute/layout.hpp>
#  include <cute/tensor_impl.hpp>
//
#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
template <typename _Tp, int _Rank>
struct __tile_pointer_iterator
{
  ::cuda::std::array<::cuda::std::size_t, _Rank> __shapes;
  ::cuda::std::array<::cuda::std::int64_t, _Rank> __strides;
  ::cuda::std::size_t __contigous_size;
  _Tp* __data;

  template <typename _Tensor>
  _CCCL_HOST_API __tile_pointer_iterator(_Tensor __tensor, ::cuda::std::size_t __contigous_size) noexcept
      : __contigous_size{__contigous_size}
      , __data{__tensor.data()}
  {
    ::cuda::experimental::__init_layout(
      __tensor.layout().shape(),
      __tensor.layout().stride(),
      __shapes,
      __strides,
      ::cuda::std::make_index_sequence<_Rank>{});
  }

  [[nodiscard]] _CCCL_HOST_API _Tp* operator()(::cuda::std::size_t __tile_idx) const noexcept
  {
    auto __pos                   = __tile_idx * __contigous_size;
    ::cuda::std::size_t __offset = 0;
    for (int __i = _Rank - 1; __i >= 0; --__i)
    {
      __offset += (__pos % __shapes[__i]) * __strides[__i];
      __pos /= __shapes[__i];
    }
    return __data + __offset;
  }
};

template <class _Shape, class _Stride>
[[nodiscard]] _CCCL_HOST_API bool __is_not_unique(const ::cute::Layout<_Shape, _Stride>& __layout) noexcept
{
  constexpr auto __rank = __rank_v<_Shape>;
  constexpr ::cuda::std::make_index_sequence<__rank> __rank_seq{};
  ::cuda::std::array<::cuda::std::size_t, __rank> __shapes;
  ::cuda::std::array<::cuda::std::int64_t, __rank> __strides;
  ::cuda::experimental::__init_layout(__layout.shape(), __layout.stride(), __shapes, __strides, __rank_seq);
  ::cuda::experimental::__sort_by_stride_layout<__rank>(__shapes, __strides);
  for (int __i = 1; __i < __rank; ++__i)
  {
    if (::cuda::std::abs(__strides[__i]) < __shapes[__i - 1] * ::cuda::std::abs(__strides[__i - 1]))
    {
      return true;
    }
  }
  return false;
}

enum class __copy_direction
{
  __host_to_device,
  __device_to_host,
};

#  if _CCCL_CTK_AT_LEAST(13, 0)

[[nodiscard]] _CCCL_HOST_API inline ::CUmemcpyAttributes
__get_memcpy_attributes(__copy_direction __direction, const void* __src_ptr, const void* __dst_ptr) noexcept
{
  if (__direction == __copy_direction::__host_to_device)
  {
    const int __device_ordinal =
      ::cuda::__driver::__pointerGetAttribute<::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL>(__dst_ptr);
    return ::CUmemcpyAttributes{
      ::CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
      ::CUmemLocation{CU_MEM_LOCATION_TYPE_HOST, 0},
      ::CUmemLocation{CU_MEM_LOCATION_TYPE_DEVICE, __device_ordinal},
      0};
  }
  else
  {
    const int __device_ordinal =
      ::cuda::__driver::__pointerGetAttribute<::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL>(__src_ptr);
    return ::CUmemcpyAttributes{
      ::CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
      ::CUmemLocation{CU_MEM_LOCATION_TYPE_DEVICE, __device_ordinal},
      ::CUmemLocation{CU_MEM_LOCATION_TYPE_HOST, 0},
      0};
  }
}
#  endif // _CCCL_CTK_AT_LEAST(13, 0)

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
  __copy_direction __direction,
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
  if constexpr (_ExtentsIn::rank() == 0)
  {
    ::cuda::__driver::__memcpyAsync(__dst.data_handle(), __src.data_handle(), sizeof(_TpIn), __stream.get());
  }
  else
  {
    const auto __src1 = ::cuda::experimental::to_cute(__src);
    const auto __dst1 = ::cuda::experimental::to_cute(__dst);
    if (::cuda::experimental::__is_not_unique(__dst1.layout()))
    {
      _CCCL_THROW(std::invalid_argument, "destination mdspan must have unique layout");
    }
    // TODO: potential optimizations: merge contiguous modes, remove 1-size modes, use fast_div_mod in tile iterator
    constexpr auto __rank_max = ::cuda::std::max(_ExtentsIn::rank(), _ExtentsOut::rank());
    auto __src2               = ::cute::make_tensor(__src1.data(), ::cute::append<__rank_max>(__src1.layout()));
    auto __dst2               = ::cute::make_tensor(__dst1.data(), ::cute::append<__rank_max>(__dst1.layout()));
    cute::print(__src2.layout());
    printf("\n");
    cute::print(__dst2.layout());
    printf("\n------------------------\n");
    const auto __tile_size  = ::cuda::experimental::__max_common_contiguous_size(__src2.layout(), __dst2.layout());
    const auto __copy_bytes = __tile_size * sizeof(_TpIn);
    const auto __num_tiles  = static_cast<::cuda::std::size_t>(::cute::size(__src2) / __tile_size);
    printf("__tile_size: %zu\n", __tile_size);
    printf("__num_tiles: %zu\n", __num_tiles);

#  if _CCCL_CTK_AT_LEAST(13, 0)
    // Use the memcpy batch API to copy all tiles in one call
    ::std::vector<const void*> __src_ptr_vector(__num_tiles);
    ::std::vector<void*> __dst_ptr_vector(__num_tiles);
    __tile_pointer_iterator<_TpIn, __rank_max> __src_tiles_iterator(__src2, __tile_size);
    __tile_pointer_iterator<_TpOut, __rank_max> __dst_tiles_iterator(__dst2, __tile_size);
    for (size_t __tile_idx = 0; __tile_idx < __num_tiles; ++__tile_idx)
    {
      __src_ptr_vector[__tile_idx] = __src_tiles_iterator(__tile_idx);
      __dst_ptr_vector[__tile_idx] = __dst_tiles_iterator(__tile_idx);
    }
    ::std::vector<size_t> __sizes(__num_tiles, __copy_bytes);
    auto __attributes =
      ::cuda::experimental::__get_memcpy_attributes(__direction, __src.data_handle(), __dst.data_handle());
    size_t __num_attributes = 0;
    ::cuda::__driver::__memcpyBatchAsync(
      __dst_ptr_vector.data(),
      __src_ptr_vector.data(),
      __sizes.data(),
      __num_tiles,
      &__attributes,
      &__num_attributes,
      1,
      __stream.get());
#  else
    // Use individual memcpy calls for each tile
    for (size_t __tile_idx = 0; __tile_idx < __num_tiles; ++__tile_idx)
    {
      auto __src_ptr = static_cast<const void*>(__src_tiles_iterator(__tile_idx));
      auto __dst_ptr = static_cast<void*>(__dst_tiles_iterator(__tile_idx));
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
