//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_COPY_MDSPAN_TO_CUTE_H
#define __CUDAX_COPY_MDSPAN_TO_CUTE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/mdspan>

#include <cute/layout.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/tensor_impl.hpp>
#include <cutlass/version.h>
//
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
static_assert(CUTLASS_MAJOR >= 4, "CUTLASS 4.x required");

template <typename _Extents, ::cuda::std::size_t _I>
[[nodiscard]] _CCCL_API auto __to_cute_extent(const _Extents& __ext)
{
  constexpr auto __static_ext = _Extents::static_extent(_I);
  if constexpr (__static_ext != ::cuda::std::dynamic_extent)
  {
    return ::cute::C<__static_ext>{};
  }
  else
  {
    return __ext.extent(_I);
  }
}

template <typename _Extents, ::cuda::std::size_t _I>
[[nodiscard]] _CCCL_API constexpr auto __get_layout_right_stride()
{
  if constexpr (_I + 1 >= _Extents::rank())
  {
    return typename _Extents::index_type{1};
  }
  else
  {
    return _Extents::static_extent(_I + 1) * ::cuda::experimental::__get_layout_right_stride<_Extents, _I + 1>();
  }
}

template <typename _Extents, ::cuda::std::size_t _I>
[[nodiscard]] _CCCL_API constexpr auto __get_layout_left_stride()
{
  if constexpr (_I == 0)
  {
    return typename _Extents::index_type{1};
  }
  else
  {
    return _Extents::static_extent(_I - 1) * ::cuda::experimental::__get_layout_left_stride<_Extents, _I - 1>();
  }
}

template <typename _Extents, typename _LayoutPolicy, typename _Accessor, ::cuda::std::size_t _I, typename _Tp>
[[nodiscard]] _CCCL_API auto __to_cute_stride(const ::cuda::std::mdspan<_Tp, _Extents, _LayoutPolicy, _Accessor>& __src)
{
  if constexpr (_Extents::rank_dynamic() == 0)
  {
    if constexpr (::cuda::std::is_same_v<_LayoutPolicy, ::cuda::std::layout_right>)
    {
      constexpr auto __s = ::cuda::experimental::__get_layout_right_stride<_Extents, _I>();
      return ::cute::C<__s>{};
    }
    else if constexpr (::cuda::std::is_same_v<_LayoutPolicy, ::cuda::std::layout_left>)
    {
      constexpr auto __s = ::cuda::experimental::__get_layout_left_stride<_Extents, _I>();
      return ::cute::C<__s>{};
    }
    else
    {
      return __src.stride(_I);
    }
  }
  else
  {
    return __src.stride(_I);
  }
}

template <typename _Tp, typename _Extents, typename _LayoutPolicy, typename _Accessor, ::cuda::std::size_t... _Is>
[[nodiscard]] _CCCL_API auto __to_cute_impl(const ::cuda::std::mdspan<_Tp, _Extents, _LayoutPolicy, _Accessor>& __src,
                                            ::cuda::std::index_sequence<_Is...>)
{
  const auto __shape = ::cute::make_shape(::cuda::experimental::__to_cute_extent<_Extents, _Is>(__src.extents())...);
  const auto __stride =
    ::cute::make_stride(::cuda::experimental::__to_cute_stride<_Extents, _LayoutPolicy, _Accessor, _Is>(__src)...);
  return ::cute::make_tensor(__src.data_handle(), ::cute::make_layout(__shape, __stride));
}

/***********************************************************************************************************************
 * Public API
 **********************************************************************************************************************/

/// @brief Convert a cuda::std::mdspan to a CuTe tensor
///
/// @return A CuTe tensor with dynamic layout that views the same data as the mdspan
///
template <typename _Tp, typename _Extents, typename _LayoutPolicy, typename _Accessor>
[[nodiscard]] _CCCL_API auto to_cute(const ::cuda::std::mdspan<_Tp, _Extents, _LayoutPolicy, _Accessor>& __src)
{
  static_assert(::cuda::std::is_convertible_v<_Accessor, ::cuda::std::default_accessor<_Tp>>,
                "Accessor must be convertible to cuda::std::default_accessor");
  return ::cuda::experimental::__to_cute_impl(__src, ::cuda::std::make_index_sequence<_Extents::rank()>{});
}
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDAX_COPY_MDSPAN_TO_CUTE_H
