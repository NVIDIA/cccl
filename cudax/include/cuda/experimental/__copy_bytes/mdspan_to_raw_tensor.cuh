//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_COPY_MDSPAN_TO_RAW_TENSOR_H
#define __CUDAX_COPY_MDSPAN_TO_RAW_TENSOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__mdspan/mdspan.h>
#  include <cuda/std/__type_traits/void_t.h>

#  include <cuda/experimental/__copy_bytes/types.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief Tag constant used to enable extent-1 mode removal in @ref __to_raw_tensor.
inline constexpr auto __remove_extent1 = ::cuda::std::true_type{};

template <typename _Mapping, typename = void>
struct __mapping_stride_type
{
  using type = typename _Mapping::index_type;
};

template <typename _Mapping>
struct __mapping_stride_type<_Mapping, ::cuda::std::void_t<typename _Mapping::stride_type>>
{
  using type = typename _Mapping::stride_type;
};

template <typename _Extents, typename _LayoutPolicy>
using __mdspan_stride_t = typename __mapping_stride_type<typename _LayoutPolicy::template mapping<_Extents>>::type;

template <::cuda::std::size_t _MaxRank, typename _Tp, typename _Extents, typename _LayoutPolicy>
using __to_raw_tensor_t =
  __raw_tensor<typename _Extents::index_type, __mdspan_stride_t<_Extents, _LayoutPolicy>, _Tp, _MaxRank>;

/**
 * @brief Converts an mdspan view to a raw tensor descriptor using its native extent and stride types.
 *
 * The descriptor stores data pointer, rank, extents, and strides in arrays to use them at runtime.
 */
template <typename _Tp, typename _Extents, typename _LayoutPolicy, typename _AccessorPolicy, bool _RemoveExtent1 = false>
[[nodiscard]]
_CCCL_HOST_API constexpr __to_raw_tensor_t<_Extents::rank(), _Tp, _Extents, _LayoutPolicy>
__to_raw_tensor(const ::cuda::std::mdspan<_Tp, _Extents, _LayoutPolicy, _AccessorPolicy>& __mdspan,
                ::cuda::std::bool_constant<_RemoveExtent1> = {}) noexcept
{
  using __raw_tensor_t = __to_raw_tensor_t<_Extents::rank(), _Tp, _Extents, _LayoutPolicy>;
  using __extent_t     = typename __raw_tensor_t::__unsigned_extent_t;
  using __stride_t     = __mdspan_stride_t<_Extents, _LayoutPolicy>;
  using __rank_t       = typename _Extents::rank_type;
  __raw_tensor_t __result{__mdspan.data_handle(), 0, {}, {}};
  if constexpr (_Extents::rank() > 0)
  {
    ::cuda::std::size_t __r = 0;
    for (::cuda::std::size_t __i = 0; __i < _Extents::rank(); ++__i)
    {
      const auto __rank   = static_cast<__rank_t>(__i);
      const auto __extent = static_cast<__extent_t>(__mdspan.extent(__rank));
      if (!_RemoveExtent1 || __extent != 1)
      {
        __result.__extents[__r] = __extent;
        __result.__strides[__r] = static_cast<__stride_t>(__mdspan.stride(__rank));
        ++__r;
      }
    }
    __result.__rank = __r;
  }
  return __result;
}

/**
 * @brief Converts an mdspan view to a raw tensor descriptor.
 *
 * The descriptor stores data pointer, rank, extents, and strides in arrays to use them at runtime.
 */
template <typename _ExtentT,
          typename _StrideT,
          ::cuda::std::size_t _MaxRank,
          typename _Tp,
          typename _Extents,
          typename _LayoutPolicy,
          typename _AccessorPolicy,
          bool _RemoveExtent1 = false>
[[nodiscard]]
_CCCL_HOST_API constexpr __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>
__to_raw_tensor(const ::cuda::std::mdspan<_Tp, _Extents, _LayoutPolicy, _AccessorPolicy>& __mdspan,
                ::cuda::std::bool_constant<_RemoveExtent1> = {}) noexcept
{
  using __raw_tensor_t = __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>;
  using __extent_t     = typename __raw_tensor_t::__unsigned_extent_t;
  using __rank_t       = typename _Extents::rank_type;
  __raw_tensor_t __result{__mdspan.data_handle(), 0, {}, {}};
  if constexpr (_Extents::rank() > 0)
  {
    ::cuda::std::size_t __r = 0;
    for (::cuda::std::size_t __i = 0; __i < _Extents::rank(); ++__i)
    {
      const auto __rank   = static_cast<__rank_t>(__i);
      const auto __extent = static_cast<__extent_t>(__mdspan.extent(__rank));
      if (!_RemoveExtent1 || __extent != 1)
      {
        __result.__extents[__r] = __extent;
        __result.__strides[__r] = static_cast<_StrideT>(__mdspan.stride(__rank));
        ++__r;
      }
    }
    __result.__rank = __r;
  }
  return __result;
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // __CUDAX_COPY_MDSPAN_TO_RAW_TENSOR_H
