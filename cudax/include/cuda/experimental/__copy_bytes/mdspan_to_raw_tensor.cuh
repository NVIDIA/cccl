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

#  include <cuda/__mdspan/traits.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__mdspan/mdspan.h>
#  include <cuda/std/__type_traits/remove_cvref.h>

#  include <cuda/experimental/__copy_bytes/types.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief Extracts the stride type from a layout mapping, defaulting to `index_type` when absent.
template <typename _LayoutPolicy, typename _Mapping>
struct __mapping_stride_type
{
  using __type = typename _Mapping::index_type;
};

template <typename _Mapping>
struct __mapping_stride_type<::cuda::layout_stride_relaxed, _Mapping>
{
  using __type = typename _Mapping::offset_type;
};

//! @brief Convenience alias: stride type of a layout mapping for given extents and layout policy.
//!
//! For `layout_stride_relaxed`, uses `offset_type` (signed) since strides can be negative.
//! For other layouts, uses `stride_type` if available, otherwise `index_type`.
template <typename _LayoutPolicy, typename _Mapping>
using __mdspan_stride_t = typename __mapping_stride_type<_LayoutPolicy, ::cuda::std::remove_cvref_t<_Mapping>>::__type;

//! @brief Convenience alias: `__raw_tensor` type produced by @ref __to_raw_tensor for a given mdspan.
template <typename _MdspanCVRef, typename _Mdspan = ::cuda::std::remove_cvref_t<_MdspanCVRef>>
using __to_raw_tensor_t =
  __raw_tensor<typename _Mdspan::index_type,
               __mdspan_stride_t<typename _Mdspan::layout_type, typename _Mdspan::mapping_type>,
               typename _Mdspan::element_type,
               _Mdspan::rank()>;

//! @brief Converts an mdspan to a @ref __raw_tensor with explicitly specified extent, stride types.
//!
//! Extent-1 modes are removed from the resulting tensor.
//!
//! @param[in] __mdspan Source mdspan view
//! @return @ref __raw_tensor descriptor with data pointer, rank, extents, and strides
template <typename _ExtentT,
          typename _StrideT,
          ::cuda::std::size_t _MaxRank,
          typename _Tp,
          typename _Extents,
          typename _LayoutPolicy,
          typename _AccessorPolicy>
[[nodiscard]]
_CCCL_HOST_API constexpr __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>
__to_raw_tensor(const ::cuda::std::mdspan<_Tp, _Extents, _LayoutPolicy, _AccessorPolicy>& __mdspan) noexcept
{
  static_assert(_MaxRank >= _Extents::rank(), "_MaxRank must be at least _Extents::rank()");
  using __raw_tensor_t = __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>;
  using __rank_t       = typename _Extents::rank_type;
  auto* __data         = __mdspan.data_handle();
  if constexpr (::cuda::__is_layout_stride_relaxed_v<_LayoutPolicy>)
  {
    __data += __mdspan.mapping().offset();
  }
  __raw_tensor_t __result{__data, 0, {}, {}};
  if constexpr (_Extents::rank() > 0)
  {
    __rank_t __r = 0;
    for (__rank_t __i = 0; __i < _Extents::rank(); ++__i)
    {
      const auto __extent = static_cast<_ExtentT>(__mdspan.extent(__i));
      if (__extent != _ExtentT{1})
      {
        __result.__extents[__r] = __extent;
        __result.__strides[__r] = static_cast<_StrideT>(__mdspan.stride(__i));
        ++__r;
      }
    }
    __result.__rank = __r;
  }
  return __result;
}

//! @brief Converts an mdspan to a @ref __raw_tensor using its native extent and stride types.
//!
//! Extent-1 modes are removed from the resulting tensor.
//!
//! @param[in] __mdspan Source mdspan view
//! @return @ref __raw_tensor descriptor with data pointer, rank, extents, and strides
template <typename _Tp, typename _Extents, typename _LayoutPolicy, typename _AccessorPolicy>
[[nodiscard]]
_CCCL_HOST_API constexpr auto
__to_raw_tensor(const ::cuda::std::mdspan<_Tp, _Extents, _LayoutPolicy, _AccessorPolicy>& __mdspan) noexcept
  -> __to_raw_tensor_t<decltype(__mdspan)>
{
  using __extent_t = typename _Extents::index_type;
  using __stride_t = __mdspan_stride_t<_LayoutPolicy, decltype(__mdspan.mapping())>;
  return ::cuda::experimental::__to_raw_tensor<__extent_t, __stride_t, _Extents::rank()>(__mdspan);
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // __CUDAX_COPY_MDSPAN_TO_RAW_TENSOR_H
