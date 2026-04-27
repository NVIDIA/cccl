//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_COPY_TENSOR_QUERY_H
#define __CUDAX_COPY_TENSOR_QUERY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/std/__algorithm/stable_sort.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__mdspan/mdspan.h>
#  include <cuda/std/array>

#  include <cuda/experimental/__copy_bytes/abs_integer.cuh>
#  include <cuda/experimental/__copy_bytes/mdspan_to_raw_tensor.cuh>
#  include <cuda/experimental/__copy_bytes/types.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief Checks whether two raw tensors have the same rank and identical extents.
//!
//! @param[in] __tensor_in  First raw tensor
//! @param[in] __tensor_out Second raw tensor
//! @return true if rank and all extents match element-wise
template <typename _ExtentTIn,
          typename _StrideTIn,
          typename _TpIn,
          ::cuda::std::size_t _MaxRankIn,
          typename _ExtentTOut,
          typename _StrideTOut,
          typename _TpOut,
          ::cuda::std::size_t _MaxRankOut>
[[nodiscard]] _CCCL_HOST_API constexpr bool
__same_extents(const __raw_tensor<_ExtentTIn, _StrideTIn, _TpIn, _MaxRankIn>& __tensor_in,
               const __raw_tensor<_ExtentTOut, _StrideTOut, _TpOut, _MaxRankOut>& __tensor_out) noexcept
{
  if (__tensor_in.__rank != __tensor_out.__rank)
  {
    return false;
  }
  using __raw_tensor_t = __raw_tensor<_ExtentTIn, _StrideTIn, _TpIn, _MaxRankIn>;
  using __rank_t       = typename __raw_tensor_t::__rank_t;
  for (__rank_t __i = 0; __i < __tensor_in.__rank; ++__i)
  {
    if (__tensor_in.__extents[__i] != __tensor_out.__extents[__i])
    {
      return false;
    }
  }
  return true;
}

// lambdas are painful without --extended-lambda and when used with __host__ __device__ functions
template <typename _StrideT, ::cuda::std::size_t _MaxRank>
struct __stride_compare
{
  const ::cuda::std::array<_StrideT, _MaxRank>& __strides;

  template <typename _Idx>
  [[nodiscard]] _CCCL_API bool operator()(const _Idx __lhs, const _Idx __rhs) const noexcept
  {
    return ::cuda::experimental::__abs_integer(__strides[__lhs])
         < ::cuda::experimental::__abs_integer(__strides[__rhs]);
  }
};

//! @brief Reorders tensor modes by ascending absolute stride.
//!
//! After sorting, mode 0 has the smallest absolute stride (innermost) and mode rank-1 has the largest (outermost).
//!
//! @param[in] __tensor Raw tensor to sort
//! @return Raw tensor with modes reordered by ascending absolute stride
template <typename _ExtentT, typename _StrideT, typename _Tp, ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API constexpr __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>
__sort_by_stride(const __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>& __tensor) noexcept
{
  using __raw_tensor_t = __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>;
  using __rank_t       = typename __raw_tensor_t::__rank_t;
  const auto __rank    = __tensor.__rank;
  ::cuda::std::array<__rank_t, _MaxRank> __perm{};
  for (__rank_t __i = 0; __i < __rank; ++__i)
  {
    __perm[__i] = __i;
  }
  ::cuda::std::stable_sort(
    __perm.begin(), __perm.begin() + __rank, __stride_compare<_StrideT, _MaxRank>{__tensor.__strides});

  __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank> __result{__tensor.__data, __rank};
  for (__rank_t __i = 0; __i < __rank; ++__i)
  {
    __result.__extents[__i] = __tensor.__extents[__perm[__i]];
    __result.__strides[__i] = __tensor.__strides[__perm[__i]];
  }
  return __result;
}

//! @brief Conservative check for interleaved stride order in tensor layouts.
//!
//! Sorts modes by ascending absolute stride, then verifies two conditions:
//! 1. No mode with extent > 1 has stride == 0 (broadcast)
//! 2. No mode's span (extent * |stride|) exceeds the next mode's |stride|
//!
//! Returns true when the layout fails this non-interleaving rule. This is stronger than a mathematical injectivity
//! check and may reject some layouts with distinct offsets.
//!
//! @param[in] __mdspan Mdspan view to inspect
//! @return true if the layout has interleaved strides
template <typename _Tp, typename _Extents, typename _LayoutPolicy, typename _AccessorPolicy>
[[nodiscard]] _CCCL_HOST_API constexpr bool __has_interleaved_stride_order(
  const ::cuda::std::mdspan<_Tp, _Extents, _LayoutPolicy, _AccessorPolicy>& __mdspan) noexcept
{
  if constexpr (_Extents::rank() > 0)
  {
    namespace cudax       = ::cuda::experimental;
    const auto __tensor   = cudax::__to_raw_tensor(__mdspan);
    const auto __sorted   = cudax::__sort_by_stride(__tensor);
    using __stride_t      = decltype(__sorted.__strides[0]);
    using __rank_t        = typename _Extents::rank_type;
    const auto& __extents = __sorted.__extents;
    const auto& __strides = __sorted.__strides;
    const auto __rank     = __sorted.__rank;
    for (__rank_t __i = 0; __i < __rank; ++__i)
    {
      if (__extents[__i] > 1 && __strides[__i] == 0)
      {
        return true;
      }
    }
    for (__rank_t __i = 0; __i + 1 < __rank; ++__i)
    {
      const auto __extent = static_cast<__stride_t>(__extents[__i]);
      if (__extent * cudax::__abs_integer(__strides[__i]) > cudax::__abs_integer(__strides[__i + 1]))
      {
        return true;
      }
    }
    return false;
  }
  else
  {
    return false;
  }
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // __CUDAX_COPY_TENSOR_QUERY_H
