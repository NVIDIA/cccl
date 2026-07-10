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

#  include <iostream>

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
  [[nodiscard]] _CCCL_HOST_DEVICE_API bool operator()(const _Idx __lhs, const _Idx __rhs) const noexcept
  {
    return ::cuda::experimental::__abs_integer(__strides[__lhs])
         < ::cuda::experimental::__abs_integer(__strides[__rhs]);
  }
};

//! @brief Computes the mode permutation that orders a tensor by ascending absolute stride.
//!
//! @param[in] __tensor Raw tensor whose stride order is inspected
//! @return Mode permutation sorted by ascending absolute stride
template <typename _ExtentT, typename _StrideT, typename _Tp, ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API constexpr ::cuda::std::array<::cuda::std::size_t, _MaxRank>
__stride_order(const __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>& __tensor) noexcept
{
  ::cuda::std::array<::cuda::std::size_t, _MaxRank> __perm{};
  for (::cuda::std::size_t __i = 0; __i < _MaxRank; ++__i)
  {
    __perm[__i] = __i;
  }
  ::cuda::std::stable_sort(
    __perm.begin(), __perm.begin() + __tensor.__rank, __stride_compare<_StrideT, _MaxRank>{__tensor.__strides});
  return __perm;
}

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
  const auto __perm    = ::cuda::experimental::__stride_order(__tensor);

  __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank> __result{__tensor.__data, __rank};
  for (__rank_t __i = 0; __i < __rank; ++__i)
  {
    __result.__extents[__i] = __tensor.__extents[__perm[__i]];
    __result.__strides[__i] = __tensor.__strides[__perm[__i]];
  }
  return __result;
}

//! @brief Conservative check for non-unique/interleaved layout.
//!
//! A layout is non-unique/interleaved if there exists two different indices
//! that map to the same element (i.e. `dot(idx1, strides) == dot(idx2, strides)`).
//!
//! The check is exact for layouts that are dense or are a result of common
//! tensor layout transformations (transposing, slicing, broadcasting, reshaping,
//! vectorizing) applied to a dense layout. In general case, it may
//! incorrectly return true for non-interleaved/unique layouts.
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
    using __stride_t      = ::cuda::std::remove_cvref_t<decltype(__sorted.__strides[0])>;
    using __rank_t        = typename _Extents::rank_type;
    const auto& __extents = __sorted.__extents;
    const auto& __strides = __sorted.__strides;
    const auto __rank     = __sorted.__rank;
    for (__rank_t __i = 0; __i < __rank; ++__i)
    {
      if (__extents[__i] == 0)
      {
        // there are no elements in zero-volume layout
        return false;
      }
    }
    // max_dist is maximal distance between two elements in sub-layout
    // restricted to i - 1 dims with smallest strides
    // max_offset = sum((e_j - 1) * s_j for j in range(i) if s_j > 0)
    // min_offset = sum((e_j - 1) * s_j for j in range(i) if s_j < 0)
    // max_dist = max_offset - min_offset = sum((e_j - 1) * |s_j| for j in range(i))
    __stride_t __max_dist = 0;
    for (__rank_t __i = 0; __i < __rank; ++__i)
    {
      const __stride_t extent = __extents[__i];
      if (extent != 1)
      {
        const auto __abs_stride = cudax::__abs_integer(__strides[__i]);
        if (__abs_stride <= __max_dist)
        {
          return true;
        }
        // note that slicing a layout, while it may increase _abs_stride,
        // cannot increase dimension's contribution to max_dist
        __max_dist += (extent - 1) * __abs_stride;
      }
    }
    // Assume for contradiction that there exists two different n-dimensional
    // indices idx1 and idx2 such that `dot(idx1, strides) == dot(idx2, strides)`.
    // Without loss of generality, assume that `idx1[n-1] != idx2[n-1]`
    // (we can ignore the suffix where idx1 and idx2 are equal).
    // We have `dot(idx1, strides) = dot(idx2, strides)`. Rearranging the terms we get:
    // `(idx1[n-1] - idx2[n-1]) * s[n-1] = dot(idx2[:n-1], strides[:n-1]) - dot(idx1[:n-1], strides[:n-1])`.
    // |(idx1[n-1] - idx2[n-1]) * s[n-1]| = |dot(idx2[:n-1], strides[:n-1]) - dot(idx1[:n-1], strides[:n-1])|.
    // Now, |s[n-1]| <= |(idx1[n-1] - idx2[n-1])| * |s[n-1]| = LHS
    // And RHS is a distance between two elements in sub-layout restricted to n-1 dims
    // thus RHS <= max_dist, so we must have returned true for i = n - 1.
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
