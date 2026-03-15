//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_COPY_SIMPLIFY_PAIRED_H
#define __CUDAX_COPY_SIMPLIFY_PAIRED_H

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
#  include <cuda/std/array>
#  include <cuda/std/tuple>

#  include <cuda/experimental/__copy_bytes/tensor_query.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief Reverses the order of active modes in a raw tensor.
//!
//! This helps to get a single logic for __tile_iterator_linearized
//!
//! @param[in] __input Raw tensor whose modes are reversed
//! @return New raw tensor with extents and strides in reversed mode order
template <typename _ExtentT, typename _StrideT, typename _Tp, ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>
__reverse_modes(const __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>& __input) noexcept
{
  using __raw_tensor_t = __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>;
  using __rank_t       = typename __raw_tensor_t::__rank_t;
  __raw_tensor_t __result{__input.__data, __input.__rank, {}, {}};
  _CCCL_ASSERT(__input.__rank > 0, "cudax::reverse_modes: input tensor must have rank > 0");
  for (__rank_t __i = 0; __i < __input.__rank; ++__i)
  {
    const auto __j          = __input.__rank - 1 - __i;
    __result.__extents[__i] = __input.__extents[__j];
    __result.__strides[__i] = __input.__strides[__j];
  }
  return __result;
}

// lambdas are painful without --extended-lambda and when used with __host__ __device__ functions
struct __mode_compare_paired
{
  template <typename _ExtentT, typename _StrideT>
  [[nodiscard]] _CCCL_API bool operator()(const ::cuda::std::tuple<_ExtentT, _StrideT, _StrideT>& __lhs,
                                          const ::cuda::std::tuple<_ExtentT, _StrideT, _StrideT>& __rhs) const noexcept
  {
    namespace cudax      = ::cuda::experimental;
    const auto __src_lhs = ::cuda::std::get<1>(__lhs);
    const auto __src_rhs = ::cuda::std::get<1>(__rhs);
    const auto __dst_lhs = ::cuda::std::get<2>(__lhs);
    const auto __dst_rhs = ::cuda::std::get<2>(__rhs);
    return cudax::__abs_integer(__src_lhs) < cudax::__abs_integer(__src_rhs)
        || (cudax::__abs_integer(__src_lhs) == cudax::__abs_integer(__src_rhs)
            && cudax::__abs_integer(__dst_lhs) < cudax::__abs_integer(__dst_rhs));
  }
};

//! @brief Sorts a source/destination tensor pair by ascending absolute destination stride.
//!
//! Both tensors are reordered in lockstep so that corresponding modes remain paired.
//!
//! @pre @p __src and @p __dst must have the same extents
//!
//! @param[in,out] __src Source raw tensor (modes reordered in place)
//! @param[in,out] __dst Destination raw tensor (modes reordered in place)
template <typename _ExtentT, typename _StrideT, typename _TpSrc, typename _TpDst, ::cuda::std::size_t _MaxRank>
_CCCL_HOST_API void __sort_by_stride_paired(__raw_tensor<_ExtentT, _StrideT, _TpSrc, _MaxRank>& __src,
                                            __raw_tensor<_ExtentT, _StrideT, _TpDst, _MaxRank>& __dst) noexcept
{
  namespace cudax      = ::cuda::experimental;
  using __raw_tensor_t = __raw_tensor<_ExtentT, _StrideT, _TpSrc, _MaxRank>;
  using __rank_t       = typename __raw_tensor_t::__rank_t;
  using __mode_t       = ::cuda::std::tuple<_ExtentT, _StrideT, _StrideT>;
  const auto __rank    = __src.__rank;
  _CCCL_ASSERT(cudax::__same_extents(__src, __dst), "Source and destination tensors must have the same extents");
  ::cuda::std::array<__mode_t, _MaxRank> __modes{};
  for (__rank_t __i = 0; __i < __rank; ++__i)
  {
    __modes[__i] = {__src.__extents[__i], __src.__strides[__i], __dst.__strides[__i]};
  }
  ::cuda::std::stable_sort(__modes.begin(), __modes.begin() + __rank, __mode_compare_paired{});
  for (__rank_t __i = 0; __i < __rank; ++__i)
  {
    ::cuda::std::tie(__src.__extents[__i], __src.__strides[__i], __dst.__strides[__i]) = __modes[__i];
  }
  __dst.__extents = __src.__extents;
}

//! @brief Flips modes where both source and destination strides are negative.
//!
//! For each such mode, the base pointer is advanced to the last element and the stride is negated, yielding an
//! equivalent tensor with positive strides.
//!
//! @pre @p __src and @p __dst must have the same extents
//!
//! @param[in,out] __src Source raw tensor (data pointer and strides may be modified)
//! @param[in,out] __dst Destination raw tensor (data pointer and strides may be modified)
template <typename _ExtentT, typename _StrideT, typename _TpSrc, typename _TpDst, ::cuda::std::size_t _MaxRank>
_CCCL_HOST_API void __flip_negative_strides_paired(__raw_tensor<_ExtentT, _StrideT, _TpSrc, _MaxRank>& __src,
                                                   __raw_tensor<_ExtentT, _StrideT, _TpDst, _MaxRank>& __dst) noexcept
{
  if constexpr (!::cuda::std::is_unsigned_v<_StrideT>)
  {
    using __raw_tensor_t = __raw_tensor<_ExtentT, _StrideT, _TpSrc, _MaxRank>;
    using __rank_t       = typename __raw_tensor_t::__rank_t;
    _CCCL_ASSERT(::cuda::experimental::__same_extents(__src, __dst),
                 "cudax::flip_negative_strides_paired: Source and destination tensors must have the same extents");
    for (__rank_t __i = 0; __i < __src.__rank; ++__i)
    {
      if (__src.__strides[__i] < 0 && __dst.__strides[__i] < 0)
      {
        const auto __extent         = __src.__extents[__i];
        const auto __src_adjustment = static_cast<_StrideT>(__extent - 1) * __src.__strides[__i];
        const auto __dst_adjustment = static_cast<_StrideT>(__extent - 1) * __dst.__strides[__i];
        __src.__data += __src_adjustment;
        __dst.__data += __dst_adjustment;
        __src.__strides[__i] = -__src.__strides[__i];
        __dst.__strides[__i] = -__dst.__strides[__i];
      }
    }
  }
}

//! @brief Merges adjacent modes that are contiguous in both source and destination tensors.
//!
//! Two consecutive modes are merged when `extent[i-1] * stride[i-1] == stride[i]` holds for both tensors.
//! The resulting tensor pair has fewer modes but represents the same memory layout.
//!
//! @pre @p __src and @p __dst must have the same extents
//!
//! @param[in,out] __src Source raw tensor (rank and modes may be reduced)
//! @param[in,out] __dst Destination raw tensor (rank and modes may be reduced)
template <typename _ExtentT, typename _StrideT, typename _TpSrc, typename _TpDst, ::cuda::std::size_t _MaxRank>
_CCCL_HOST_API void __coalesce_paired(__raw_tensor<_ExtentT, _StrideT, _TpSrc, _MaxRank>& __src,
                                      __raw_tensor<_ExtentT, _StrideT, _TpDst, _MaxRank>& __dst) noexcept
{
  _CCCL_ASSERT(::cuda::experimental::__same_extents(__src, __dst),
               "Source and destination tensors must have the same extents");
  if (__src.__rank <= 1)
  {
    return;
  }
  using __raw_tensor_t = __raw_tensor<_ExtentT, _StrideT, _TpSrc, _MaxRank>;
  using __rank_t       = typename __raw_tensor_t::__rank_t;
  __rank_t __out_r     = 1;
  for (__rank_t __i = 1; __i < __src.__rank; ++__i)
  {
    const auto __prev_extent    = static_cast<_StrideT>(__src.__extents[__out_r - 1]);
    const bool __src_contiguous = (__prev_extent * __src.__strides[__out_r - 1] == __src.__strides[__i]);
    const bool __dst_contiguous = (__prev_extent * __dst.__strides[__out_r - 1] == __dst.__strides[__i]);
    if (__src_contiguous && __dst_contiguous)
    {
      __src.__extents[__out_r - 1] *= __src.__extents[__i];
      continue;
    }
    __src.__extents[__out_r] = __src.__extents[__i];
    __src.__strides[__out_r] = __src.__strides[__i];
    __dst.__strides[__out_r] = __dst.__strides[__i];
    ++__out_r;
  }
  __src.__rank    = __out_r;
  __dst.__rank    = __out_r;
  __dst.__extents = __src.__extents;
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // __CUDAX_COPY_SIMPLIFY_PAIRED_H
