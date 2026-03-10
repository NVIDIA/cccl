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
#  include <cuda/std/__type_traits/make_unsigned.h>
#  include <cuda/std/array>
#  include <cuda/std/tuple>

#  include <cuda/experimental/__copy_bytes/tensor_query.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
template <typename _ExtentT, typename _StrideT, typename _Tp, ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>
__reverse_modes(const __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>& __input) noexcept
{
  __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank> __result{__input.__data, __input.__rank, {}, {}};
  for (::cuda::std::size_t __i = 0; __i < __input.__rank; ++__i)
  {
    const auto __j          = __input.__rank - 1 - __i;
    __result.__extents[__i] = __input.__extents[__j];
    __result.__strides[__i] = __input.__strides[__j];
  }
  return __result;
}

template <typename _ExtentT, typename _StrideT, typename _TpSrc, typename _TpDst, ::cuda::std::size_t _MaxRank>
_CCCL_HOST_API void __sort_by_stride_paired(__raw_tensor<_ExtentT, _StrideT, _TpSrc, _MaxRank>& __src,
                                            __raw_tensor<_ExtentT, _StrideT, _TpDst, _MaxRank>& __dst) noexcept
{
  namespace cudax = ::cuda::experimental;
  using ::cuda::std::size_t;
  using __unsigned_extent_t = ::cuda::std::make_unsigned_t<_ExtentT>;
  using __mode_t            = ::cuda::std::tuple<__unsigned_extent_t, _StrideT, _StrideT>;
  _CCCL_ASSERT(cudax::__same_extents(__src, __dst), "Source and destination tensors must have the same extents");
  ::cuda::std::array<__mode_t, _MaxRank> __modes{};
  const auto __rank = __src.__rank;
  for (size_t __i = 0; __i < __rank; ++__i)
  {
    __modes[__i] = {__src.__extents[__i], __src.__strides[__i], __dst.__strides[__i]};
  }
  ::cuda::std::stable_sort(__modes.begin(), __modes.begin() + __rank, [](const __mode_t& __lhs, const __mode_t& __rhs) {
    return cudax::__abs_integer(::cuda::std::get<2>(__lhs)) < cudax::__abs_integer(::cuda::std::get<2>(__rhs));
  });
  for (size_t __i = 0; __i < __rank; ++__i)
  {
    ::cuda::std::tie(__src.__extents[__i], __src.__strides[__i], __dst.__strides[__i]) = __modes[__i];
  }
  __dst.__extents = __src.__extents;
}

template <typename _ExtentT, typename _StrideT, typename _TpSrc, typename _TpDst, ::cuda::std::size_t _MaxRank>
_CCCL_HOST_API void __flip_negative_strides_paired(__raw_tensor<_ExtentT, _StrideT, _TpSrc, _MaxRank>& __src,
                                                   __raw_tensor<_ExtentT, _StrideT, _TpDst, _MaxRank>& __dst) noexcept
{
  _CCCL_ASSERT(::cuda::experimental::__same_extents(__src, __dst),
               "Source and destination tensors must have the same extents");
  for (::cuda::std::size_t __i = 0; __i < __src.__rank; ++__i)
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
  ::cuda::std::size_t __out = 1;
  for (::cuda::std::size_t __i = 1; __i < __src.__rank; ++__i)
  {
    const auto __prev_extent    = static_cast<_StrideT>(__src.__extents[__out - 1]);
    const bool __src_contiguous = (__prev_extent * __src.__strides[__out - 1] == __src.__strides[__i]);
    const bool __dst_contiguous = (__prev_extent * __dst.__strides[__out - 1] == __dst.__strides[__i]);
    if (__src_contiguous && __dst_contiguous)
    {
      __src.__extents[__out - 1] *= __src.__extents[__i];
      continue;
    }
    __src.__extents[__out] = __src.__extents[__i];
    __src.__strides[__out] = __src.__strides[__i];
    __dst.__strides[__out] = __dst.__strides[__i];
    ++__out;
  }
  __src.__rank    = __out;
  __dst.__rank    = __out;
  __dst.__extents = __src.__extents;
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // __CUDAX_COPY_SIMPLIFY_PAIRED_H
