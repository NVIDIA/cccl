//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_COPY_CUTE_MAX_COMMON_LAYOUT_H
#define __CUDAX_COPY_CUTE_MAX_COMMON_LAYOUT_H

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
#  include <cuda/std/__numeric/gcd_lcm.h>
#  include <cuda/std/cstdint>

#  include <cuda/experimental/__copy/cute/utils.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
template <typename _TpA, typename _TpB, ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API constexpr ::cuda::std::size_t __max_common_contiguous_size(
  const __raw_tensor_ordered<_TpA, _MaxRank>& __tensor_a,
  const __raw_tensor_ordered<_TpB, _MaxRank>& __tensor_b) noexcept
{
  using ::cuda::std::int64_t;
  using ::cuda::std::size_t;
  _CCCL_ASSERT(__tensor_a.__rank == __tensor_b.__rank, "The ranks of the tensors must be the same");
  const auto& __shapes_a  = __tensor_a.__shapes;
  const auto& __strides_a = __tensor_a.__strides;
  const auto& __orders_a  = __tensor_a.__orders;
  const auto& __shapes_b  = __tensor_b.__shapes;
  const auto& __strides_b = __tensor_b.__strides;
  const auto& __orders_b  = __tensor_b.__orders;
  size_t __curr_a         = 1;
  size_t __curr_b         = 1;
  for (int __i = static_cast<int>(__tensor_a.__rank) - 1; __i >= 0; --__i)
  {
    if (__orders_a[__i] != __orders_b[__i]
        || (__strides_a[__i] != static_cast<int64_t>(__curr_a) && __strides_a[__i] != 0)
        || (__strides_b[__i] != static_cast<int64_t>(__curr_b) && __strides_b[__i] != 0))
    {
      break;
    }
    __curr_a *= __shapes_a[__i];
    __curr_b *= __shapes_b[__i];
  }
  return ::cuda::std::gcd(__curr_a, __curr_b);
}

template <typename _TpA, typename _TpB, ::cuda::std::size_t _MaxRankA, ::cuda::std::size_t _MaxRankB>
[[nodiscard]] _CCCL_HOST_API constexpr ::cuda::std::size_t __max_common_contiguous_size(
  const __raw_tensor_ordered<_TpA, _MaxRankA>& __tensor_a,
  const __raw_tensor_ordered<_TpB, _MaxRankB>& __tensor_b) noexcept
{
  constexpr auto __rank_max = ::cuda::std::max(_MaxRankA, _MaxRankB);
  const auto __rank_uniform = ::cuda::std::max(__tensor_a.__rank, __tensor_b.__rank);
  const auto __tensor_a1    = ::cuda::experimental::__append<__rank_max>(__tensor_a, __rank_uniform);
  const auto __tensor_b1    = ::cuda::experimental::__append<__rank_max>(__tensor_b, __rank_uniform);
  __println(__tensor_a1);
  __println(__tensor_b1);
  return ::cuda::experimental::__max_common_contiguous_size(__tensor_a1, __tensor_b1);
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // __CUDAX_COPY_CUTE_MAX_COMMON_LAYOUT_H
