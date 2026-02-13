//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_COPY_CUTE_UTILS_H
#define __CUDAX_COPY_CUTE_UTILS_H

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
#  include <cuda/std/cstdint>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
template <typename _Tp, ::cuda::std::size_t _MaxRank>
struct __raw_tensor
{
  _Tp* __data;
  ::cuda::std::size_t __rank;
  ::cuda::std::array<::cuda::std::size_t, _MaxRank> __shapes;
  ::cuda::std::array<::cuda::std::int64_t, _MaxRank> __strides;
};

template <typename _Tp, ::cuda::std::size_t _MaxRank>
struct __raw_tensor_ordered
{
  _Tp* __data;
  ::cuda::std::size_t __rank;
  ::cuda::std::array<::cuda::std::size_t, _MaxRank> __shapes;
  ::cuda::std::array<::cuda::std::int64_t, _MaxRank> __strides;
  ::cuda::std::array<::cuda::std::size_t, _MaxRank> __orders;
};

template <typename _Tp, typename _Extents, typename _LayoutPolicy, typename _AccessorPolicy>
[[nodiscard]] _CCCL_HOST_API constexpr __raw_tensor<_Tp, _Extents::rank()>
__to_raw_tensor(const ::cuda::std::mdspan<_Tp, _Extents, _LayoutPolicy, _AccessorPolicy>& __mdspan) noexcept
{
  __raw_tensor<_Tp, _Extents::rank()> __result{__mdspan.data_handle(), _Extents::rank()};
  for (::cuda::std::size_t __i = 0; __i < _Extents::rank(); ++__i)
  {
    __result.__shapes[__i]  = static_cast<::cuda::std::size_t>(__mdspan.extent(__i));
    __result.__strides[__i] = static_cast<::cuda::std::int64_t>(__mdspan.stride(__i));
  }
  return __result;
}

template <typename _Tp, ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API constexpr __raw_tensor_ordered<_Tp, _MaxRank>
__sort_by_stride_desc(const __raw_tensor<_Tp, _MaxRank>& __tensor) noexcept
{
  __raw_tensor_ordered<_Tp, _MaxRank> __result{__tensor.__data, __tensor.__rank};
  auto& __input_strides = __tensor.__strides;
  auto& __orders        = __result.__orders;
  auto& __shapes        = __result.__shapes;
  auto& __strides       = __result.__strides;
  for (::cuda::std::size_t __i = 0; __i < __tensor.__rank; ++__i)
  {
    __orders[__i] = __i;
  }
  // Sort by strides
  ::cuda::std::stable_sort(__orders.begin(), __orders.begin() + __tensor.__rank, [&](auto __a, auto __b) {
    return ::cuda::std::abs(__input_strides[__a]) > ::cuda::std::abs(__input_strides[__b]); // descending order
  });
  for (::cuda::std::size_t __i = 0; __i < __tensor.__rank; ++__i)
  {
    __shapes[__i]  = __tensor.__shapes[__orders[__i]];
    __strides[__i] = __tensor.__strides[__orders[__i]];
  }
  return __result;
}

template <::cuda::std::size_t _MaxRankOut, typename _Tp, ::cuda::std::size_t _MaxRankIn>
[[nodiscard]] _CCCL_HOST_API constexpr __raw_tensor_ordered<_Tp, _MaxRankOut>
__append(const __raw_tensor_ordered<_Tp, _MaxRankIn>& __tensor_in, ::cuda::std::size_t __rank_out) noexcept
{
  static_assert(_MaxRankIn <= _MaxRankOut);
  __raw_tensor_ordered<_Tp, _MaxRankOut> __result{__tensor_in.__data, __rank_out};
  for (::cuda::std::size_t __i = 0; __i < __tensor_in.__rank; ++__i)
  {
    __result.__shapes[__i]  = __tensor_in.__shapes[__i];
    __result.__strides[__i] = __tensor_in.__strides[__i];
    __result.__orders[__i]  = __tensor_in.__orders[__i];
  }
  for (::cuda::std::size_t __i = __tensor_in.__rank; __i < __rank_out; ++__i)
  {
    __result.__shapes[__i]  = 1;
    __result.__strides[__i] = 1;
    __result.__orders[__i]  = __i;
  }
  return __result;
}

template <typename _TpIn, typename _TpOut, ::cuda::std::size_t _Rank>
[[nodiscard]] _CCCL_HOST_API constexpr bool __same_stride_order(
  const __raw_tensor_ordered<_TpIn, _Rank>& __tensor_a, const __raw_tensor_ordered<_TpOut, _Rank>& __tensor_b) noexcept
{
  return __tensor_a.__rank == __tensor_b.__rank && __tensor_a.__orders == __tensor_b.__orders;
}

template <typename _Tp, ::cuda::std::size_t _Rank>
_CCCL_HOST_API constexpr void __println(const __raw_tensor<_Tp, _Rank>& __tensor)
{
  const auto __rank = static_cast<int>(__tensor.__rank);
  printf("(");
  for (int __i = 0; __i < __rank - 1; ++__i)
  {
    printf("%zu,", __tensor.__shapes[__i]);
  }
  if (__rank > 0)
  {
    printf("%zu", __tensor.__shapes[__rank - 1]);
  }
  printf("):(");
  for (int __i = 0; __i < __rank - 1; ++__i)
  {
    printf("%zu,", __tensor.__strides[__i]);
  }
  if (__rank > 0)
  {
    printf("%zu", __tensor.__strides[__rank - 1]);
  }
  printf(")\n");
}

template <typename _Tp, ::cuda::std::size_t _Rank>
_CCCL_HOST_API constexpr void __println(const __raw_tensor_ordered<_Tp, _Rank>& __tensor)
{
  const auto __rank = static_cast<int>(__tensor.__rank);
  printf("(");
  for (int __i = 0; __i < __rank - 1; ++__i)
  {
    printf("%zu,", __tensor.__shapes[__i]);
  }
  if (__rank > 0)
  {
    printf("%zu", __tensor.__shapes[__rank - 1]);
  }
  printf("):(");
  for (int __i = 0; __i < __rank - 1; ++__i)
  {
    printf("%zu,", __tensor.__strides[__i]);
  }
  if (__rank > 0)
  {
    printf("%zu", __tensor.__strides[__rank - 1]);
  }
  printf(") perm:(");
  for (int __i = 0; __i < __rank - 1; ++__i)
  {
    printf("%zu,", __tensor.__orders[__i]);
  }
  if (__rank > 0)
  {
    printf("%zu", __tensor.__orders[__rank - 1]);
  }
  printf(")\n");
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // __CUDAX_COPY_CUTE_UTILS_H
