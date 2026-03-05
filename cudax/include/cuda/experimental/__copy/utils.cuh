//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_COPY_UTILS_H
#define __CUDAX_COPY_UTILS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/std/__algorithm/max.h>
#  include <cuda/std/__algorithm/stable_sort.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__cstdlib/abs.h>
#  include <cuda/std/__mdspan/mdspan.h>
#  include <cuda/std/cstdint>

#  include <cuda/experimental/__copy/types.cuh>

#  include <cstdio>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief Tag constant used to enable extent-1 mode removal in @ref __to_raw_tensor.
inline constexpr auto __remove_extent1 = ::cuda::std::true_type{};

/**
 * @brief Converts an mdspan view to a raw tensor descriptor.
 *
 * The descriptor stores data pointer, rank, extents, and strides in arrays to use them at runtime.
 */
template <typename _Tp, typename _Extents, typename _LayoutPolicy, typename _AccessorPolicy, bool _RemoveExtent1 = false>
[[nodiscard]] _CCCL_HOST_API constexpr __raw_tensor<::cuda::std::size_t, ::cuda::std::int64_t, _Tp, _Extents::rank()>
__to_raw_tensor(const ::cuda::std::mdspan<_Tp, _Extents, _LayoutPolicy, _AccessorPolicy>& __mdspan,
                ::cuda::std::bool_constant<_RemoveExtent1> = {}) noexcept
{
  __raw_tensor<::cuda::std::size_t, ::cuda::std::int64_t, _Tp, _Extents::rank()> __result{
    __mdspan.data_handle(), 0, {}, {}};
  ::cuda::std::size_t __r = 0;
  for (::cuda::std::size_t __i = 0; __i < _Extents::rank(); ++__i)
  {
    const auto __extent = static_cast<::cuda::std::size_t>(__mdspan.extent(__i));
    if (!_RemoveExtent1 || __extent != 1)
    {
      __result.__extents[__r] = __extent;
      __result.__strides[__r] = static_cast<::cuda::std::int64_t>(__mdspan.stride(__i));
      ++__r;
    }
  }
  __result.__rank = __r;
  return __result;
}

/**
 * @brief Promote a raw tensor to a larger compile-time capacity without changing its runtime content.
 *
 * Copies active modes (extents and strides) and zero-initializes the rest.
 */
template <::cuda::std::size_t _TargetMaxRank,
          typename _ExtentT,
          typename _StrideT,
          typename _Tp,
          ::cuda::std::size_t _SrcMaxRank>
[[nodiscard]] _CCCL_HOST_API constexpr __raw_tensor<_ExtentT, _StrideT, _Tp, _TargetMaxRank>
__widen(const __raw_tensor<_ExtentT, _StrideT, _Tp, _SrcMaxRank>& __src) noexcept
{
  static_assert(_TargetMaxRank >= _SrcMaxRank, "Target max rank must be >= source max rank");
  __raw_tensor<_ExtentT, _StrideT, _Tp, _TargetMaxRank> __result{__src.__data, __src.__rank, {}, {}};
  for (::cuda::std::size_t __i = 0; __i < __src.__rank; ++__i)
  {
    __result.__extents[__i] = __src.__extents[__i];
    __result.__strides[__i] = __src.__strides[__i];
  }
  return __result;
}

/**
 * @brief Coalesce adjacent contiguous modes of a raw tensor without reordering.
 *
 * Merges adjacent modes (i, i+1) that are contiguous in either direction:
 * - `extent[i] * stride[i] == stride[i+1]` (mode i is inner), or
 * - `stride[i] == extent[i+1] * stride[i+1]` (mode i+1 is inner).
 *
 * The merged mode keeps the inner stride and the product of extents.
 *
 * @pre All active extents must be > 1 (no degenerate modes).
 */
template <typename _ExtentT, typename _StrideT, typename _Tp, ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API constexpr __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>
__coalesce_adjacent(const __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>& __input) noexcept
{
  if (__input.__rank <= 1)
  {
    return __input;
  }
  __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank> __result{__input.__data, 0, {}, {}};
  __result.__extents[0]     = __input.__extents[0];
  __result.__strides[0]     = __input.__strides[0];
  ::cuda::std::size_t __out = 0;
  for (::cuda::std::size_t __i = 1; __i < __input.__rank; ++__i)
  {
    const auto __prev_extent = static_cast<_StrideT>(__result.__extents[__out]);
    if (__prev_extent * __result.__strides[__out] == __input.__strides[__i])
    {
      __result.__extents[__out] *= __input.__extents[__i];
    }
    else if (__result.__strides[__out] == static_cast<_StrideT>(__input.__extents[__i]) * __input.__strides[__i])
    {
      __result.__extents[__out] *= __input.__extents[__i];
      __result.__strides[__out] = __input.__strides[__i];
    }
    else
    {
      ++__out;
      __result.__extents[__out] = __input.__extents[__i];
      __result.__strides[__out] = __input.__strides[__i];
    }
  }
  __result.__rank = __out + 1;
  return __result;
}

/**
 * @brief Reorders tensor modes by ascending absolute stride.
 *
 * After sorting, mode 0 has the smallest absolute stride (innermost) and mode rank-1 has the largest (outermost).
 */
template <typename _ExtentT, typename _StrideT, typename _Tp, ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API constexpr __raw_tensor_ordered<_ExtentT, _StrideT, _Tp, _MaxRank>
__sort_by_stride(const __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>& __tensor) noexcept
{
  __raw_tensor_ordered<_ExtentT, _StrideT, _Tp, _MaxRank> __result{__tensor.__data, __tensor.__rank};
  auto& __input_strides = __tensor.__strides;
  auto& __orders        = __result.__orders;
  auto& __extents       = __result.__extents;
  auto& __strides       = __result.__strides;
  for (::cuda::std::size_t __i = 0; __i < __tensor.__rank; ++__i)
  {
    __orders[__i] = __i;
  }
  ::cuda::std::stable_sort(__orders.begin(), __orders.begin() + __tensor.__rank, [&](auto __a, auto __b) {
    return ::cuda::std::abs(__input_strides[__a]) < ::cuda::std::abs(__input_strides[__b]);
  });
  for (::cuda::std::size_t __i = 0; __i < __tensor.__rank; ++__i)
  {
    __extents[__i] = __tensor.__extents[__orders[__i]];
    __strides[__i] = __tensor.__strides[__orders[__i]];
  }
  return __result;
}

/**
 * @brief Appends identity modes up to a target rank.
 *
 * Extra modes are represented as shape=1, stride=1, order=i.
 */
template <::cuda::std::size_t _MaxRankOut, typename _ExtentT, typename _StrideT, typename _Tp, ::cuda::std::size_t _MaxRankIn>
[[nodiscard]] _CCCL_HOST_API constexpr __raw_tensor_ordered<_ExtentT, _StrideT, _Tp, _MaxRankOut>
__append(const __raw_tensor_ordered<_ExtentT, _StrideT, _Tp, _MaxRankIn>& __tensor_in,
         ::cuda::std::size_t __target_rank) noexcept
{
  static_assert(_MaxRankIn <= _MaxRankOut);
  __raw_tensor_ordered<_ExtentT, _StrideT, _Tp, _MaxRankOut> __result{__tensor_in.__data, __target_rank};
  for (::cuda::std::size_t __i = 0; __i < __tensor_in.__rank; ++__i)
  {
    __result.__extents[__i] = __tensor_in.__extents[__i];
    __result.__strides[__i] = __tensor_in.__strides[__i];
    __result.__orders[__i]  = __tensor_in.__orders[__i];
  }
  for (::cuda::std::size_t __i = __tensor_in.__rank; __i < __target_rank; ++__i)
  {
    __result.__extents[__i] = 1;
    __result.__strides[__i] = 1;
    __result.__orders[__i]  = __i;
  }
  return __result;
}

/**
 * @brief Conservative uniqueness check for tensor layouts.
 *
 * Sorts modes by ascending absolute stride, then verifies two conditions:
 * 1. No mode with extent > 1 has stride == 0 (broadcast)
 * 2. No mode's span (extent * |stride|) exceeds the next mode's |stride| (overlap)
 *
 * Returns true when non-uniqueness is detected, meaning multiple logical indices may map to the same address.
 */
template <typename _ExtentT, typename _StrideT, typename _Tp, ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API constexpr bool
__is_not_unique(const __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>& __tensor) noexcept
{
  if constexpr (_MaxRank > 0)
  {
    const auto __sorted   = ::cuda::experimental::__sort_by_stride(__tensor);
    const auto& __extents = __sorted.__extents;
    const auto& __strides = __sorted.__strides;
    for (::cuda::std::size_t __i = 0; __i < __sorted.__rank; ++__i)
    {
      if (__extents[__i] > 1 && __strides[__i] == 0)
      {
        return true;
      }
    }
    for (::cuda::std::size_t __i = 0; __i + 1 < __sorted.__rank; ++__i)
    {
      if (static_cast<_StrideT>(__extents[__i]) * ::cuda::std::abs(__strides[__i])
          > ::cuda::std::abs(__strides[__i + 1]))
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

/// @brief Convenience overload: converts an mdspan to a raw tensor, then checks uniqueness.
template <typename _Tp, typename _Extents, typename _LayoutPolicy, typename _AccessorPolicy>
[[nodiscard]] _CCCL_HOST_API constexpr bool
__is_not_unique(const ::cuda::std::mdspan<_Tp, _Extents, _LayoutPolicy, _AccessorPolicy>& __mdspan) noexcept
{
  if constexpr (_Extents::rank() > 0)
  {
    using __stride_t      = ::cuda::std::int64_t; // TODO: generalize
    const auto __tensor   = ::cuda::experimental::__to_raw_tensor(__mdspan, __remove_extent1);
    const auto __sorted   = ::cuda::experimental::__sort_by_stride(__tensor);
    const auto& __extents = __sorted.__extents;
    const auto& __strides = __sorted.__strides;
    for (::cuda::std::size_t __i = 0; __i < __sorted.__rank; ++__i)
    {
      if (__strides[__i] == 0)
      {
        return true;
      }
    }
    for (::cuda::std::size_t __i = 0; __i + 1 < __sorted.__rank; ++__i)
    {
      const auto __extent = static_cast<__stride_t>(__extents[__i]);
      if (__extent * ::cuda::std::abs(__strides[__i]) > ::cuda::std::abs(__strides[__i + 1]))
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

/// @brief Checks whether two tensors have the same stride-based mode order.
template <typename _ExtentT,
          typename _StrideT,
          typename _TpIn,
          typename _TpOut,
          ::cuda::std::size_t _MaxRankA,
          ::cuda::std::size_t _MaxRankB>
[[nodiscard]] _CCCL_HOST_API constexpr bool
__same_stride_order(const __raw_tensor_ordered<_ExtentT, _StrideT, _TpIn, _MaxRankA>& __tensor_a,
                    const __raw_tensor_ordered<_ExtentT, _StrideT, _TpOut, _MaxRankB>& __tensor_b) noexcept
{
  const auto __rank_a       = __tensor_a.__rank;
  const auto __rank_b       = __tensor_b.__rank;
  const auto __rank_uniform = ::cuda::std::max(__rank_a, __rank_b);
  for (::cuda::std::size_t __i = 0; __i < __rank_uniform; ++__i)
  {
    const auto __order_a = __i < __rank_a ? __tensor_a.__orders[__i] : __i;
    const auto __order_b = __i < __rank_b ? __tensor_b.__orders[__i] : __i;
    if (__order_a != __order_b)
    {
      return false;
    }
  }
  return true;
}

template <typename _ExtentT, typename _StrideT, typename _Tp, ::cuda::std::size_t _Rank>
_CCCL_HOST_API void __println(const __raw_tensor<_ExtentT, _StrideT, _Tp, _Rank>& __tensor)
{
  const auto __rank = static_cast<int>(__tensor.__rank);
  ::printf("(");
  for (int __i = 0; __i < __rank - 1; ++__i)
  {
    ::printf("%llu, ", static_cast<unsigned long long>(__tensor.__extents[__i]));
  }
  if (__rank > 0)
  {
    ::printf("%llu", static_cast<unsigned long long>(__tensor.__extents[__rank - 1]));
  }
  ::printf("):(");
  for (int __i = 0; __i < __rank - 1; ++__i)
  {
    ::printf("%lld, ", static_cast<long long>(__tensor.__strides[__i]));
  }
  if (__rank > 0)
  {
    ::printf("%lld", static_cast<long long>(__tensor.__strides[__rank - 1]));
  }
  ::printf(")\n");
}

template <typename _ExtentT, typename _StrideT, typename _Tp, ::cuda::std::size_t _Rank>
_CCCL_HOST_API void __println(const __raw_tensor_ordered<_ExtentT, _StrideT, _Tp, _Rank>& __tensor)
{
  const auto __rank = static_cast<int>(__tensor.__rank);
  ::printf("(");
  for (int __i = 0; __i < __rank - 1; ++__i)
  {
    ::printf("%llu, ", static_cast<unsigned long long>(__tensor.__extents[__i]));
  }
  if (__rank > 0)
  {
    ::printf("%llu", static_cast<unsigned long long>(__tensor.__extents[__rank - 1]));
  }
  ::printf("):(");
  for (int __i = 0; __i < __rank - 1; ++__i)
  {
    ::printf("%lld, ", static_cast<long long>(__tensor.__strides[__i]));
  }
  if (__rank > 0)
  {
    ::printf("%lld", static_cast<long long>(__tensor.__strides[__rank - 1]));
  }
  ::printf(") perm:(");
  for (int __i = 0; __i < __rank - 1; ++__i)
  {
    ::printf("%zu,", __tensor.__orders[__i]);
  }
  if (__rank > 0)
  {
    ::printf("%zu", __tensor.__orders[__rank - 1]);
  }
  ::printf(")\n");
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // __CUDAX_COPY_UTILS_H
