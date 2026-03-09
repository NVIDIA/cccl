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
#  include <cuda/std/__mdspan/mdspan.h>
#  include <cuda/std/__type_traits/common_type.h>
#  include <cuda/std/__type_traits/make_signed.h>
#  include <cuda/std/__type_traits/void_t.h>
#  include <cuda/std/cstdint>

#  include <cuda/experimental/__copy/abs_integer.cuh>
#  include <cuda/experimental/__copy/types.cuh>

#  include <cstdio>

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

template <typename _ExtentOut,
          typename _StrideOut,
          typename _Tp,
          ::cuda::std::size_t _MaxRank,
          typename _ExtentIn,
          typename _StrideIn>
[[nodiscard]] _CCCL_HOST_API constexpr __raw_tensor<_ExtentOut, _StrideOut, _Tp, _MaxRank>
__cast_raw_tensor(const __raw_tensor<_ExtentIn, _StrideIn, _Tp, _MaxRank>& __src) noexcept
{
  using __raw_tensor_out_t = __raw_tensor<_ExtentOut, _StrideOut, _Tp, _MaxRank>;
  using __extent_out_t     = typename __raw_tensor_out_t::__unsigned_extent_t;
  __raw_tensor_out_t __result{__src.__data, __src.__rank, {}, {}};
  for (::cuda::std::size_t __i = 0; __i < __src.__rank; ++__i)
  {
    __result.__extents[__i] = static_cast<__extent_out_t>(__src.__extents[__i]);
    __result.__strides[__i] = static_cast<_StrideOut>(__src.__strides[__i]);
  }
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
  namespace cudax = ::cuda::experimental;
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
    return cudax::__abs_integer(__input_strides[__a]) < cudax::__abs_integer(__input_strides[__b]);
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
 * @brief Conservative check for interleaved stride order in tensor layouts.
 *
 * Sorts modes by ascending absolute stride, then verifies two conditions:
 * 1. No mode with extent > 1 has stride == 0 (broadcast)
 * 2. No mode's span (extent * |stride|) exceeds the next mode's |stride|
 *
 * Returns true when the layout fails this non-interleaving rule. This is stronger than
 * a mathematical injectivity check and may reject some layouts with distinct offsets.
 */
template <typename _ExtentT, typename _StrideT, typename _Tp, ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API constexpr bool
__has_interleaved_stride_order(const __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>& __tensor) noexcept
{
  if constexpr (_MaxRank > 0)
  {
    using __stride_t      = ::cuda::std::common_type_t<::cuda::std::int64_t, ::cuda::std::make_signed_t<_StrideT>>;
    const auto __tensor_s = ::cuda::experimental::__cast_raw_tensor<_ExtentT, __stride_t>(__tensor);
    const auto __sorted   = ::cuda::experimental::__sort_by_stride(__tensor_s);
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
      if (static_cast<__stride_t>(__extents[__i]) * ::cuda::experimental::__abs_integer(__strides[__i])
          > ::cuda::experimental::__abs_integer(__strides[__i + 1]))
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

/// @brief Convenience overload: converts an mdspan to a raw tensor, then checks stride interleaving.
template <typename _Tp, typename _Extents, typename _LayoutPolicy, typename _AccessorPolicy>
[[nodiscard]] _CCCL_HOST_API constexpr bool __has_interleaved_stride_order(
  const ::cuda::std::mdspan<_Tp, _Extents, _LayoutPolicy, _AccessorPolicy>& __mdspan) noexcept
{
  return ::cuda::experimental::__has_interleaved_stride_order(
    ::cuda::experimental::__to_raw_tensor(__mdspan, __remove_extent1));
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
  const auto __rank_a = __tensor_a.__rank;
  const auto __rank_b = __tensor_b.__rank;
  if (__rank_a != __rank_b)
  {
    return false;
  }
  for (::cuda::std::size_t __i = 0; __i < __rank_a; ++__i)
  {
    const auto __order_a = __tensor_a.__orders[__i];
    const auto __order_b = __tensor_b.__orders[__i];
    if (__order_a != __order_b)
    {
      return false;
    }
  }
  return true;
}

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
  for (::cuda::std::size_t __i = 0; __i < __tensor_in.__rank; ++__i)
  {
    if (__tensor_in.__extents[__i] != __tensor_out.__extents[__i])
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
