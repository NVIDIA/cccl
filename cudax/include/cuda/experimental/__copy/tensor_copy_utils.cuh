//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__COPY_TENSOR_COPY_UTILS_H
#define _CUDAX__COPY_TENSOR_COPY_UTILS_H

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
#  include <cuda/__memory/ptr_alignment.h>
#  include <cuda/__memory/ranges_overlap.h>
#  include <cuda/__utility/in_range.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__mdspan/mdspan.h>
#  include <cuda/std/__memory/is_sufficiently_aligned.h>
#  include <cuda/std/__numeric/gcd_lcm.h>
#  include <cuda/std/__type_traits/conditional.h>
#  include <cuda/std/__type_traits/is_const.h>

#  include <cuda/experimental/__copy/vector_access.cuh>
#  include <cuda/experimental/__copy_bytes/abs_integer.cuh>
#  include <cuda/experimental/__copy_bytes/types.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief Compute the maximum vectorization width in bytes for a raw tensor.
//!
//! Expects mode 0 to be the contiguous mode (stride == 1), as established by
//! @ref __sort_by_stride_paired. Computes the largest power-of-two vector width such that:
//! - The pointer is aligned to that width.
//! - All non-contiguous strides (in bytes) are divisible by it.
//! - The contiguous mode's shape is divisible by the element count.
//! The result is capped at 16 bytes. If mode 0 is not contiguous, returns sizeof(_Tp).
//!
//! @pre `__tensor.__rank` is in [1, _MaxRank].
//! @pre All shapes must be > 1 (no degenerate modes).
//! @pre Strides are sorted by @ref __sort_by_stride_paired (mode 0 has the smallest absolute stride).
//!
//! @param[in] __tensor Raw tensor with strides sorted by @ref __sort_by_stride_paired
//! @return Maximum safe vectorization width in bytes, in [sizeof(_Tp), 16]
template <typename _ExtentT, typename _StrideT, typename _Tp, ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API ::cuda::std::size_t
__max_alignment(const __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>& __tensor) noexcept
{
  using ::cuda::std::size_t;
  using __raw_tensor_t = __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>;
  using __rank_t       = typename __raw_tensor_t::__rank_t;
  _CCCL_ASSERT(::cuda::in_range(__tensor.__rank, size_t{1}, _MaxRank), "Invalid tensor rank");
  if (__tensor.__strides[0] != 1)
  {
    return sizeof(_Tp);
  }
  // (1) pointer alignment
  size_t __alignment = ::cuda::__ptr_alignment(__tensor.__data);
  // (2) alignment over all strides
  for (__rank_t __i = 0; __i < __tensor.__rank; ++__i)
  {
    const auto __stride = ::cuda::experimental::__abs_integer(__tensor.__strides[__i]);
    if (__stride != 1)
    {
      const size_t __stride_bytes = static_cast<size_t>(__stride) * sizeof(_Tp);
      __alignment                 = ::cuda::std::gcd(__alignment, __stride_bytes);
    }
  }
  _CCCL_ASSERT(__alignment % sizeof(_Tp) == 0, "alignment is not a multiple of the element size");
  // (3) Compute the number of items per vector over the contiguous mode
  size_t __elem_alignment = __alignment / sizeof(_Tp);
  __elem_alignment        = ::cuda::std::gcd(__elem_alignment, static_cast<size_t>(__tensor.__extents[0]));
  return __elem_alignment * sizeof(_Tp);
}

template <::cuda::std::size_t _VectorBytes, typename _Tp>
using __reshape_vector_type =
  ::cuda::std::conditional_t<::cuda::std::is_const_v<_Tp>,
                             const ::cuda::experimental::__vector_access_t<_VectorBytes>,
                             ::cuda::experimental::__vector_access_t<_VectorBytes>>;

//! @brief Reshape a raw tensor for vectorized access by widening the element type.
//!
//! @pre Mode 0 must be contiguous (stride == 1).
//! @pre The innermost extent (in bytes) must be divisible by @p _VectorBytes.
//! @pre All non-innermost strides must be divisible by the elements-per-vector ratio.
//!
//! @tparam _VectorBytes Target vector width in bytes
//! @param[in] __tensor Raw tensor with contiguous innermost mode
//! @return Raw tensor with element type replaced by the vector type and adjusted extents/strides
template <::cuda::std::size_t _VectorBytes, typename _ExtentT, typename _StrideT, typename _Tp, ::cuda::std::size_t _MaxRank>
[[nodiscard]]
_CCCL_HOST_API __raw_tensor<_ExtentT, _StrideT, __reshape_vector_type<_VectorBytes, _Tp>, _MaxRank>
__reshape_vectorized(const __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>& __tensor) noexcept
{
  using __vector_t = __reshape_vector_type<_VectorBytes, _Tp>;
  using __rank_t   = typename __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>::__rank_t;
  static_assert(_VectorBytes % sizeof(_Tp) == 0, "vector size must be a multiple of element size");

  constexpr auto __elems_per_vector = _VectorBytes / sizeof(_Tp);
  _CCCL_ASSERT(__tensor.__strides[0] == 1, "innermost mode must be contiguous");
  _CCCL_ASSERT(__tensor.__extents[0] % __elems_per_vector == 0,
               "innermost extent must be divisible by elements per vector");
  _CCCL_ASSERT(::cuda::std::is_sufficiently_aligned<alignof(__vector_t)>(__tensor.__data),
               "tensor data is not sufficiently aligned to the extents and strides");

  const auto __data = reinterpret_cast<__vector_t*>(__tensor.__data);
  __raw_tensor<_ExtentT, _StrideT, __vector_t, _MaxRank> __result{
    __data, __tensor.__rank, __tensor.__extents, __tensor.__strides};
  __result.__extents[0] /= __elems_per_vector;

  for (__rank_t __i = 1; __i < __result.__rank; ++__i)
  {
    _CCCL_ASSERT(__result.__strides[__i] % _StrideT{__elems_per_vector} == 0,
                 "non-innermost strides must be divisible by elements per vector");
    __result.__strides[__i] /= _StrideT{__elems_per_vector};
  }
  return __result;
}

//! @brief Compute the total number of elements in a raw tensor.
//!
//! @param[in] __tensor Raw tensor descriptor
//! @return Product of all extents
template <typename _ExtentT, typename _StrideT, typename _Tp, ::cuda::std::size_t _MaxRank>
[[nodiscard]]
_CCCL_HOST_API _ExtentT __total_size(const __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>& __tensor) noexcept
{
  using __raw_tensor_t  = __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>;
  using __rank_t        = typename __raw_tensor_t::__rank_t;
  _ExtentT __total_size = 1;
  for (__rank_t __i = 0; __i < __tensor.__rank; ++__i)
  {
    __total_size *= __tensor.__extents[__i];
  }
  return __total_size;
}

//! @brief Conservative check whether two mdspans may access overlapping memory.
//!
//! Uses each mapping's @c required_span_size() to compute the half-open byte range
//! @c [data_handle, data_handle + required_span_size * sizeof(T)) and checks for intersection.
//! NOTE: the function doesn't check strict overlap for non-contiguous layouts, for example, the padding could be
//! between the two mdspans.
//!
//! Empty mdspans (size == 0) are considered non-overlapping.
//!
//! @param[in] __a First mdspan
//! @param[in] __b Second mdspan
//! @return true if the byte ranges of the two mdspans overlap
template <typename _Tp1,
          typename _Extents1,
          typename _LayoutPolicy1,
          typename _AccessorPolicy1,
          typename _Tp2,
          typename _Extents2,
          typename _LayoutPolicy2,
          typename _AccessorPolicy2>
[[nodiscard]] _CCCL_HOST_API bool
__may_overlap(const ::cuda::std::mdspan<_Tp1, _Extents1, _LayoutPolicy1, _AccessorPolicy1>& __a,
              const ::cuda::std::mdspan<_Tp2, _Extents2, _LayoutPolicy2, _AccessorPolicy2>& __b) noexcept
{
  if (__a.size() == 0 || __b.size() == 0)
  {
    return false;
  }
  const auto* __a_begin = reinterpret_cast<const char*>(__a.data_handle());
  const auto* __b_begin = reinterpret_cast<const char*>(__b.data_handle());
  const auto* __a_end   = __a_begin + __a.mapping().required_span_size() * sizeof(_Tp1);
  const auto* __b_end   = __b_begin + __b.mapping().required_span_size() * sizeof(_Tp2);
  return ::cuda::ranges_overlap(__a_begin, __a_end, __b_begin, __b_end);
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // _CUDAX__COPY_TENSOR_COPY_UTILS_H
