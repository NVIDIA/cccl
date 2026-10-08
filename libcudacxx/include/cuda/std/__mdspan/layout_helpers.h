// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___MDSPAN_LAYOUT_HELPERS_H
#define _CUDA_STD___MDSPAN_LAYOUT_HELPERS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__numeric/mul_overflow.h>
#include <cuda/__numeric/overflow_result.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

namespace __mdspan_detail
{
// Rank 0 is a single-element index space. Any zero extent makes the space empty.
template <class _Extents>
[[nodiscard]] _CCCL_API constexpr bool __is_empty_extents(const _Extents& __ext) noexcept
{
  using _RankType  = typename _Extents::rank_type;
  using _IndexType = typename _Extents::index_type;
  if constexpr (_Extents::rank() == 0)
  {
    return false;
  }
  else
  {
    for (_RankType __r = 0; __r != _Extents::rank(); ++__r)
    {
      if (__ext.extent(__r) == _IndexType{0})
      {
        return true;
      }
    }
    return false;
  }
}

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4702)

// Product of extents in [__begin, __end) as _IndexType.
// An empty range yields 1. A zero extent in range yields 0 without overflow.
template <class _IndexType, class _Extents>
[[nodiscard]] _CCCL_API constexpr ::cuda::overflow_result<_IndexType> __extents_product(
  const _Extents& __ext,
  typename _Extents::rank_type __begin = 0,
  typename _Extents::rank_type __end   = _Extents::rank()) noexcept
{
  using _RankType = typename _Extents::rank_type;
  if (__begin == __end)
  {
    return {_IndexType{1}, false};
  }

  for (_RankType __r = __begin; __r != __end; ++__r)
  {
    if (__ext.extent(__r) == typename _Extents::index_type{0})
    {
      return {_IndexType{0}, false};
    }
  }

  _IndexType __prod = 1;
  for (_RankType __r = __begin; __r != __end; ++__r)
  {
    if (::cuda::mul_overflow(__prod, __prod, __ext.extent(__r)))
    {
      return {__prod, true};
    }
  }
  return {__prod, false};
}
_CCCL_DIAG_POP

template <class _IndexType, class _Extents>
[[nodiscard]] _CCCL_API constexpr _IndexType __extents_product_value(
  const _Extents& __ext,
  typename _Extents::rank_type __begin = 0,
  typename _Extents::rank_type __end   = _Extents::rank()) noexcept
{
  const auto __prod = ::cuda::std::__mdspan_detail::__extents_product<_IndexType>(__ext, __begin, __end);
  return __prod.overflow ? _IndexType{0} : __prod.value;
}

// layout_left / layout_right required_span_size is the product of the extents.
template <class _Extents>
[[nodiscard]] _CCCL_API constexpr bool __required_span_size_is_representable(const _Extents& __ext) noexcept
{
  using _IndexType = typename _Extents::index_type;
  return !::cuda::std::__mdspan_detail::__extents_product<_IndexType>(__ext).overflow;
}
} // namespace __mdspan_detail

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___MDSPAN_LAYOUT_HELPERS_H
