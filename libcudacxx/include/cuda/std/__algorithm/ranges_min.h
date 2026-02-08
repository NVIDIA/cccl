//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ALGORITHM_RANGES_MIN_H
#define _CUDA_STD___ALGORITHM_RANGES_MIN_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/min_element.h>
#include <cuda/std/__concepts/copyable.h>
#include <cuda/std/__functional/identity.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__functional/ranges_operations.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/projected.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/initializer_list>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_RANGES
_CCCL_BEGIN_NAMESPACE_CPO(__min)

struct __fn
{
  _CCCL_TEMPLATE(class _Tp, class _Proj = identity, class _Comp = ::cuda::std::ranges::less)
  _CCCL_REQUIRES(indirect_strict_weak_order<_Comp, projected<const _Tp*, _Proj>>)
  [[nodiscard]] _CCCL_API constexpr const _Tp&
  operator()(const _Tp& __a, const _Tp& __b, _Comp __comp = {}, _Proj __proj = {}) const
  {
    return ::cuda::std::invoke(__comp, ::cuda::std::invoke(__proj, __b), ::cuda::std::invoke(__proj, __a)) ? __b : __a;
  }

  _CCCL_TEMPLATE(class _Tp, class _Proj = identity, class _Comp = ::cuda::std::ranges::less)
  _CCCL_REQUIRES(indirect_strict_weak_order<_Comp, projected<const _Tp*, _Proj>>)
  [[nodiscard]] _CCCL_API constexpr _Tp
  operator()(initializer_list<_Tp> __il, _Comp __comp = {}, _Proj __proj = {}) const
  {
    _CCCL_ASSERT(__il.begin() != __il.end(), "initializer_list must contain at least one element");
    return *::cuda::std::__min_element(__il.begin(), __il.end(), __comp, __proj);
  }

  _CCCL_TEMPLATE(class _Rp, class _Proj = identity, class _Comp = ::cuda::std::ranges::less)
  _CCCL_REQUIRES(input_range<_Rp> _CCCL_AND indirect_strict_weak_order<_Comp, projected<iterator_t<_Rp>, _Proj>>
                   _CCCL_AND indirectly_copyable_storable<iterator_t<_Rp>, range_value_t<_Rp>*>)
  [[nodiscard]] _CCCL_API constexpr range_value_t<_Rp> operator()(_Rp&& __r, _Comp __comp = {}, _Proj __proj = {}) const
  {
    auto __first = ::cuda::std::ranges::begin(__r);
    auto __last  = ::cuda::std::ranges::end(__r);

    _CCCL_ASSERT(__first != __last, "range must contain at least one element");

    if constexpr (forward_range<_Rp>)
    {
      return *::cuda::std::__min_element(__first, __last, __comp, __proj);
    }
    else
    {
      range_value_t<_Rp> __result = *__first;
      while (++__first != __last)
      {
        if (::cuda::std::invoke(__comp, ::cuda::std::invoke(__proj, *__first), ::cuda::std::invoke(__proj, __result)))
        {
          __result = *__first;
        }
      }
      return __result;
    }
  }
};
_CCCL_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto min = __min::__fn{};
} // namespace __cpo

_CCCL_END_NAMESPACE_CUDA_STD_RANGES

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ALGORITHM_RANGES_MIN_H
