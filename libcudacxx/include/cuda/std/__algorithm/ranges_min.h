//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_RANGES_MIN_H
#define _LIBCUDACXX___ALGORITHM_RANGES_MIN_H

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

_CCCL_PUSH_MACROS

#if _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__min)

struct __fn
{
  _LIBCUDACXX_TEMPLATE(class _Tp, class _Proj = identity, class _Comp = _CUDA_VRANGES::less)
  _LIBCUDACXX_REQUIRES(indirect_strict_weak_order<_Comp, projected<const _Tp*, _Proj>>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr const _Tp&
  operator()(const _Tp& __a, const _Tp& __b, _Comp __comp = {}, _Proj __proj = {}) const
  {
    return _CUDA_VSTD::invoke(__comp, _CUDA_VSTD::invoke(__proj, __b), _CUDA_VSTD::invoke(__proj, __a)) ? __b : __a;
  }

  _LIBCUDACXX_TEMPLATE(class _Tp, class _Proj = identity, class _Comp = _CUDA_VRANGES::less)
  _LIBCUDACXX_REQUIRES(indirect_strict_weak_order<_Comp, projected<const _Tp*, _Proj>>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp
  operator()(initializer_list<_Tp> __il, _Comp __comp = {}, _Proj __proj = {}) const
  {
    _CCCL_ASSERT(__il.begin() != __il.end(), "initializer_list must contain at least one element");
    return *_CUDA_VSTD::__min_element(__il.begin(), __il.end(), __comp, __proj);
  }

  _LIBCUDACXX_TEMPLATE(class _Rp, class _Proj = identity, class _Comp = _CUDA_VRANGES::less)
  _LIBCUDACXX_REQUIRES(
    input_range<_Rp> _LIBCUDACXX_AND indirect_strict_weak_order<_Comp, projected<iterator_t<_Rp>, _Proj>>
      _LIBCUDACXX_AND indirectly_copyable_storable<iterator_t<_Rp>, range_value_t<_Rp>*>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr range_value_t<_Rp>
  operator()(_Rp&& __r, _Comp __comp = {}, _Proj __proj = {}) const
  {
    auto __first = _CUDA_VRANGES::begin(__r);
    auto __last  = _CUDA_VRANGES::end(__r);

    _CCCL_ASSERT(__first != __last, "range must contain at least one element");

    if constexpr (forward_range<_Rp>)
    {
      return *_CUDA_VSTD::__min_element(__first, __last, __comp, __proj);
    }
    else
    {
      range_value_t<_Rp> __result = *__first;
      while (++__first != __last)
      {
        if (_CUDA_VSTD::invoke(__comp, _CUDA_VSTD::invoke(__proj, *__first), _CUDA_VSTD::invoke(__proj, __result)))
        {
          __result = *__first;
        }
      }
      return __result;
    }
    _CCCL_UNREACHABLE();
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto min = __min::__fn{};
} // namespace __cpo

_LIBCUDACXX_END_NAMESPACE_RANGES

#endif // _CCCL_STD_VER >= 2017 && !_CCCL_COMPILER_MSVC_2017

_CCCL_POP_MACROS

#endif // _LIBCUDACXX___ALGORITHM_RANGES_MIN_H
