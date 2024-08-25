//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_RANGES_MIN_ELEMENT_H
#define _LIBCUDACXX___ALGORITHM_RANGES_MIN_ELEMENT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__functional/identity.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__functional/ranges_operations.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/projected.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/dangling.h>
#include <cuda/std/__utility/forward.h>

#if _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

// TODO(ranges): `ranges::min_element` can now simply delegate to `_CUDA_VSTD::__min_element`.
template <class _Ip, class _Sp, class _Proj, class _Comp>
_LIBCUDACXX_INLINE_VISIBILITY static constexpr _Ip
__min_element_impl(_Ip __first, _Sp __last, _Comp& __comp, _Proj& __proj)
{
  if (__first == __last)
  {
    return __first;
  }

  _Ip __i = __first;
  while (++__i != __last)
  {
    if (_CUDA_VSTD::invoke(__comp, _CUDA_VSTD::invoke(__proj, *__i), _CUDA_VSTD::invoke(__proj, *__first)))
    {
      __first = __i;
    }
  }
  return __first;
}

_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__min_element)
struct __fn
{
  _LIBCUDACXX_TEMPLATE(class _Ip, class _Sp, class _Proj = identity, class _Comp = _CUDA_VRANGES::less)
  _LIBCUDACXX_REQUIRES(forward_iterator<_Ip> _LIBCUDACXX_AND sentinel_for<_Sp, _Ip> _LIBCUDACXX_AND
                         indirect_strict_weak_order<_Comp, projected<_Ip, _Proj>>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr _Ip
  operator()(_Ip __first, _Sp __last, _Comp __comp = {}, _Proj __proj = {}) const
  {
    return _CUDA_VRANGES::__min_element_impl(__first, __last, __comp, __proj);
  }

  _LIBCUDACXX_TEMPLATE(class _Rp, class _Proj = identity, class _Comp = _CUDA_VRANGES::less)
  _LIBCUDACXX_REQUIRES(
    forward_range<_Rp> _LIBCUDACXX_AND indirect_strict_weak_order<_Comp, projected<iterator_t<_Rp>, _Proj>>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr borrowed_iterator_t<_Rp>
  operator()(_Rp&& __r, _Comp __comp = {}, _Proj __proj = {}) const
  {
    return _CUDA_VRANGES::__min_element_impl(_CUDA_VRANGES::begin(__r), _CUDA_VRANGES::end(__r), __comp, __proj);
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo
{
_LIBCUDACXX_CPO_ACCESSIBILITY auto min_element = __min_element::__fn{};
} // namespace __cpo

_LIBCUDACXX_END_NAMESPACE_RANGES

#endif // _CCCL_STD_VER >= 2017 && !_CCCL_COMPILER_MSVC_2017

#endif // _LIBCUDACXX___ALGORITHM_RANGES_MIN_ELEMENT_H
