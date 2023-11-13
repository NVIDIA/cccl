//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_MOVE_SENTINEL_H
#define _LIBCUDACXX___ITERATOR_MOVE_SENTINEL_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__concepts/assignable.h"
#include "../__concepts/convertible_to.h"
#include "../__concepts/semiregular.h"
#include "../__utility/move.h"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 17
template <semiregular _Sent>
#else
template<class _Sent, enable_if_t<semiregular<_Sent>, int> = 0>
#endif
class _LIBCUDACXX_TEMPLATE_VIS move_sentinel
{
public:
  _LIBCUDACXX_HIDE_FROM_ABI
  constexpr move_sentinel() = default;

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  explicit move_sentinel(_Sent __s) : __last_(_CUDA_VSTD::move(__s)) {}

  _LIBCUDACXX_TEMPLATE(class _S2)
    _LIBCUDACXX_REQUIRES( convertible_to<const _S2&, _Sent>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  move_sentinel(const move_sentinel<_S2>& __s) : __last_(__s.base()) {}

  _LIBCUDACXX_TEMPLATE(class _S2)
    _LIBCUDACXX_REQUIRES( assignable_from<const _S2&, _Sent>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  move_sentinel& operator=(const move_sentinel<_S2>& __s)
    { __last_ = __s.base(); return *this; }

  _LIBCUDACXX_INLINE_VISIBILITY constexpr _Sent base() const { return __last_; }

private:
    _Sent __last_ = _Sent();
};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX_STD_VER > 14

#endif // _LIBCUDACXX___ITERATOR_MOVE_SENTINEL_H
