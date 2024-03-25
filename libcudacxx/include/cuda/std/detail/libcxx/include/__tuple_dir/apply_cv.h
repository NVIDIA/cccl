//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TUPLE_APPLY_CV_H
#define _LIBCUDACXX___TUPLE_APPLY_CV_H

#ifndef __cuda_std__
#  include <__config>
#endif // __cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include "../__type_traits/is_const.h"
#include "../__type_traits/is_reference.h"
#include "../__type_traits/is_volatile.h"
#include "../__type_traits/remove_reference.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <bool _ApplyLV, bool _ApplyConst, bool _ApplyVolatile>
struct __apply_cv_mf;
template <>
struct __apply_cv_mf<false, false, false>
{
  template <class _Tp>
  using __apply = _Tp;
};
template <>
struct __apply_cv_mf<false, true, false>
{
  template <class _Tp>
  using __apply _LIBCUDACXX_NODEBUG_TYPE = const _Tp;
};
template <>
struct __apply_cv_mf<false, false, true>
{
  template <class _Tp>
  using __apply _LIBCUDACXX_NODEBUG_TYPE = volatile _Tp;
};
template <>
struct __apply_cv_mf<false, true, true>
{
  template <class _Tp>
  using __apply _LIBCUDACXX_NODEBUG_TYPE = const volatile _Tp;
};
template <>
struct __apply_cv_mf<true, false, false>
{
  template <class _Tp>
  using __apply _LIBCUDACXX_NODEBUG_TYPE = _Tp&;
};
template <>
struct __apply_cv_mf<true, true, false>
{
  template <class _Tp>
  using __apply _LIBCUDACXX_NODEBUG_TYPE = const _Tp&;
};
template <>
struct __apply_cv_mf<true, false, true>
{
  template <class _Tp>
  using __apply _LIBCUDACXX_NODEBUG_TYPE = volatile _Tp&;
};
template <>
struct __apply_cv_mf<true, true, true>
{
  template <class _Tp>
  using __apply _LIBCUDACXX_NODEBUG_TYPE = const volatile _Tp&;
};
template <class _Tp, class _RawTp = __libcpp_remove_reference_t<_Tp> >
using __apply_cv_t _LIBCUDACXX_NODEBUG_TYPE =
    __apply_cv_mf<_LIBCUDACXX_TRAIT(is_lvalue_reference, _Tp),
                  _LIBCUDACXX_TRAIT(is_const, _RawTp),
                  _LIBCUDACXX_TRAIT(is_volatile, _RawTp)>;

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TUPLE_APPLY_CV_H
