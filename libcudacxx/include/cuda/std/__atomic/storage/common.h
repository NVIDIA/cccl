// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ATOMIC_STORAGE_COMMON_H
#define _LIBCUDACXX___ATOMIC_STORAGE_COMMON_H

#include <cuda/std/type_traits>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// [atomics.types.generic]p1 guarantees _Tp is trivially copyable. Because
// the default operator= in an object is not volatile, a byte-by-byte copy
// is required.
template <typename _Tp, typename _Tv>
__enable_if_t<is_assignable<_Tp&, _Tv>::value>
_LIBCUDACXX_HOST_DEVICE __atomic_assign_volatile(_Tp& __a_value, _Tv const& __val) {
  __a_value = __val;
}

template <typename _Tp, typename _Tv>
__enable_if_t<is_assignable<_Tp&, _Tv>::value>
_LIBCUDACXX_HOST_DEVICE __atomic_assign_volatile(_Tp volatile& __a_value, _Tv volatile const& __val) {
  volatile char* __to = reinterpret_cast<volatile char*>(&__a_value);
  volatile char* __end = __to + sizeof(_Tp);
  volatile const char* __from = reinterpret_cast<volatile const char*>(&__val);
  while (__to != __end)
    *__to++ = *__from++;
}

template <typename _Tp>
using __atomic_underlying_t = typename __remove_cvref_t<_Tp>::__underlying_t;

template <typename _Tp>
using __atomic_tag_t = typename __remove_cvref_t<_Tp>::__tag_t;

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ATOMIC_STORAGE_COMMON_H
