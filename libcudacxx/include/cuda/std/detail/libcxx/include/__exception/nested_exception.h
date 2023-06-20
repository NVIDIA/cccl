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

#ifndef _LIBCUDACXX___EXCEPTION_NESTED_EXCEPTION_H
#define _LIBCUDACXX___EXCEPTION_NESTED_EXCEPTION_H

#ifndef __cuda_std__
#include <__config>
#endif //__cuda_std__

#include "../__exception/exception_ptr.h"
#include "../__memory/addressof.h"
#include "../__type_traits/decay.h"
#include "../__type_traits/is_base_of.h"
#include "../__type_traits/is_class.h"
#include "../__type_traits/is_convertible.h"
#include "../__type_traits/is_copy_constructible.h"
#include "../__type_traits/is_final.h"
#include "../__type_traits/is_polymorphic.h"
#include "../__utility/forward.h"
#include "../cstddef"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD_NOVERSION // purposefully not using versioning namespace

class _LIBCUDACXX_EXCEPTION_ABI nested_exception {
  exception_ptr __ptr_;

public:
  _LIBCUDACXX_INLINE_VISIBILITY nested_exception() noexcept;
  nested_exception(const nested_exception&) noexcept = default;
  nested_exception& operator=(const nested_exception&) noexcept = default;
  _LIBCUDACXX_HOST_DEVICE virtual ~nested_exception() noexcept;

  // access functions
  _LIBCUDACXX_NORETURN _LIBCUDACXX_HOST_DEVICE void rethrow_nested() const;
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY exception_ptr nested_ptr() const noexcept { return __ptr_; }
};

template <class _Tp>
struct __nested : public _Tp, public nested_exception {
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY explicit __nested(const _Tp& __t) : _Tp(__t) {}
};

#ifndef _LIBCUDACXX_NO_EXCEPTIONS
template <class _Tp, class _Up, bool>
struct __throw_with_nested;

template <class _Tp, class _Up>
struct __throw_with_nested<_Tp, _Up, true> {
  _LIBCUDACXX_NORETURN static inline _LIBCUDACXX_INLINE_VISIBILITY void __do_throw(_Tp&& __t) {
    throw __nested<_Up>(std::forward<_Tp>(__t));
  }
};

template <class _Tp, class _Up>
struct __throw_with_nested<_Tp, _Up, false> {
  _LIBCUDACXX_NORETURN static inline _LIBCUDACXX_INLINE_VISIBILITY void __do_throw(_Tp&& __t) { throw std::forward<_Tp>(__t); }
};
#endif // !_LIBCUDACXX_NO_EXCEPTIONS

template <class _Tp>
_LIBCUDACXX_NORETURN inline  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
void throw_with_nested(_Tp&& __t) {
#ifndef _LIBCUDACXX_NO_EXCEPTIONS
  using _Up = __decay_t<_Tp>;
  static_assert(is_copy_constructible<_Up>::value, "type thrown must be CopyConstructible");
  __throw_with_nested<_Tp,
                      _Up,
                      is_class<_Up>::value && !is_base_of<nested_exception, _Up>::value &&
                          !__libcpp_is_final<_Up>::value>::__do_throw(std::forward<_Tp>(__t));
#else // ^^^ !_LIBCUDACXX_NO_EXCEPTIONS ^^^ / vvv _LIBCUDACXX_NO_EXCEPTIONS vvv
  ((void)__t);
  _LIBCUDACXX_UNREACHABLE();
#endif // _LIBCUDACXX_NO_EXCEPTIONS
}

template <class _From, class _To>
struct __can_dynamic_cast
    : _BoolConstant< is_polymorphic<_From>::value &&
                     (!is_base_of<_To, _From>::value || is_convertible<const _From*, const _To*>::value)> {};

template <class _Ep>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
void rethrow_if_nested(const _Ep& __e, __enable_if_t< __can_dynamic_cast<_Ep, nested_exception>::value>* = 0) {
  const nested_exception* __nep = dynamic_cast<const nested_exception*>(std::addressof(__e));
  if (__nep)
    __nep->rethrow_nested();
}

template <class _Ep>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
void rethrow_if_nested(const _Ep&, __enable_if_t<!__can_dynamic_cast<_Ep, nested_exception>::value>* = 0) {}

_LIBCUDACXX_END_NAMESPACE_STD_NOVERSION

#endif // _LIBCUDACXX___EXCEPTION_NESTED_EXCEPTION_H
