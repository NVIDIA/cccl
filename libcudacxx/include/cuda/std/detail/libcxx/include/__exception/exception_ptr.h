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

#ifndef _LIBCUDACXX___EXCEPTION_EXCEPTION_PTR_H
#define _LIBCUDACXX___EXCEPTION_EXCEPTION_PTR_H

#ifndef __cuda_std__
#include <__config>
#endif //__cuda_std__

#include "../__exception/operations.h"
#include "../__memory/addressof.h"
#include "../cstddef"
#include "../cstdlib"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD_NOVERSION // purposefully not using versioning namespace

#ifndef _LIBCUDACXX_ABI_MICROSOFT

class _LIBCUDACXX_TYPE_VIS exception_ptr {
  void* __ptr_;

public:
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY exception_ptr() noexcept : __ptr_() {}
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY exception_ptr(nullptr_t) noexcept : __ptr_() {}

  _LIBCUDACXX_HOST_DEVICE exception_ptr(const exception_ptr&) noexcept;
  _LIBCUDACXX_HOST_DEVICE exception_ptr& operator=(const exception_ptr&) noexcept;
  _LIBCUDACXX_HOST_DEVICE ~exception_ptr() noexcept;

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY explicit operator bool() const noexcept { return __ptr_ != nullptr; }

  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  bool operator==(const exception_ptr& __x, const exception_ptr& __y) noexcept {
    return __x.__ptr_ == __y.__ptr_;
  }

  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  bool operator!=(const exception_ptr& __x, const exception_ptr& __y) noexcept {
    return !(__x == __y);
  }

  friend _LIBCUDACXX_FUNC_VIS exception_ptr current_exception() noexcept;
  friend _LIBCUDACXX_FUNC_VIS void rethrow_exception(exception_ptr);
};

template <class _Ep>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
exception_ptr make_exception_ptr(_Ep __e) noexcept {
#ifndef _LIBCUDACXX_NO_EXCEPTIONS
  try {
    throw __e;
  } catch (...) {
    return current_exception();
  }
#else // ^^^ !_LIBCUDACXX_NO_EXCEPTIONS ^^^ / vvv _LIBCUDACXX_NO_EXCEPTIONS vvv
  ((void)__e);
  _LIBCUDACXX_UNREACHABLE();
#endif // _LIBCUDACXX_NO_EXCEPTIONS
}

#else // ^^^ !_LIBCUDACXX_ABI_MICROSOFT ^^^ / vvv _LIBCUDACXX_ABI_MICROSOFT vvv

class _LIBCUDACXX_TYPE_VIS exception_ptr {
  _LIBCUDACXX_DIAGNOSTIC_PUSH
  _LIBCUDACXX_CLANG_DIAGNOSTIC_IGNORED("-Wunused-private-field")
  void* __ptr1_;
  void* __ptr2_;
  _LIBCUDACXX_DIAGNOSTIC_POP

public:
  _LIBCUDACXX_HOST_DEVICE exception_ptr() noexcept;
  _LIBCUDACXX_HOST_DEVICE exception_ptr(nullptr_t) noexcept;
  _LIBCUDACXX_HOST_DEVICE exception_ptr(const exception_ptr& __other) noexcept;
  _LIBCUDACXX_HOST_DEVICE exception_ptr& operator=(const exception_ptr& __other) noexcept;
  _LIBCUDACXX_HOST_DEVICE exception_ptr& operator=(nullptr_t) noexcept;
  _LIBCUDACXX_HOST_DEVICE ~exception_ptr() noexcept;
  _LIBCUDACXX_HOST_DEVICE explicit operator bool() const noexcept;
};

_LIBCUDACXX_FUNC_VIS
bool operator==(const exception_ptr& __x, const exception_ptr& __y) noexcept;

inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
bool operator!=(const exception_ptr& __x, const exception_ptr& __y) noexcept {
  return !(__x == __y);
}

_LIBCUDACXX_FUNC_VIS void swap(exception_ptr&, exception_ptr&) noexcept;

_LIBCUDACXX_FUNC_VIS exception_ptr __copy_exception_ptr(void* __except, const void* __ptr);
_LIBCUDACXX_FUNC_VIS exception_ptr current_exception() noexcept;
_LIBCUDACXX_NORETURN _LIBCUDACXX_FUNC_VIS void rethrow_exception(exception_ptr p);

// This is a built-in template function which automagically extracts the required
// information.
template <class _E>
_LIBCUDACXX_HOST_DEVICE void* __GetExceptionInfo(_E);

template <class _Ep>
_LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
exception_ptr make_exception_ptr(_Ep __e) noexcept
{
  return __copy_exception_ptr(_CUDA_VSTD::addressof(__e), __GetExceptionInfo(__e));
}

#endif // _LIBCUDACXX_ABI_MICROSOFT

_LIBCUDACXX_END_NAMESPACE_STD_NOVERSION

#endif // _LIBCUDACXX___EXCEPTION_EXCEPTION_PTR_H
