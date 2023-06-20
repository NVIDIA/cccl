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

#ifndef _LIBCUDACXX___EXCEPTION_EXCEPTION_H
#define _LIBCUDACXX___EXCEPTION_EXCEPTION_H

#ifndef __cuda_std__
#include <__config>
#endif //__cuda_std__

// <vcruntime_exception.h> defines its own std::exception and std::bad_exception types,
// which we use in order to be ABI-compatible with other STLs on Windows.
#if defined(_LIBCUDACXX_ABI_VCRUNTIME)
#  include <vcruntime_exception.h>
#endif

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD_NOVERSION // purposefully not using versioning namespace

#if defined(_LIBCUDACXX_ABI_VCRUNTIME) && (!defined(_HAS_EXCEPTIONS) || _HAS_EXCEPTIONS != 0)
// The std::exception class was already included above, but we're explicit about this condition here for clarity.

#elif defined(_LIBCUDACXX_ABI_VCRUNTIME) && _HAS_EXCEPTIONS == 0
// However, <vcruntime_exception.h> does not define std::exception and std::bad_exception
// when _HAS_EXCEPTIONS == 0.
//
// Since libc++ still wants to provide the std::exception hierarchy even when _HAS_EXCEPTIONS == 0
// (after all those are simply types like any other), we define an ABI-compatible version
// of the VCRuntime std::exception and std::bad_exception types in that mode.

struct __std_exception_data {
  char const* _What;
  bool _DoFree;
};

class exception { // base of all library exceptions
public:
  _LIBCUDACXX_INLINE_VISIBILITY exception() noexcept : __data_() {}

  _LIBCUDACXX_INLINE_VISIBILITY explicit exception(char const* __message) noexcept : __data_() {
    __data_._What   = __message;
    __data_._DoFree = true;
  }

  _LIBCUDACXX_INLINE_VISIBILITY exception(exception const&) noexcept {}

  _LIBCUDACXX_INLINE_VISIBILITY exception& operator=(exception const&) noexcept { return *this; }

  _LIBCUDACXX_INLINE_VISIBILITY virtual ~exception() noexcept {}

  _LIBCUDACXX_INLINE_VISIBILITY virtual char const* what() const noexcept { return __data_._What ? __data_._What : "Unknown exception"; }

private:
  __std_exception_data __data_;
};

class bad_exception : public exception {
public:
  _LIBCUDACXX_INLINE_VISIBILITY bad_exception() noexcept : exception("bad exception") {}
};

#else  // !defined(_LIBCUDACXX_ABI_VCRUNTIME)
// On all other platforms, we define our own std::exception and std::bad_exception types
// regardless of whether exceptions are turned on as a language feature.

class _LIBCUDACXX_EXCEPTION_ABI exception {
public:
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY exception() noexcept {}
  _LIBCUDACXX_HIDE_FROM_ABI exception(const exception&) noexcept = default;
  _LIBCUDACXX_HOST_DEVICE virtual ~exception() noexcept;
  _LIBCUDACXX_HOST_DEVICE virtual const char* what() const noexcept;
};

class _LIBCUDACXX_EXCEPTION_ABI bad_exception : public exception {
public:
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY bad_exception() noexcept {}
  _LIBCUDACXX_HOST_DEVICE virtual ~bad_exception() noexcept;
  _LIBCUDACXX_HOST_DEVICE virtual const char* what() const noexcept;
};
#endif // !_LIBCUDACXX_ABI_VCRUNTIME

_LIBCUDACXX_END_NAMESPACE_STD_NOVERSION

#endif // _LIBCUDACXX___EXCEPTION_EXCEPTION_H
