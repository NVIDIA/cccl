//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___EXPECTED_BAD_EXPECTED_ACCESS_H
#define _LIBCUDACXX___EXPECTED_BAD_EXPECTED_ACCESS_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include "../__utility/move.h"
#include "../exception"

#if _LIBCUDACXX_STD_VER  > 11

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Err>
class bad_expected_access;

template <>
class bad_expected_access<void> : public exception {
protected:
  bad_expected_access() noexcept                             = default;
  bad_expected_access(const bad_expected_access&)            = default;
  bad_expected_access(bad_expected_access&&)                 = default;
  bad_expected_access& operator=(const bad_expected_access&) = default;
  bad_expected_access& operator=(bad_expected_access&&)      = default;
  ~bad_expected_access() noexcept override                   = default;

public:
  // The way this has been designed (by using a class template below) means that we'll already
  // have a profusion of these vtables in TUs, and the dynamic linker will already have a bunch
  // of work to do. So it is not worth hiding the <void> specialization in the dylib, given that
  // it adds deployment target restrictions.
  _LIBCUDACXX_INLINE_VISIBILITY
  const char* what() const noexcept override { return "bad access to std::expected"; }
};

template <class _Err>
class bad_expected_access : public bad_expected_access<void> {
public:
  _LIBCUDACXX_INLINE_VISIBILITY
  explicit bad_expected_access(_Err __e) : __unex_(_CUDA_VSTD::move(__e)) {}

  _LIBCUDACXX_INLINE_VISIBILITY
  _Err& error() & noexcept { return __unex_; }

  _LIBCUDACXX_INLINE_VISIBILITY
  const _Err& error() const& noexcept { return __unex_; }

  _LIBCUDACXX_INLINE_VISIBILITY
  _Err&& error() && noexcept { return _CUDA_VSTD::move(__unex_); }

  _LIBCUDACXX_INLINE_VISIBILITY
  const _Err&& error() const&& noexcept { return _CUDA_VSTD::move(__unex_); }

private:
  _Err __unex_;
};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX_STD_VER > 11

#endif // _LIBCUDACXX___EXPECTED_BAD_EXPECTED_ACCESS_H
