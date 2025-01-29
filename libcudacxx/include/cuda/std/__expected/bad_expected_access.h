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

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__exception/terminate.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>

#include <nv/target>

#if _CCCL_STD_VER > 2011

#  ifndef _CCCL_NO_EXCEPTIONS
#    ifdef __cpp_lib_expected
#      include <expected>
#    else // ^^^ __cpp_lib_expected ^^^ / vvv !__cpp_lib_expected vvv
#      include <exception>
#    endif // !__cpp_lib_expected
#  endif // _CCCL_NO_EXCEPTIONS

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#  ifndef _CCCL_NO_EXCEPTIONS

#    ifdef __cpp_lib_expected

using ::std::bad_expected_access;

#    else // ^^^ __cpp_lib_expected ^^^ / vvv !__cpp_lib_expected vvv

template <class _Err>
class bad_expected_access;

template <>
class bad_expected_access<void> : public ::std::exception
{
public:
  // The way this has been designed (by using a class template below) means that we'll already
  // have a profusion of these vtables in TUs, and the dynamic linker will already have a bunch
  // of work to do. So it is not worth hiding the <void> specialization in the dylib, given that
  // it adds deployment target restrictions.
  const char* what() const noexcept override
  {
    return "bad access to cuda::std::expected";
  }
};

template <class _Err>
class bad_expected_access : public bad_expected_access<void>
{
public:
#      if _CCCL_CUDA_COMPILER(CLANG) // Clang needs this or it breaks with device only types
  _CCCL_HOST_DEVICE
#      endif // _CCCL_CUDA_COMPILER(CLANG)
  _CCCL_HIDE_FROM_ABI explicit bad_expected_access(_Err __e)
      : __unex_(_CUDA_VSTD::move(__e))
  {}

#      if _CCCL_CUDA_COMPILER(CLANG) // Clang needs this or it breaks with device only types
  _CCCL_HOST_DEVICE
#      endif // _CCCL_CUDA_COMPILER(CLANG)
  _CCCL_HIDE_FROM_ABI ~bad_expected_access() noexcept
  {
    __unex_.~_Err();
  }

  _LIBCUDACXX_HIDE_FROM_ABI _Err& error() & noexcept
  {
    return __unex_;
  }

  _LIBCUDACXX_HIDE_FROM_ABI const _Err& error() const& noexcept
  {
    return __unex_;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _Err&& error() && noexcept
  {
    return _CUDA_VSTD::move(__unex_);
  }

  _LIBCUDACXX_HIDE_FROM_ABI const _Err&& error() const&& noexcept
  {
    return _CUDA_VSTD::move(__unex_);
  }

private:
  _Err __unex_;
};
#    endif // !__cpp_lib_expected

#  endif // !_CCCL_NO_EXCEPTIONS

template <class _Err, class _Arg>
_CCCL_NORETURN _LIBCUDACXX_HIDE_FROM_ABI void __throw_bad_expected_access(_Arg&& __arg)
{
#  ifndef _CCCL_NO_EXCEPTIONS
  NV_IF_ELSE_TARGET(NV_IS_HOST,
                    (throw _CUDA_VSTD::bad_expected_access<_Err>(_CUDA_VSTD::forward<_Arg>(__arg));),
                    ((void) __arg; _CUDA_VSTD_NOVERSION::terminate();))
#  else // ^^^ !_CCCL_NO_EXCEPTIONS ^^^ / vvv _CCCL_NO_EXCEPTIONS vvv
  (void) __arg;
  _CUDA_VSTD_NOVERSION::terminate();
#  endif // _CCCL_NO_EXCEPTIONS
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _CCCL_STD_VER > 2011

#endif // _LIBCUDACXX___EXPECTED_BAD_EXPECTED_ACCESS_H
