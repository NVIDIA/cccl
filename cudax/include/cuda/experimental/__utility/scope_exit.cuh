//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__EXPERIMENTAL_UTILITY_SCOPE_EXIT
#define _CUDAX__EXPERIMENTAL_UTILITY_SCOPE_EXIT

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__utility/forward.h>

namespace cuda::experimental
{
// See: https://en.cppreference.com/w/cpp/experimental/scope_exit
template <class _Fn>
struct scope_exit
{
  static_assert(::cuda::std::__is_nothrow_callable_v<_Fn&>,
                "The scope_guard function must be nothrow lvalue-callable with no arguments.");

  template <class _Fn2>
  _CCCL_API explicit scope_exit(_Fn2&& __fn) noexcept(::cuda::std::is_nothrow_constructible_v<_Fn, _Fn2>)
      : scope_exit(::cuda::std::forward<_Fn2>(__fn), ::cuda::std::is_nothrow_constructible<_Fn, _Fn2>{})
  {
    static_assert(::cuda::std::is_nothrow_constructible_v<_Fn, _Fn2> || ::cuda::std::is_constructible_v<_Fn, _Fn2&>,
                  "The scope_guard function must be nothrow constructible from the provided callable or "
                  "constructible from an lvalue reference to the provided callable.");
  }

  scope_exit(scope_exit&&)            = default;
  scope_exit& operator=(scope_exit&&) = delete;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API ~scope_exit()
  {
    if (__active_)
    {
      __fn_();
    }
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API void release() noexcept
  {
    __active_ = false;
  }

private:
  // Handle the case where _Fn is nothrow constructible from _Fn2.
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Fn2>
  _CCCL_API explicit scope_exit(_Fn2&& __fn, ::cuda::std::true_type) noexcept
      : __fn_(::cuda::std::forward<_Fn2>(__fn))
  {}

  // Handle the case where _Fn is not nothrow constructible from _Fn2, but is
  // constructible from _Fn2&. In this case we need to make a copy of __fn first to ensure
  // that if the copy throws we don't end up with a partially constructed scope_exit
  // object. We do this by creating a temporary scope_exit object that holds a reference
  // to the original callable, and then releasing it if the copy succeeds.
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Fn2>
  _CCCL_API explicit scope_exit(_Fn2&& __fn, ::cuda::std::false_type) noexcept(false)
      : scope_exit(__fn, scope_exit<_Fn2&>(__fn))
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Fn2>
  _CCCL_API explicit scope_exit(_Fn2& __fn, scope_exit<_Fn2&>&& __scope) noexcept(false)
      : __fn_(__fn) // copy not move because we don't want to invalidate __scope if the copy throws
  {
    __scope.release(); // the copy succeeded, so release __scope
  }

  _Fn __fn_;
  bool __active_{true};
};

template <class _Fn>
_CCCL_HOST_DEVICE scope_exit(_Fn) -> scope_exit<_Fn>;
} // namespace cuda::experimental

#endif // _CUDAX__EXPERIMENTAL_UTILITY_SCOPE_EXIT
