//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDA_STD___ATOMIC_API_REFERENCE_H
#define __CUDA_STD___ATOMIC_API_REFERENCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__atomic/api/common.h>
#include <cuda/std/__atomic/order.h>
#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/__atomic/types.h>
#include <cuda/std/__atomic/wait/notify_wait.h>
#include <cuda/std/__atomic/wait/polling.h>
#include <cuda/std/__type_traits/conditional.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <typename _Tp, typename _Sco>
struct __atomic_ref_common
{
  _CCCL_API constexpr __atomic_ref_common(_Tp& __v)
      : __a(&__v)
  {}

  __atomic_ref_storage<_Tp> __a;

#if defined(_CCCL_ATOMIC_ALWAYS_LOCK_FREE)
  static constexpr bool is_always_lock_free = _CCCL_ATOMIC_ALWAYS_LOCK_FREE(sizeof(_Tp), 0);
#endif // defined(_CCCL_ATOMIC_ALWAYS_LOCK_FREE)

  _LIBCUDACXX_ATOMIC_COMMON_IMPL(const, )
};

template <typename _Tp, typename _Sco>
struct __atomic_ref_arithmetic
{
  _CCCL_API constexpr __atomic_ref_arithmetic(_Tp& __v)
      : __a(&__v)
  {}

  __atomic_ref_storage<_Tp> __a;

#if defined(_CCCL_ATOMIC_ALWAYS_LOCK_FREE)
  static constexpr bool is_always_lock_free = _CCCL_ATOMIC_ALWAYS_LOCK_FREE(sizeof(_Tp), 0);
#endif // defined(_CCCL_ATOMIC_ALWAYS_LOCK_FREE)

  _LIBCUDACXX_ATOMIC_COMMON_IMPL(const, )
  _LIBCUDACXX_ATOMIC_ARITHMETIC_IMPL(const, )
};

template <typename _Tp, typename _Sco>
struct __atomic_ref_bitwise
{
  _CCCL_API constexpr __atomic_ref_bitwise(_Tp& __v)
      : __a(&__v)
  {}

  __atomic_ref_storage<_Tp> __a;

#if defined(_CCCL_ATOMIC_ALWAYS_LOCK_FREE)
  static constexpr bool is_always_lock_free = _CCCL_ATOMIC_ALWAYS_LOCK_FREE(sizeof(_Tp), 0);
#endif // defined(_CCCL_ATOMIC_ALWAYS_LOCK_FREE)

  _LIBCUDACXX_ATOMIC_COMMON_IMPL(const, )
  _LIBCUDACXX_ATOMIC_ARITHMETIC_IMPL(const, )
  _LIBCUDACXX_ATOMIC_BITWISE_IMPL(const, )
};

template <typename _Tp, typename _Sco>
struct __atomic_ref_pointer
{
  _CCCL_API constexpr __atomic_ref_pointer(_Tp& __v)
      : __a(&__v)
  {}

  __atomic_ref_storage<_Tp> __a;

#if defined(_CCCL_ATOMIC_ALWAYS_LOCK_FREE)
  static constexpr bool is_always_lock_free = _CCCL_ATOMIC_ALWAYS_LOCK_FREE(sizeof(_Tp), 0);
#endif // defined(_CCCL_ATOMIC_ALWAYS_LOCK_FREE)

  _LIBCUDACXX_ATOMIC_COMMON_IMPL(const, )
  _LIBCUDACXX_ATOMIC_POINTER_IMPL(const, )
};

template <typename _Tp, thread_scope _Sco = thread_scope_system>
using __atomic_ref_impl =
  _If<is_pointer_v<_Tp>,
      __atomic_ref_pointer<_Tp, __scope_to_tag<_Sco>>,
      _If<is_floating_point_v<_Tp>,
          __atomic_ref_arithmetic<_Tp, __scope_to_tag<_Sco>>,
          _If<is_integral_v<_Tp>,
              __atomic_ref_bitwise<_Tp, __scope_to_tag<_Sco>>,
              __atomic_ref_common<_Tp, __scope_to_tag<_Sco>>>>>;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDA_STD___ATOMIC_API_REFERENCE_H
