//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __LIBCUDACXX___ATOMIC_FUNCTIONS_DERIVED_H
#define __LIBCUDACXX___ATOMIC_FUNCTIONS_DERIVED_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__atomic/functions/cuda_ptx_generated.h>
#include <cuda/std/cstdint>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_CUDA_COMPILER)

template <typename _Tp, typename _Sco, __enable_if_t<sizeof(_Tp) <= 2, int> = 0>
bool _CCCL_DEVICE __atomic_compare_exchange_cuda(
  _Tp volatile* __ptr, _Tp* __expected, const _Tp __desired, bool, int __success_memorder, int __failure_memorder, _Sco)
{
  auto const __aligned = (uint32_t*) ((intptr_t) __ptr & ~(sizeof(uint32_t) - 1));
  auto const __offset  = uint32_t((intptr_t) __ptr & (sizeof(uint32_t) - 1)) * 8;
  auto const __mask    = ((1 << sizeof(_Tp) * 8) - 1) << __offset;

  uint32_t __old = *__expected << __offset;
  uint32_t __old_value;
  while (1)
  {
    __old_value = (__old & __mask) >> __offset;
    if (__old_value != *__expected)
    {
      break;
    }
    uint32_t const __attempt = (__old & ~__mask) | (*__desired << __offset);
    if (__atomic_compare_exchange_cuda(
          __aligned, &__old, &__attempt, true, __success_memorder, __failure_memorder, _Sco{}))
    {
      return true;
    }
  }
  *__expected = __old_value;
  return false;
}

template <typename _Tp, typename _Sco, __enable_if_t<sizeof(_Tp) <= 2, int> = 0>
void _CCCL_DEVICE __atomic_exchange_cuda(_Tp volatile* __ptr, _Tp* __val, _Tp* __ret, int __memorder, _Sco)
{
  _Tp __expected = __atomic_load_n_cuda(__ptr, __ATOMIC_RELAXED, _Sco{});
  while (!__atomic_compare_exchange_cuda(__ptr, &__expected, __val, true, __memorder, __memorder, _Sco{}))
    ;
  *__ret = __expected;
}

template <typename _Tp, typename _Up, typename _Sco, __enable_if_t<sizeof(_Tp) <= 2, int> = 0>
_Tp _CCCL_DEVICE __atomic_fetch_add_cuda(_Tp volatile* __ptr, _Up __val, int __memorder, _Sco)
{
  _Tp __expected = __atomic_load_n_cuda(__ptr, __ATOMIC_RELAXED, _Sco{});
  _Tp __desired  = __expected + __val;
  while (!__atomic_compare_exchange_cuda(__ptr, &__expected, __desired, true, __memorder, __memorder, _Sco{}))
  {
    __desired = __expected + __val;
  }
  return __expected;
}

template <typename _Tp,
          typename _Up,
          typename _Sco,
          __enable_if_t<sizeof(_Tp) <= 2 || _CUDA_VSTD::is_floating_point<_Tp>::value, int> = 0>
_Tp _CCCL_HOST_DEVICE __atomic_fetch_max_cuda(_Tp volatile* __ptr, _Up __val, int __memorder, _Sco)
{
  _Tp __expected = __atomic_load_n_cuda(__ptr, __ATOMIC_RELAXED, _Sco{});
  _Tp __desired  = __expected > __val ? __expected : __val;

  while (__desired == __val
         && !__atomic_compare_exchange_cuda(__ptr, &__expected, __desired, true, __memorder, __memorder, _Sco{}))
  {
    __desired = __expected > __val ? __expected : __val;
  }

  return __expected;
}

template <typename _Tp,
          typename _Up,
          typename _Sco,
          __enable_if_t<sizeof(_Tp) <= 2 || _CUDA_VSTD::is_floating_point<_Tp>::value, int> = 0>
_Tp _CCCL_HOST_DEVICE __atomic_fetch_min_cuda(_Tp volatile* __ptr, _Up __val, int __memorder, _Sco)
{
  _Tp __expected = __atomic_load_n_cuda(__ptr, __ATOMIC_RELAXED, _Sco{});
  _Tp __desired  = __expected < __val ? __expected : __val;

  while (__desired == __val
         && !__atomic_compare_exchange_cuda(__ptr, &__expected, __desired, true, __memorder, __memorder, _Sco{}))
  {
    __desired = __expected < __val ? __expected : __val;
  }

  return __expected;
}

template <typename _Tp, typename _Up, typename _Sco, __enable_if_t<sizeof(_Tp) <= 2, int> = 0>
_Tp _CCCL_DEVICE __atomic_fetch_sub_cuda(_Tp volatile* __ptr, _Up __val, int __memorder, _Sco)
{
  _Tp __expected = __atomic_load_n_cuda(__ptr, __ATOMIC_RELAXED, _Sco{});
  _Tp __desired  = __expected - __val;
  while (!__atomic_compare_exchange_cuda(__ptr, &__expected, __desired, true, __memorder, __memorder, _Sco{}))
  {
    __desired = __expected - __val;
  }
  return __expected;
}

template <typename _Tp, typename _Up, typename _Sco, __enable_if_t<sizeof(_Tp) <= 2, int> = 0>
_Tp _CCCL_DEVICE __atomic_fetch_and_cuda(_Tp volatile* __ptr, _Up __val, int __memorder, _Sco)
{
  _Tp __expected = __atomic_load_n_cuda(__ptr, __ATOMIC_RELAXED, _Sco{});
  _Tp __desired  = __expected & __val;
  while (!__atomic_compare_exchange_cuda(__ptr, &__expected, __desired, true, __memorder, __memorder, _Sco{}))
  {
    __desired = __expected & __val;
  }
  return __expected;
}

template <typename _Tp, typename _Up, typename _Sco, __enable_if_t<sizeof(_Tp) <= 2, int> = 0>
_Tp _CCCL_DEVICE __atomic_fetch_xor_cuda(_Tp volatile* __ptr, _Up __val, int __memorder, _Sco)
{
  _Tp __expected = __atomic_load_n_cuda(__ptr, __ATOMIC_RELAXED, _Sco{});
  _Tp __desired  = __expected ^ __val;
  while (!__atomic_compare_exchange_cuda(__ptr, &__expected, __desired, true, __memorder, __memorder, _Sco{}))
  {
    __desired = __expected ^ __val;
  }
  return __expected;
}

template <typename _Tp, typename _Up, typename _Sco, __enable_if_t<sizeof(_Tp) <= 2, int> = 0>
_Tp _CCCL_DEVICE __atomic_fetch_or_cuda(_Tp volatile* __ptr, _Up __val, int __memorder, _Sco)
{
  _Tp __expected = __atomic_load_n_cuda(__ptr, __ATOMIC_RELAXED, _Sco{});
  _Tp __desired  = __expected | __val;
  while (!__atomic_compare_exchange_cuda(__ptr, &__expected, __desired, true, __memorder, __memorder, _Sco{}))
  {
    __desired = __expected | __val;
  }
  return __expected;
}

template <typename _Tp, typename _Sco>
_Tp _CCCL_DEVICE __atomic_load_n_cuda(const _Tp volatile* __ptr, int __memorder, _Sco)
{
  _Tp __ret;
  __atomic_load_cuda(__ptr, &__ret, __memorder, _Sco{});
  return __ret;
}

template <typename _Tp, typename _Sco>
void _CCCL_DEVICE __atomic_store_n_cuda(_Tp volatile* __ptr, _Tp __val, int __memorder, _Sco)
{
  __atomic_store_cuda(__ptr, &__val, __memorder, _Sco{});
}

template <typename _Tp, typename _Sco>
bool _CCCL_DEVICE __atomic_compare_exchange_n_cuda(
  _Tp volatile* __ptr, _Tp* __expected, _Tp __desired, bool __weak, int __success_memorder, int __failure_memorder, _Sco)
{
  return __atomic_compare_exchange_cuda(
    __ptr, __expected, __desired, __weak, __success_memorder, __failure_memorder, _Sco{});
}

template <typename _Tp, typename _Sco>
_Tp _CCCL_DEVICE __atomic_exchange_n_cuda(_Tp volatile* __ptr, _Tp __val, int __memorder, _Sco)
{
  _Tp __ret;
  __atomic_exchange_cuda(__ptr, &__val, &__ret, __memorder, _Sco{});
  return __ret;
}

static inline _CCCL_DEVICE void __atomic_signal_fence_cuda(int)
{
  asm volatile("" ::: "memory");
}

#endif // defined(_CCCL_CUDA_COMPILER)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // __LIBCUDACXX___ATOMIC_FUNCTIONS_DERIVED_H
