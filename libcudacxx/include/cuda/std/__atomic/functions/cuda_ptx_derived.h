//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __LIBCUDACXX___ATOMIC_FUNCTIONS_DERIVED_H
#define __LIBCUDACXX___ATOMIC_FUNCTIONS_DERIVED_H

#include <cuda/std/detail/__config>

#include <cstdint>

#include "cuda/std/__atomic/functions/cuda_ptx_generated_helper.h"
#include "cuda_ptx_generated.h"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__atomic/functions/cuda_ptx_generated.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_scalar.h>
#include <cuda/std/cstdint>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_CUDA_COMPILER)

template <class _Operand>
using __atomic_cuda_enable_non_native_size = typename enable_if<_Operand::__size <= 16, bool>::type;

template <class _Operand>
using __atomic_cuda_enable_native_size = typename enable_if<_Operand::__size >= 32, bool>::type;

template <class _Type, class _Order, class _Operand, class _Sco, __atomic_cuda_enable_non_native_size<_Operand> = 0>
static inline _CCCL_DEVICE void
__cuda_atomic_load(const _Type* __ptr, _Type& __dst, _Order, _Operand, _Sco, __atomic_cuda_mmio_disable)
{
  uint32_t* __aligned     = (uint32_t*) ((intptr_t) __ptr & ~(sizeof(uint32_t) - 1));
  const uint32_t __offset = uint32_t((intptr_t) __ptr & (sizeof(uint32_t) - 1)) * 8;

  uint32_t __value = 0;

  __cuda_atomic_load(__aligned, __value, _Order{}, __atomic_cuda_operand_b32{}, _Sco{}, __atomic_cuda_mmio_disable{});
  __dst = static_cast<_Type>(__value >> __offset);
}

template <class _Type, class _Order, class _Operand, class _Sco, __atomic_cuda_enable_non_native_size<_Operand> = 0>
static inline _CCCL_DEVICE bool
__cuda_atomic_compare_exchange(_Type* __ptr, _Type& __dst, _Type __cmp, _Type __op, _Order, _Operand, _Sco)
{
  uint32_t* __aligned     = (uint32_t*) ((intptr_t) __ptr & ~(sizeof(uint32_t) - 1));
  const uint32_t __offset = uint32_t((intptr_t) __ptr & (sizeof(uint32_t) - 1)) * 8;
  const uint32_t __mask   = ((1 << (sizeof(_Type) * 8)) - 1) << __offset;

  // Algorithm for 8b CAS with 32b intrinsics
  // __old = __window[0:32] where [__cmp] resides within any of the potential offsets
  // First CAS attempt 'guesses' that the masked portion of the window is 0x00.
  uint32_t __old       = (uint32_t(__op) << __offset);
  uint32_t __old_value = 0;

  // Reemit CAS instructions until either of two conditions are met
  while (1)
  {
    // Combine the desired value and most recently fetched expected masked portion of the window
    uint32_t __attempt = (__old & ~__mask) | (uint32_t(__op) << __offset);

    if (__cuda_atomic_compare_exchange(
          __aligned, __old, __old, __attempt, _Order{}, __atomic_cuda_operand_b32{}, _Sco{}))
    {
      // CAS was successful
      return true;
    }
    __old_value = (__old & __mask) >> __offset;
    // The expected value no longer matches inside the CAS.
    if (__old_value != __cmp)
    {
      __dst = __old_value;
      break;
    }
  }
  return false;
}

// Lower level fetch_update that bypasses memorder dispatch
template <class _Type, class _Fn, class _Sco, class _Order, class _Operand>
_CCCL_DEVICE _Type __cuda_atomic_fetch_update(_Type* __ptr, const _Fn& __op, _Order, _Operand, _Sco)
{
  _Type __expected = 0;
  __atomic_load_cuda(__ptr, __expected, __ATOMIC_RELAXED, _Sco{});
  _Type __desired = __op(__expected);
  while (!__cuda_atomic_compare_exchange(__ptr, __expected, __expected, __desired, _Order{}, _Operand{}, _Sco{}))
  {
    __desired = __op(__expected);
  }
  return __expected;
}

template <class _Type, class _Order, class _Operand, class _Sco, __atomic_cuda_enable_non_native_size<_Operand> = 0>
static inline _CCCL_DEVICE void __cuda_atomic_fetch_add(_Type* __ptr, _Type& __dst, _Type __op, _Order, _Operand, _Sco)
{
  __dst = __cuda_atomic_fetch_update(
    __ptr,
    [__op](_Type __old) {
      return __old + __op;
    },
    _Order{},
    __atomic_cuda_operand_tag<__atomic_cuda_operand::_b, _Operand::__size>{},
    _Sco{});
}

template <class _Type, class _Order, class _Operand, class _Sco, __atomic_cuda_enable_non_native_size<_Operand> = 0>
static inline _CCCL_DEVICE void __cuda_atomic_fetch_and(_Type* __ptr, _Type& __dst, _Type __op, _Order, _Operand, _Sco)
{
  __dst = __cuda_atomic_fetch_update(
    __ptr,
    [__op](_Type __old) {
      return __old & __op;
    },
    _Order{},
    __atomic_cuda_operand_tag<__atomic_cuda_operand::_b, _Operand::__size>{},
    _Sco{});
}

template <class _Type, class _Order, class _Operand, class _Sco, __atomic_cuda_enable_non_native_size<_Operand> = 0>
static inline _CCCL_DEVICE void __cuda_atomic_fetch_xor(_Type* __ptr, _Type& __dst, _Type __op, _Order, _Operand, _Sco)
{
  __dst = __cuda_atomic_fetch_update(
    __ptr,
    [__op](_Type __old) {
      return __old ^ __op;
    },
    _Order{},
    __atomic_cuda_operand_tag<__atomic_cuda_operand::_b, _Operand::__size>{},
    _Sco{});
}

template <class _Type, class _Order, class _Operand, class _Sco, __atomic_cuda_enable_non_native_size<_Operand> = 0>
static inline _CCCL_DEVICE void __cuda_atomic_fetch_or(_Type* __ptr, _Type& __dst, _Type __op, _Order, _Operand, _Sco)
{
  __dst = __cuda_atomic_fetch_update(
    __ptr,
    [__op](_Type __old) {
      return __old | __op;
    },
    _Order{},
    __atomic_cuda_operand_tag<__atomic_cuda_operand::_b, _Operand::__size>{},
    _Sco{});
}

template <class _Type, class _Order, class _Operand, class _Sco, __atomic_cuda_enable_non_native_size<_Operand> = 0>
static inline _CCCL_DEVICE void __cuda_atomic_fetch_min(_Type* __ptr, _Type& __dst, _Type __op, _Order, _Operand, _Sco)
{
  __dst = __cuda_atomic_fetch_update(
    __ptr,
    [__op](_Type __old) {
      return __old < __op ? __old : __op;
    },
    _Order{},
    __atomic_cuda_operand_tag<__atomic_cuda_operand::_b, _Operand::__size>{},
    _Sco{});
}
template <class _Type, class _Order, class _Operand, class _Sco, __atomic_cuda_enable_non_native_size<_Operand> = 0>
static inline _CCCL_DEVICE void __cuda_atomic_fetch_max(_Type* __ptr, _Type& __dst, _Type __op, _Order, _Operand, _Sco)
{
  __dst = __cuda_atomic_fetch_update(
    __ptr,
    [__op](_Type __old) {
      return __old < __op ? __old : __op;
    },
    _Order{},
    __atomic_cuda_operand_tag<__atomic_cuda_operand::_b, _Operand::__size>{},
    _Sco{});
}

template <class _Type, class _Order, class _Operand, class _Sco, __atomic_cuda_enable_non_native_size<_Operand> = 0>
static inline _CCCL_DEVICE void __cuda_atomic_exchange(_Type* __ptr, _Type& __dst, _Type __op, _Order, _Operand, _Sco)
{
  __dst = __cuda_atomic_fetch_update(
    __ptr,
    [__op](_Type __old) {
      return __op;
    },
    _Order{},
    __atomic_cuda_operand_tag<__atomic_cuda_operand::_b, _Operand::__size>{},
    _Sco{});
}

template <class _Type, class _Order, class _Operand, class _Sco, __atomic_cuda_enable_non_native_size<_Operand> = 0>
static inline _CCCL_DEVICE void
__cuda_atomic_store(_Type* __ptr, _Type __val, _Order, _Operand, _Sco, __atomic_cuda_mmio_disable)
{
  // Store requires cas on 8b types
  __cuda_atomic_fetch_update(
    __ptr,
    [__val](_Type __old) {
      return __val;
    },
    _Order{},
    __atomic_cuda_operand_tag<__atomic_cuda_operand::_b, _Operand::__size>{},
    _Sco{});
}

template <typename _Tp, typename _Fn, typename _Sco>
_CCCL_DEVICE _Tp __atomic_fetch_update_cuda(_Tp* __ptr, const _Fn& __op, int __memorder, _Sco)
{
  _Tp __expected = __atomic_load_n_cuda(__ptr, __ATOMIC_RELAXED, _Sco{});
  _Tp __desired  = __op(__expected);
  while (!__atomic_compare_exchange_cuda(__ptr, &__expected, __desired, true, __memorder, __memorder, _Sco{}))
  {
    __desired = __op(__expected);
  }
  return __expected;
}
template <typename _Tp, typename _Fn, typename _Sco>
_CCCL_DEVICE _Tp __atomic_fetch_update_cuda(_Tp volatile* __ptr, const _Fn& __op, int __memorder, _Sco)
{
  _Tp __expected = __atomic_load_n_cuda(__ptr, __ATOMIC_RELAXED, _Sco{});
  _Tp __desired  = __op(__expected);
  while (!__atomic_compare_exchange_cuda(__ptr, &__expected, __desired, true, __memorder, __memorder, _Sco{}))
  {
    __desired = __op(__expected);
  }
  return __expected;
}

template <typename _Tp, typename _Sco>
_CCCL_DEVICE _Tp __atomic_load_n_cuda(const _Tp* __ptr, int __memorder, _Sco)
{
  _Tp __ret;
  __atomic_load_cuda(__ptr, __ret, __memorder, _Sco{});
  return __ret;
}
template <typename _Tp, typename _Sco>
_CCCL_DEVICE _Tp __atomic_load_n_cuda(const _Tp volatile* __ptr, int __memorder, _Sco)
{
  _Tp __ret;
  __atomic_load_cuda(__ptr, __ret, __memorder, _Sco{});
  return __ret;
}

template <typename _Tp, typename _Sco>
_CCCL_DEVICE void __atomic_store_n_cuda(_Tp* __ptr, _Tp __val, int __memorder, _Sco)
{
  __atomic_store_cuda(__ptr, __val, __memorder, _Sco{});
}
template <typename _Tp, typename _Sco>
_CCCL_DEVICE void __atomic_store_n_cuda(_Tp volatile* __ptr, _Tp __val, int __memorder, _Sco)
{
  __atomic_store_cuda(__ptr, __val, __memorder, _Sco{});
}

template <typename _Tp, typename _Sco>
_CCCL_DEVICE _Tp __atomic_exchange_n_cuda(_Tp* __ptr, _Tp __val, int __memorder, _Sco)
{
  _Tp __ret;
  __atomic_exchange_cuda(__ptr, __ret, __val, __memorder, _Sco{});
  return __ret;
}
template <typename _Tp, typename _Sco>
_CCCL_DEVICE _Tp __atomic_exchange_n_cuda(_Tp volatile* __ptr, _Tp __val, int __memorder, _Sco)
{
  _Tp __ret;
  __atomic_exchange_cuda(__ptr, __ret, __val, __memorder, _Sco{});
  return __ret;
}

template <typename _Tp, typename _Up, typename _Sco, __atomic_enable_if_not_native_minmax<_Tp> = 0>
_CCCL_DEVICE _Tp __atomic_fetch_min_cuda(_Tp* __ptr, _Up __val, int __memorder, _Sco)
{
  return __atomic_fetch_update_cuda(
    __ptr,
    [__val](_Tp __old) {
      return __old < __val ? __old : __val;
    },
    __memorder,
    _Sco{});
}
template <typename _Tp, typename _Up, typename _Sco, __atomic_enable_if_not_native_minmax<_Tp> = 0>
_CCCL_DEVICE _Tp __atomic_fetch_min_cuda(volatile _Tp* __ptr, _Up __val, int __memorder, _Sco)
{
  return __atomic_fetch_update_cuda(
    __ptr,
    [__val](_Tp __old) {
      return __old < __val ? __old : __val;
    },
    __memorder,
    _Sco{});
}

template <typename _Tp, typename _Up, typename _Sco, __atomic_enable_if_not_native_minmax<_Tp> = 0>
_CCCL_DEVICE _Tp __atomic_fetch_max_cuda(_Tp* __ptr, _Up __val, int __memorder, _Sco)
{
  return __atomic_fetch_update_cuda(
    __ptr,
    [__val](_Tp __old) {
      return __old > __val ? __old : __val;
    },
    __memorder,
    _Sco{});
}
template <typename _Tp, typename _Up, typename _Sco, __atomic_enable_if_not_native_minmax<_Tp> = 0>
_CCCL_DEVICE _Tp __atomic_fetch_max_cuda(volatile _Tp* __ptr, _Up __val, int __memorder, _Sco)
{
  return __atomic_fetch_update_cuda(
    __ptr,
    [__val](_Tp __old) {
      return __old > __val ? __old : __val;
    },
    __memorder,
    _Sco{});
}

_CCCL_DEVICE static inline void __atomic_signal_fence_cuda(int)
{
  asm volatile("" ::: "memory");
}

#endif // defined(_CCCL_CUDA_COMPILER)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // __LIBCUDACXX___ATOMIC_FUNCTIONS_DERIVED_H
