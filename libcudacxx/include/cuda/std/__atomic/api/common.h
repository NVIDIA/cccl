//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDA_STD___ATOMIC_API_COMMON_H
#define __CUDA_STD___ATOMIC_API_COMMON_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__atomic/types/base.h>
#include <cuda/std/__type_traits/remove_cv.h>

// API definitions for the base atomic implementation
#define _LIBCUDACXX_ATOMIC_COMMON_IMPL(_CONST, _VOLATILE, _VT)                                                        \
  _CCCL_HOST_DEVICE_API inline bool is_lock_free() const _VOLATILE noexcept                                           \
  {                                                                                                                   \
    return _LIBCUDACXX_ATOMIC_IS_LOCK_FREE(sizeof(_Tp));                                                              \
  }                                                                                                                   \
  _CCCL_HOST_DEVICE_API inline void store(_VT __d, memory_order __m = memory_order_seq_cst)                           \
    _CONST _VOLATILE noexcept _LIBCUDACXX_CHECK_STORE_MEMORY_ORDER(__m)                                               \
  {                                                                                                                   \
    __atomic_store_dispatch(&__a, __d, __m, _Sco{});                                                                  \
  }                                                                                                                   \
  _CCCL_HOST_DEVICE_API inline _VT load(memory_order __m = memory_order_seq_cst)                                      \
    const _VOLATILE noexcept _LIBCUDACXX_CHECK_LOAD_MEMORY_ORDER(__m)                                                 \
  {                                                                                                                   \
    return __atomic_load_dispatch(&__a, __m, _Sco{});                                                                 \
  }                                                                                                                   \
  _CCCL_HOST_DEVICE_API inline operator _VT() const _VOLATILE noexcept                                                \
  {                                                                                                                   \
    return load();                                                                                                    \
  }                                                                                                                   \
  _CCCL_HOST_DEVICE_API inline _VT exchange(_VT __d, memory_order __m = memory_order_seq_cst)                         \
    _CONST _VOLATILE noexcept                                                                                         \
  {                                                                                                                   \
    return __atomic_exchange_dispatch(&__a, __d, __m, _Sco{});                                                        \
  }                                                                                                                   \
  _CCCL_HOST_DEVICE_API inline bool compare_exchange_weak(_VT& __e, _VT __d, memory_order __s, memory_order __f)      \
    _CONST _VOLATILE noexcept _LIBCUDACXX_CHECK_EXCHANGE_MEMORY_ORDER(__s, __f)                                       \
  {                                                                                                                   \
    return __atomic_compare_exchange_weak_dispatch(&__a, &__e, __d, __s, __f, _Sco{});                                \
  }                                                                                                                   \
  _CCCL_HOST_DEVICE_API inline bool compare_exchange_strong(_VT& __e, _VT __d, memory_order __s, memory_order __f)    \
    _CONST _VOLATILE noexcept _LIBCUDACXX_CHECK_EXCHANGE_MEMORY_ORDER(__s, __f)                                       \
  {                                                                                                                   \
    return __atomic_compare_exchange_strong_dispatch(&__a, &__e, __d, __s, __f, _Sco{});                              \
  }                                                                                                                   \
  _CCCL_HOST_DEVICE_API inline bool compare_exchange_weak(_VT& __e, _VT __d, memory_order __m = memory_order_seq_cst) \
    _CONST _VOLATILE noexcept                                                                                         \
  {                                                                                                                   \
    if (memory_order_acq_rel == __m)                                                                                  \
      return __atomic_compare_exchange_weak_dispatch(&__a, &__e, __d, __m, memory_order_acquire, _Sco{});             \
    else if (memory_order_release == __m)                                                                             \
      return __atomic_compare_exchange_weak_dispatch(&__a, &__e, __d, __m, memory_order_relaxed, _Sco{});             \
    else                                                                                                              \
      return __atomic_compare_exchange_weak_dispatch(&__a, &__e, __d, __m, __m, _Sco{});                              \
  }                                                                                                                   \
  _CCCL_HOST_DEVICE_API inline bool compare_exchange_strong(                                                          \
    _VT& __e, _VT __d, memory_order __m = memory_order_seq_cst) _CONST _VOLATILE noexcept                             \
  {                                                                                                                   \
    if (memory_order_acq_rel == __m)                                                                                  \
      return __atomic_compare_exchange_strong_dispatch(&__a, &__e, __d, __m, memory_order_acquire, _Sco{});           \
    else if (memory_order_release == __m)                                                                             \
      return __atomic_compare_exchange_strong_dispatch(&__a, &__e, __d, __m, memory_order_relaxed, _Sco{});           \
    else                                                                                                              \
      return __atomic_compare_exchange_strong_dispatch(&__a, &__e, __d, __m, __m, _Sco{});                            \
  }                                                                                                                   \
  _CCCL_HOST_DEVICE_API inline void wait(_VT __v, memory_order __m = memory_order_seq_cst) const _VOLATILE noexcept   \
  {                                                                                                                   \
    __atomic_wait(&__a, __v, __m, _Sco{});                                                                            \
  }                                                                                                                   \
  _CCCL_HOST_DEVICE_API inline void notify_one() _CONST _VOLATILE noexcept                                            \
  {                                                                                                                   \
    __atomic_notify_one(&__a, _Sco{});                                                                                \
  }                                                                                                                   \
  _CCCL_HOST_DEVICE_API inline void notify_all() _CONST _VOLATILE noexcept                                            \
  {                                                                                                                   \
    __atomic_notify_all(&__a, _Sco{});                                                                                \
  }

// API definitions for arithmetic atomics
#define _LIBCUDACXX_ATOMIC_ARITHMETIC_IMPL(_CONST, _VOLATILE, _VT)                              \
  _CCCL_HOST_DEVICE_API inline _VT fetch_add(_VT __op, memory_order __m = memory_order_seq_cst) \
    _CONST _VOLATILE noexcept                                                                   \
  {                                                                                             \
    return __atomic_fetch_add_dispatch(&__a, __op, __m, _Sco{});                                \
  }                                                                                             \
  _CCCL_HOST_DEVICE_API inline _VT fetch_sub(_VT __op, memory_order __m = memory_order_seq_cst) \
    _CONST _VOLATILE noexcept                                                                   \
  {                                                                                             \
    return __atomic_fetch_sub_dispatch(&__a, __op, __m, _Sco{});                                \
  }                                                                                             \
  _CCCL_HOST_DEVICE_API inline _VT operator++(int) _CONST _VOLATILE noexcept                    \
  {                                                                                             \
    return fetch_add(_VT(1));                                                                   \
  }                                                                                             \
  _CCCL_HOST_DEVICE_API inline _VT operator--(int) _CONST _VOLATILE noexcept                    \
  {                                                                                             \
    return fetch_sub(_VT(1));                                                                   \
  }                                                                                             \
  _CCCL_HOST_DEVICE_API inline _VT operator++() _CONST _VOLATILE noexcept                       \
  {                                                                                             \
    return fetch_add(_VT(1)) + _VT(1);                                                          \
  }                                                                                             \
  _CCCL_HOST_DEVICE_API inline _VT operator--() _CONST _VOLATILE noexcept                       \
  {                                                                                             \
    return fetch_sub(_VT(1)) - _VT(1);                                                          \
  }                                                                                             \
  _CCCL_HOST_DEVICE_API inline _VT operator+=(_VT __op) _CONST _VOLATILE noexcept               \
  {                                                                                             \
    return fetch_add(__op) + __op;                                                              \
  }                                                                                             \
  _CCCL_HOST_DEVICE_API inline _VT operator-=(_VT __op) _CONST _VOLATILE noexcept               \
  {                                                                                             \
    return fetch_sub(__op) - __op;                                                              \
  }

// API definitions for bitwise atomics
#define _LIBCUDACXX_ATOMIC_BITWISE_IMPL(_CONST, _VOLATILE, _VT)                                 \
  _CCCL_HOST_DEVICE_API inline _VT fetch_and(_VT __op, memory_order __m = memory_order_seq_cst) \
    _CONST _VOLATILE noexcept                                                                   \
  {                                                                                             \
    return __atomic_fetch_and_dispatch(&__a, __op, __m, _Sco{});                                \
  }                                                                                             \
  _CCCL_HOST_DEVICE_API inline _VT fetch_or(_VT __op, memory_order __m = memory_order_seq_cst)  \
    _CONST _VOLATILE noexcept                                                                   \
  {                                                                                             \
    return __atomic_fetch_or_dispatch(&__a, __op, __m, _Sco{});                                 \
  }                                                                                             \
  _CCCL_HOST_DEVICE_API inline _VT fetch_xor(_VT __op, memory_order __m = memory_order_seq_cst) \
    _CONST _VOLATILE noexcept                                                                   \
  {                                                                                             \
    return __atomic_fetch_xor_dispatch(&__a, __op, __m, _Sco{});                                \
  }                                                                                             \
  _CCCL_HOST_DEVICE_API inline _VT operator&=(_VT __op) _CONST _VOLATILE noexcept               \
  {                                                                                             \
    return fetch_and(__op) & __op;                                                              \
  }                                                                                             \
  _CCCL_HOST_DEVICE_API inline _VT operator|=(_VT __op) _CONST _VOLATILE noexcept               \
  {                                                                                             \
    return fetch_or(__op) | __op;                                                               \
  }                                                                                             \
  _CCCL_HOST_DEVICE_API inline _VT operator^=(_VT __op) _CONST _VOLATILE noexcept               \
  {                                                                                             \
    return fetch_xor(__op) ^ __op;                                                              \
  }

// API definitions for atomics with pointers
#define _LIBCUDACXX_ATOMIC_POINTER_IMPL(_CONST, _VOLATILE, _VT)                                       \
  _CCCL_HOST_DEVICE_API inline _VT fetch_add(ptrdiff_t __op, memory_order __m = memory_order_seq_cst) \
    _CONST _VOLATILE noexcept                                                                         \
  {                                                                                                   \
    return __atomic_fetch_add_dispatch(&__a, __op, __m, __thread_scope_system_tag{});                 \
  }                                                                                                   \
  _CCCL_HOST_DEVICE_API inline _VT fetch_sub(ptrdiff_t __op, memory_order __m = memory_order_seq_cst) \
    _CONST _VOLATILE noexcept                                                                         \
  {                                                                                                   \
    return __atomic_fetch_sub_dispatch(&__a, __op, __m, __thread_scope_system_tag{});                 \
  }                                                                                                   \
  _CCCL_HOST_DEVICE_API inline _VT operator++(int) _CONST _VOLATILE noexcept                          \
  {                                                                                                   \
    return fetch_add(1);                                                                              \
  }                                                                                                   \
  _CCCL_HOST_DEVICE_API inline _VT operator--(int) _CONST _VOLATILE noexcept                          \
  {                                                                                                   \
    return fetch_sub(1);                                                                              \
  }                                                                                                   \
  _CCCL_HOST_DEVICE_API inline _VT operator++() _CONST _VOLATILE noexcept                             \
  {                                                                                                   \
    return fetch_add(1) + 1;                                                                          \
  }                                                                                                   \
  _CCCL_HOST_DEVICE_API inline _VT operator--() _CONST _VOLATILE noexcept                             \
  {                                                                                                   \
    return fetch_sub(1) - 1;                                                                          \
  }                                                                                                   \
  _CCCL_HOST_DEVICE_API inline _VT operator+=(ptrdiff_t __op) _CONST _VOLATILE noexcept               \
  {                                                                                                   \
    return fetch_add(__op) + __op;                                                                    \
  }                                                                                                   \
  _CCCL_HOST_DEVICE_API inline _VT operator-=(ptrdiff_t __op) _CONST _VOLATILE noexcept               \
  {                                                                                                   \
    return fetch_sub(__op) - __op;                                                                    \
  }

#endif // __CUDA_STD___ATOMIC_API_COMMON_H
