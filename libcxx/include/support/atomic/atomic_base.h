// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX_ATOMIC_BASE_H
#define _LIBCUDACXX_ATOMIC_BASE_H

#include "cxx_atomic.h"

_LIBCUDACXX_INLINE_VISIBILITY inline _LIBCUDACXX_CONSTEXPR int __cxx_atomic_order_to_int(memory_order __order) {
  // Avoid switch statement to make this a constexpr.
  return __order == memory_order_relaxed ? __ATOMIC_RELAXED:
         (__order == memory_order_acquire ? __ATOMIC_ACQUIRE:
          (__order == memory_order_release ? __ATOMIC_RELEASE:
           (__order == memory_order_seq_cst ? __ATOMIC_SEQ_CST:
            (__order == memory_order_acq_rel ? __ATOMIC_ACQ_REL:
              __ATOMIC_CONSUME))));
}

_LIBCUDACXX_INLINE_VISIBILITY inline _LIBCUDACXX_CONSTEXPR int __cxx_atomic_failure_order_to_int(memory_order __order) {
  // Avoid switch statement to make this a constexpr.
  return __order == memory_order_relaxed ? __ATOMIC_RELAXED:
         (__order == memory_order_acquire ? __ATOMIC_ACQUIRE:
          (__order == memory_order_release ? __ATOMIC_RELAXED:
           (__order == memory_order_seq_cst ? __ATOMIC_SEQ_CST:
            (__order == memory_order_acq_rel ? __ATOMIC_ACQUIRE:
              __ATOMIC_CONSUME))));
}

template <typename _Tp, typename _Up>
inline void __cxx_atomic_init(volatile _Tp* __a,  _Up __val) {
  auto __a_tmp = __cxx_atomic_base_unwrap(__a);
  __cxx_atomic_assign_volatile(*__a_tmp, __val);
}

template <typename _Tp, typename _Up>
inline void __cxx_atomic_init(_Tp* __a,  _Up __val) {
  auto __a_tmp = __cxx_atomic_base_unwrap(__a);
  *__a_tmp = __val;
}

inline
void __cxx_atomic_thread_fence(memory_order __order) {
  __atomic_thread_fence(__cxx_atomic_order_to_int(__order));
}

inline
void __cxx_atomic_signal_fence(memory_order __order) {
  __atomic_signal_fence(__cxx_atomic_order_to_int(__order));
}

template <typename _Tp, typename _Up>
inline void __cxx_atomic_store(_Tp* __a,  _Up __val,
                        memory_order __order) {
  typename _CUDA_VSTD::remove_cv<_Tp>::type __v_temp(__val);
  (void)__a;
  __atomic_store(__a, &__v_temp, __cxx_atomic_order_to_int(__order));
}

template <typename _Tp>
inline auto __cxx_atomic_load(const _Tp* __a,
                       memory_order __order) -> __cxx_atomic_underlying_t<_Tp> {
  typename _CUDA_VSTD::remove_cv<_Tp>::type __ret;
  (void)__a;
  __atomic_load(__a, &__ret, __cxx_atomic_order_to_int(__order));
  return __ret.__a_value;
}

template <typename _Tp, typename _Up>
inline auto __cxx_atomic_exchange(_Tp* __a, _Up __value,
                          memory_order __order) -> __cxx_atomic_underlying_t<_Tp> {
  typename _CUDA_VSTD::remove_cv<_Tp>::type __v_temp(__value);
  typename _CUDA_VSTD::remove_cv<_Tp>::type __ret;
  (void)__a;
  __atomic_exchange(__a, &__v_temp, &__ret, __cxx_atomic_order_to_int(__order));
  return __ret.__a_value;
}

template <typename _Tp, typename _Up>
inline bool __cxx_atomic_compare_exchange_strong(
    _Tp* __a, _Up* __expected, _Up __value, memory_order __success,
    memory_order __failure) {
  auto __a_tmp = __cxx_atomic_base_unwrap(__a);
  (void)__a_tmp;
  (void)__expected;
  return __atomic_compare_exchange(__a_tmp, __expected, &__value,
                                   false,
                                   __cxx_atomic_order_to_int(__success),
                                   __cxx_atomic_failure_order_to_int(__failure));
}

template <typename _Tp, typename _Up>
inline bool __cxx_atomic_compare_exchange_weak(
    _Tp* __a, _Up* __expected, _Up __value, memory_order __success,
    memory_order __failure) {
  auto __a_tmp = __cxx_atomic_base_unwrap(__a);
  (void)__a_tmp;
  (void)__expected;
  return __atomic_compare_exchange(__a_tmp, __expected, &__value,
                                   true,
                                   __cxx_atomic_order_to_int(__success),
                                   __cxx_atomic_failure_order_to_int(__failure));
}

template <typename _Tp>
struct __atomic_ptr_inc { enum {value = 1}; };

template <typename _Tp>
struct __atomic_ptr_inc<_Tp*> { enum {value = sizeof(_Tp)}; };

// FIXME: Haven't figured out what the spec says about using arrays with
// atomic_fetch_add. Force a failure rather than creating bad behavior.
template <typename _Tp>
struct __atomic_ptr_inc<_Tp[]> { };
template <typename _Tp, int n>
struct __atomic_ptr_inc<_Tp[n]> { };

template <typename _Tp, typename _Td>
inline auto __cxx_atomic_fetch_add(_Tp* __a, _Td __delta,
                           memory_order __order) -> __cxx_atomic_underlying_t<_Tp> {
  constexpr auto __skip_v = __atomic_ptr_inc<__cxx_atomic_underlying_t<_Tp>>::value;
  auto __a_tmp = __cxx_atomic_base_unwrap(__a);
  return __atomic_fetch_add(__a_tmp, __delta * __skip_v,
                            __cxx_atomic_order_to_int(__order));
}

template <typename _Tp, typename _Td>
inline auto __cxx_atomic_fetch_sub(_Tp* __a, _Td __delta,
                           memory_order __order) -> __cxx_atomic_underlying_t<_Tp> {
  constexpr auto __skip_v = __atomic_ptr_inc<__cxx_atomic_underlying_t<_Tp>>::value;
  auto __a_tmp = __cxx_atomic_base_unwrap(__a);
  return __atomic_fetch_sub(__a_tmp, __delta * __skip_v,
                            __cxx_atomic_order_to_int(__order));
}

template <typename _Tp, typename _Td>
inline auto __cxx_atomic_fetch_and(_Tp* __a, _Td __pattern,
                            memory_order __order) -> __cxx_atomic_underlying_t<_Tp> {
  auto __a_tmp = __cxx_atomic_base_unwrap(__a);
  return __atomic_fetch_and(__a_tmp, __pattern,
                            __cxx_atomic_order_to_int(__order));
}

template <typename _Tp, typename _Td>
inline auto __cxx_atomic_fetch_or(_Tp* __a, _Td __pattern,
                          memory_order __order) -> __cxx_atomic_underlying_t<_Tp> {
  auto __a_tmp = __cxx_atomic_base_unwrap(__a);
  return __atomic_fetch_or(__a_tmp, __pattern,
                           __cxx_atomic_order_to_int(__order));
}

template <typename _Tp, typename _Td>
inline auto __cxx_atomic_fetch_xor(_Tp* __a, _Td __pattern,
                           memory_order __order) -> __cxx_atomic_underlying_t<_Tp> {
  auto __a_tmp = __cxx_atomic_base_unwrap(__a);
  return __atomic_fetch_xor(__a_tmp, __pattern,
                            __cxx_atomic_order_to_int(__order));
}

inline constexpr
 bool __cxx_atomic_is_lock_free(size_t __x) {
  #if defined(_LIBCUDACXX_NO_RUNTIME_LOCK_FREE)
    return __x <= 8;
  #else
    return __atomic_is_lock_free(__x, 0);
  #endif
}

#endif // _LIBCUDACXX_ATOMIC_BASE_H
