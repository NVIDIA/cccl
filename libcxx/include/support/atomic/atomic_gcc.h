// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

template <typename _Tp, int _Sco>
struct __cxx_atomic_base_impl {
  using __cxx_underlying_type = _Tp;

  _LIBCUDACXX_CONSTEXPR
  __cxx_atomic_base_impl() _NOEXCEPT = default;

  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR explicit
  __cxx_atomic_base_impl(_Tp value) _NOEXCEPT : __a_value(value) {}

  _ALIGNAS(sizeof(_Tp)) _Tp __a_value;

  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
  const volatile _Tp* __get_atom() const volatile _NOEXCEPT {return &__a_value;}

  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
  const _Tp* __get_atom() const _NOEXCEPT {return &__a_value;}

  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
  volatile _Tp* __get_atom() volatile _NOEXCEPT {return &__a_value;}
};

template <typename _Tp, int _Sco>
struct __cxx_atomic_ref_base_impl {
  using __cxx_underlying_type = _Tp;

  _LIBCUDACXX_CONSTEXPR
  __cxx_atomic_ref_base_impl() _NOEXCEPT = default;

  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR explicit
  __cxx_atomic_ref_base_impl(_Tp value) _NOEXCEPT : __a_value(value) {}

  _Tp* __a_value;

  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
  const volatile _Tp* __get_atom() const volatile _NOEXCEPT {return __a_value;}

  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
  const _Tp* __get_atom() const _NOEXCEPT {return __a_value;}

  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
  volatile _Tp* __get_atom() volatile _NOEXCEPT {return __a_value;}
};

template <typename _Tp>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR auto __cxx_atomic_base_unwrap(_Tp* __a) _NOEXCEPT -> decltype(__a->__get_atom()) {
  return __a->__get_atom();
}

template <typename _Tp>
using __cxx_atomic_underlying_t = typename _Tp::__cxx_underlying_type;

_LIBCUDACXX_INLINE_VISIBILITY inline _LIBCUDACXX_CONSTEXPR int __to_gcc_order(memory_order __order) {
  // Avoid switch statement to make this a constexpr.
  return __order == memory_order_relaxed ? __ATOMIC_RELAXED:
         (__order == memory_order_acquire ? __ATOMIC_ACQUIRE:
          (__order == memory_order_release ? __ATOMIC_RELEASE:
           (__order == memory_order_seq_cst ? __ATOMIC_SEQ_CST:
            (__order == memory_order_acq_rel ? __ATOMIC_ACQ_REL:
              __ATOMIC_CONSUME))));
}

_LIBCUDACXX_INLINE_VISIBILITY inline _LIBCUDACXX_CONSTEXPR int __to_gcc_failure_order(memory_order __order) {
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
  __a = __val;
}

inline
void __cxx_atomic_thread_fence(memory_order __order) {
  __atomic_thread_fence(__to_gcc_order(__order));
}

inline
void __cxx_atomic_signal_fence(memory_order __order) {
  __atomic_signal_fence(__to_gcc_order(__order));
}

template <typename _Tp, typename _Up>
inline void __cxx_atomic_store(_Tp* __a,  _Up __val,
                        memory_order __order) {
  auto __a_tmp = __cxx_atomic_base_unwrap(__a);
  __atomic_store(__a_tmp, &__val, __to_gcc_order(__order));
}

template <typename _Tp>
inline auto __cxx_atomic_load(const _Tp* __a,
                       memory_order __order) -> __cxx_atomic_underlying_t<_Tp> {
  auto __a_tmp = __cxx_atomic_base_unwrap(__a);
  __cxx_atomic_underlying_t<_Tp> __ret;
  __atomic_load(__a_tmp, &__ret, __to_gcc_order(__order));
  return __ret;
}

template <typename _Tp, typename _Up>
inline auto __cxx_atomic_exchange(_Tp* __a, _Up __value,
                          memory_order __order) -> __cxx_atomic_underlying_t<_Tp> {
  auto __a_tmp = __cxx_atomic_base_unwrap(__a);
  __cxx_atomic_underlying_t<_Tp> __ret;
  __atomic_exchange(__a_tmp, &__value, &__ret, __to_gcc_order(__order));
  return __ret;
}

template <typename _Tp, typename _Up>
inline bool __cxx_atomic_compare_exchange_strong(
    _Tp* __a, _Up* __expected, _Up __value, memory_order __success,
    memory_order __failure) {
  auto __a_tmp = __cxx_atomic_base_unwrap(__a);
  return __atomic_compare_exchange(__a_tmp, __expected, &__value,
                                   false,
                                   __to_gcc_order(__success),
                                   __to_gcc_failure_order(__failure));
}

template <typename _Tp, typename _Up>
inline bool __cxx_atomic_compare_exchange_weak(
    _Tp* __a, _Up* __expected, _Up __value, memory_order __success,
    memory_order __failure) {
  auto __a_tmp = __cxx_atomic_base_unwrap(__a);
  return __atomic_compare_exchange(__a_tmp, __expected, &__value,
                                   true,
                                   __to_gcc_order(__success),
                                   __to_gcc_failure_order(__failure));
}

template <typename _Tp>
struct __skip_amt { enum {value = 1}; };

template <typename _Tp>
struct __skip_amt<_Tp*> { enum {value = sizeof(_Tp)}; };

// FIXME: Haven't figured out what the spec says about using arrays with
// atomic_fetch_add. Force a failure rather than creating bad behavior.
template <typename _Tp>
struct __skip_amt<_Tp[]> { };
template <typename _Tp, int n>
struct __skip_amt<_Tp[n]> { };

template <typename _Tp, typename _Td>
inline auto __cxx_atomic_fetch_add(_Tp* __a, _Td __delta,
                           memory_order __order) -> __cxx_atomic_underlying_t<_Tp> {
  constexpr auto __skip_v = __skip_amt<__cxx_atomic_underlying_t<_Tp>>::value;
  auto __a_tmp = __cxx_atomic_base_unwrap(__a);
  return __atomic_fetch_add(__a_tmp, __delta * __skip_v,
                            __to_gcc_order(__order));
}

template <typename _Tp, typename _Td>
inline auto __cxx_atomic_fetch_sub(_Tp* __a, _Td __delta,
                           memory_order __order) -> __cxx_atomic_underlying_t<_Tp> {
  constexpr auto __skip_v = __skip_amt<__cxx_atomic_underlying_t<_Tp>>::value;
  auto __a_tmp = __cxx_atomic_base_unwrap(__a);
  return __atomic_fetch_sub(__a_tmp, __delta * __skip_v,
                            __to_gcc_order(__order));
}

template <typename _Tp, typename _Td>
inline auto __cxx_atomic_fetch_and(_Tp* __a, _Td __pattern,
                            memory_order __order) -> __cxx_atomic_underlying_t<_Tp> {
  auto __a_tmp = __cxx_atomic_base_unwrap(__a);
  return __atomic_fetch_and(__a_tmp, __pattern,
                            __to_gcc_order(__order));
}

template <typename _Tp, typename _Td>
inline auto __cxx_atomic_fetch_or(_Tp* __a, _Td __pattern,
                          memory_order __order) -> __cxx_atomic_underlying_t<_Tp> {
  auto __a_tmp = __cxx_atomic_base_unwrap(__a);
  return __atomic_fetch_or(__a_tmp, __pattern,
                           __to_gcc_order(__order));
}

template <typename _Tp, typename _Td>
inline auto __cxx_atomic_fetch_xor(_Tp* __a, _Td __pattern,
                           memory_order __order) -> __cxx_atomic_underlying_t<_Tp> {
  auto __a_tmp = __cxx_atomic_base_unwrap(__a);
  return __atomic_fetch_xor(__a_tmp, __pattern,
                            __to_gcc_order(__order));
}

inline constexpr
 bool __cxx_atomic_is_lock_free(size_t __x) {
    return __atomic_is_lock_free(__x, 0);
}
