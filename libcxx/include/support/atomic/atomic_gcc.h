//

template <typename _Tp>
struct __cxx_atomic_base_impl {

  _LIBCUDACXX_INLINE_VISIBILITY
#ifndef _LIBCUDACXX_CXX03_LANG
    __cxx_atomic_base_impl() _NOEXCEPT = default;
#else
    __cxx_atomic_base_impl() _NOEXCEPT : __a_value() {}
#endif // _LIBCUDACXX_CXX03_LANG
  _LIBCUDACXX_CONSTEXPR explicit __cxx_atomic_base_impl(_Tp value) _NOEXCEPT
    : __a_value(value) {}
  _ALIGNAS(sizeof(_Tp)) _Tp __a_value;
};

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

template <typename _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
void __cxx_atomic_init(volatile __cxx_atomic_base_impl<_Tp>* __a,  _Tp __val) {
  __cxx_atomic_assign_volatile(__a->__a_value, __val);
}

template <typename _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
void __cxx_atomic_init(__cxx_atomic_base_impl<_Tp>* __a,  _Tp __val) {
  __a->__a_value = __val;
}

_LIBCUDACXX_INLINE_VISIBILITY inline
void __cxx_atomic_thread_fence(memory_order __order) {
  __atomic_thread_fence(__to_gcc_order(__order));
}

_LIBCUDACXX_INLINE_VISIBILITY inline
void __cxx_atomic_signal_fence(memory_order __order) {
  __atomic_signal_fence(__to_gcc_order(__order));
}

template <typename _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
void __cxx_atomic_store(volatile __cxx_atomic_base_impl<_Tp>* __a,  _Tp __val,
                        memory_order __order) {
  __atomic_store(&__a->__a_value, &__val,
                 __to_gcc_order(__order));
}

template <typename _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
void __cxx_atomic_store(__cxx_atomic_base_impl<_Tp>* __a,  _Tp __val,
                        memory_order __order) {
  __atomic_store(&__a->__a_value, &__val,
                 __to_gcc_order(__order));
}

template <typename _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
_Tp __cxx_atomic_load(const volatile __cxx_atomic_base_impl<_Tp>* __a,
                      memory_order __order) {
  _Tp __ret;
  __atomic_load(&__a->__a_value, &__ret,
                __to_gcc_order(__order));
  return __ret;
}

template <typename _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
_Tp __cxx_atomic_load(const __cxx_atomic_base_impl<_Tp>* __a, memory_order __order) {
  _Tp __ret;
  __atomic_load(&__a->__a_value, &__ret,
                __to_gcc_order(__order));
  return __ret;
}

template <typename _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
_Tp __cxx_atomic_exchange(volatile __cxx_atomic_base_impl<_Tp>* __a,
                          _Tp __value, memory_order __order) {
  _Tp __ret;
  __atomic_exchange(&__a->__a_value, &__value, &__ret,
                    __to_gcc_order(__order));
  return __ret;
}

template <typename _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
_Tp __cxx_atomic_exchange(__cxx_atomic_base_impl<_Tp>* __a, _Tp __value,
                          memory_order __order) {
  _Tp __ret;
  __atomic_exchange(&__a->__a_value, &__value, &__ret,
                    __to_gcc_order(__order));
  return __ret;
}

template <typename _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
bool __cxx_atomic_compare_exchange_strong(
    volatile __cxx_atomic_base_impl<_Tp>* __a, _Tp* __expected, _Tp __value,
    memory_order __success, memory_order __failure) {
  return __atomic_compare_exchange(&__a->__a_value, __expected, &__value,
                                   false,
                                   __to_gcc_order(__success),
                                   __to_gcc_failure_order(__failure));
}

template <typename _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
bool __cxx_atomic_compare_exchange_strong(
    __cxx_atomic_base_impl<_Tp>* __a, _Tp* __expected, _Tp __value, memory_order __success,
    memory_order __failure) {
  return __atomic_compare_exchange(&__a->__a_value, __expected, &__value,
                                   false,
                                   __to_gcc_order(__success),
                                   __to_gcc_failure_order(__failure));
}

template <typename _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
bool __cxx_atomic_compare_exchange_weak(
    volatile __cxx_atomic_base_impl<_Tp>* __a, _Tp* __expected, _Tp __value,
    memory_order __success, memory_order __failure) {
  return __atomic_compare_exchange(&__a->__a_value, __expected, &__value,
                                   true,
                                   __to_gcc_order(__success),
                                   __to_gcc_failure_order(__failure));
}

template <typename _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
bool __cxx_atomic_compare_exchange_weak(
    __cxx_atomic_base_impl<_Tp>* __a, _Tp* __expected, _Tp __value, memory_order __success,
    memory_order __failure) {
  return __atomic_compare_exchange(&__a->__a_value, __expected, &__value,
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
_LIBCUDACXX_INLINE_VISIBILITY
_Tp __cxx_atomic_fetch_add(volatile __cxx_atomic_base_impl<_Tp>* __a,
                           _Td __delta, memory_order __order) {
  return __atomic_fetch_add(&__a->__a_value, __delta * __skip_amt<_Tp>::value,
                            __to_gcc_order(__order));
}

template <typename _Tp, typename _Td>
_LIBCUDACXX_INLINE_VISIBILITY
_Tp __cxx_atomic_fetch_add(__cxx_atomic_base_impl<_Tp>* __a, _Td __delta,
                           memory_order __order) {
  return __atomic_fetch_add(&__a->__a_value, __delta * __skip_amt<_Tp>::value,
                            __to_gcc_order(__order));
}

template <typename _Tp, typename _Td>
_LIBCUDACXX_INLINE_VISIBILITY
_Tp __cxx_atomic_fetch_sub(volatile __cxx_atomic_base_impl<_Tp>* __a,
                           _Td __delta, memory_order __order) {
  return __atomic_fetch_sub(&__a->__a_value, __delta * __skip_amt<_Tp>::value,
                            __to_gcc_order(__order));
}

template <typename _Tp, typename _Td>
_LIBCUDACXX_INLINE_VISIBILITY
_Tp __cxx_atomic_fetch_sub(__cxx_atomic_base_impl<_Tp>* __a, _Td __delta,
                           memory_order __order) {
  return __atomic_fetch_sub(&__a->__a_value, __delta * __skip_amt<_Tp>::value,
                            __to_gcc_order(__order));
}

template <typename _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
_Tp __cxx_atomic_fetch_and(volatile __cxx_atomic_base_impl<_Tp>* __a,
                           _Tp __pattern, memory_order __order) {
  return __atomic_fetch_and(&__a->__a_value, __pattern,
                            __to_gcc_order(__order));
}

template <typename _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
_Tp __cxx_atomic_fetch_and(__cxx_atomic_base_impl<_Tp>* __a,
                           _Tp __pattern, memory_order __order) {
  return __atomic_fetch_and(&__a->__a_value, __pattern,
                            __to_gcc_order(__order));
}

template <typename _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
_Tp __cxx_atomic_fetch_or(volatile __cxx_atomic_base_impl<_Tp>* __a,
                          _Tp __pattern, memory_order __order) {
  return __atomic_fetch_or(&__a->__a_value, __pattern,
                           __to_gcc_order(__order));
}

template <typename _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
_Tp __cxx_atomic_fetch_or(__cxx_atomic_base_impl<_Tp>* __a, _Tp __pattern,
                          memory_order __order) {
  return __atomic_fetch_or(&__a->__a_value, __pattern,
                           __to_gcc_order(__order));
}

template <typename _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
_Tp __cxx_atomic_fetch_xor(volatile __cxx_atomic_base_impl<_Tp>* __a,
                           _Tp __pattern, memory_order __order) {
  return __atomic_fetch_xor(&__a->__a_value, __pattern,
                            __to_gcc_order(__order));
}

template <typename _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
_Tp __cxx_atomic_fetch_xor(__cxx_atomic_base_impl<_Tp>* __a, _Tp __pattern,
                           memory_order __order) {
  return __atomic_fetch_xor(&__a->__a_value, __pattern,
                            __to_gcc_order(__order));
}

#define __cxx_atomic_is_lock_free(__s) __atomic_is_lock_free(__s, 0)
