//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__CUDA_MINIMUM_ARCH__) && ((!defined(_MSC_VER) && __CUDA_MINIMUM_ARCH__ < 600) || (defined(_MSC_VER) && __CUDA_MINIMUM_ARCH__ < 700))
#  error "CUDA atomics are only supported for sm_60 and up on *nix and sm_70 and up on Windows."
#endif

#ifndef __CUDACC_RTC__
#include <string.h>
#include <assert.h>
#endif // __CUDACC_RTC__

#if !defined(__CLANG_ATOMIC_BOOL_LOCK_FREE) && !defined(__GCC_ATOMIC_BOOL_LOCK_FREE)
#define ATOMIC_BOOL_LOCK_FREE      2
#define ATOMIC_CHAR_LOCK_FREE      2
#define ATOMIC_CHAR16_T_LOCK_FREE  2
#define ATOMIC_CHAR32_T_LOCK_FREE  2
#define ATOMIC_WCHAR_T_LOCK_FREE   2
#define ATOMIC_SHORT_LOCK_FREE     2
#define ATOMIC_INT_LOCK_FREE       2
#define ATOMIC_LONG_LOCK_FREE      2
#define ATOMIC_LLONG_LOCK_FREE     2
#define ATOMIC_POINTER_LOCK_FREE   2
#endif //!defined(__CLANG_ATOMIC_BOOL_LOCK_FREE) && !defined(__GCC_ATOMIC_BOOL_LOCK_FREE)

#ifndef __ATOMIC_RELAXED
#define __ATOMIC_RELAXED 0
#define __ATOMIC_CONSUME 1
#define __ATOMIC_ACQUIRE 2
#define __ATOMIC_RELEASE 3
#define __ATOMIC_ACQ_REL 4
#define __ATOMIC_SEQ_CST 5
#endif //__ATOMIC_RELAXED

#ifndef __ATOMIC_BLOCK
#define __ATOMIC_SYSTEM 0 // 0 indicates default
#define __ATOMIC_DEVICE 1
#define __ATOMIC_BLOCK 2
#define __ATOMIC_THREAD 10
#endif //__ATOMIC_BLOCK

// TODO:
// How to get this into cuda::???

inline __host__ __device__ int __stronger_order_cuda(int __a, int __b) {
    int const __max = __a > __b ? __a : __b;
    if(__max != __ATOMIC_RELEASE)
        return __max;
    static int const __xform[] = {
        __ATOMIC_RELEASE,
        __ATOMIC_ACQ_REL,
        __ATOMIC_ACQ_REL,
        __ATOMIC_RELEASE };
    return __xform[__a < __b ? __a : __b];
}

enum thread_scope {
    thread_scope_system = __ATOMIC_SYSTEM,
    thread_scope_device = __ATOMIC_DEVICE,
    thread_scope_block = __ATOMIC_BLOCK,
    thread_scope_thread = __ATOMIC_THREAD
};

#define _LIBCUDACXX_ATOMIC_SCOPE_TYPE ::cuda::thread_scope
#define _LIBCUDACXX_ATOMIC_SCOPE_DEFAULT ::cuda::thread_scope::system

struct __thread_scope_thread_tag { };
struct __thread_scope_block_tag { };
struct __thread_scope_device_tag { };
struct __thread_scope_system_tag { };

template<int _Scope>  struct __scope_enum_to_tag { };
/* This would be the implementation once an actual thread-scope backend exists.
template<> struct __scope_enum_to_tag<(int)thread_scope_thread> {
    using type = __thread_scope_thread_tag; };
Until then: */
template<> struct __scope_enum_to_tag<(int)thread_scope_thread> {
    using type = __thread_scope_block_tag; };
template<> struct __scope_enum_to_tag<(int)thread_scope_block> {
    using type = __thread_scope_block_tag; };
template<> struct __scope_enum_to_tag<(int)thread_scope_device> {
    using type = __thread_scope_device_tag; };
template<> struct __scope_enum_to_tag<(int)thread_scope_system> {
    using type = __thread_scope_system_tag; };

template<int _Scope>
__host__ __device__ auto constexpr __scope_tag() ->
        typename __scope_enum_to_tag<_Scope>::type {
    return typename __scope_enum_to_tag<_Scope>::type();
}
// END TODO

// Wrap host atomic implementations into a sub-namespace
namespace host {
#if defined(_LIBCUDACXX_COMPILER_MSVC)
#  include "atomic_msvc.h"
#elif defined (_LIBCUDACXX_HAS_GCC_ATOMIC_IMP)
#  include "atomic_gcc.h"
#elif defined (_LIBCUDACXX_HAS_C11_ATOMIC_IMP)
//TODO
// #  include "atomic_c11.h"
#endif
}

#include "atomic_cuda_generated.h"
#include "atomic_cuda_derived.h"

template <typename _Tp>
struct __skip_amt { enum {value = 1}; };

template <typename _Tp>
struct __skip_amt<_Tp*> { enum {value = sizeof(_Tp)}; };

// Forward-declare the function templates that are defined libcxx later.
template <typename _Tp, typename _Tv> _LIBCUDACXX_INLINE_VISIBILITY
typename enable_if<is_assignable<_Tp&, _Tv>::value>::type
__cxx_atomic_assign_volatile(_Tp& __a_value, _Tv const& __val);

template <typename _Tp, typename _Tv> _LIBCUDACXX_INLINE_VISIBILITY
typename enable_if<is_assignable<_Tp&, _Tv>::value>::type
__cxx_atomic_assign_volatile(_Tp volatile& __a_value, _Tv volatile const& __val);

__host__ __device__ inline bool __cxx_atomic_is_lock_free(size_t __x) {
    return __x <= 8;
}
__host__ __device__ inline void __cxx_atomic_thread_fence(int __order) {
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            detail::__atomic_thread_fence_cuda(__order, detail::__thread_scope_system_tag());
        ),
        NV_IS_HOST, (
            host::__atomic_thread_fence(__order);
        )
    )
}
__host__ __device__ inline void __cxx_atomic_signal_fence(int __order) {
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            detail::__atomic_signal_fence_cuda(__order);
        ),
        NV_IS_HOST, (
            host::__atomic_signal_fence(__order);
        )
    )
}

// Atomic storage layouts:

// Implement _Sco with https://godbolt.org/z/foWdeYjEs

template <typename _Tp>
struct type {
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
    type() noexcept : __a_value() {
    }

    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
    type(_Tp __held) noexcept : __a_value(__held) {
    }

    _ALIGNAS(sizeof(_Tp)) _Tp __a_value;

    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
    _Tp* get() _NOEXCEPT {
        return &__a_value;
    }
}

template <typename _Tp, int _Sco>
struct __cxx_atomic_base_storage_aligned<_Tp> {

};

template <typename _Tp, int _Sco>
struct __cxx_atomic_base_storage_small {
    using __wrapped_type = _Tp;

    __cxx_atomic_base_storage_small() noexcept = default;
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR explicit
      __cxx_atomic_base_storage_small(_Tp __value) : __a_held(__value) {
    }

    __cxx_atomic_base_storage_aligned<uint32_t, _Sco> __a_held;

    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
      __cxx_atomic_base_storage_aligned<uint32_t, _Sco>* get() _NOEXCEPT {
        return &__a_held;
    }
};

template <typename _Tp, int _Sco>
using __cxx_atomic_base_storage = typename conditional<sizeof(_Tp) < 4,
                                    __cxx_atomic_base_storage_small<_Tp, _Sco>,
                                    __cxx_atomic_base_storage_aligned<_Tp, _Sco> >::type;

template <typename _Tp>
using __cxx_atomic_alignment_wrapper_t = __cxx_atomic_base_storage<_Tp>;

template <typename _Tp>
__host__ __device__ __cxx_atomic_alignment_wrapper_t<_Tp> __cxx_atomic_alignment_wrap(_Tp __value) {
    return __cxx_atomic_alignment_wrapper_t(__value);
}

template <typename _Tp>
__host__ __device__ _Tp __cxx_atomic_alignment_unwrap(_Tp __value, true_type) {
    return __value;
}
template <typename _Tp>
__host__ __device__ typename _Tp::__wrapped_type __cxx_atomic_alignment_unwrap(_Tp __value, false_type) {
    return *__value.get();
}
template <typename _Tp>
__host__ __device__ auto __cxx_atomic_alignment_unwrap(_Tp __value)
    -> decltype(__cxx_atomic_alignment_unwrap(__value, integral_constant<bool, _LIBCUDACXX_ALIGNOF(_Tp) == sizeof(_Tp)>{}))
{
    return __cxx_atomic_alignment_unwrap(__value, integral_constant<bool, _LIBCUDACXX_ALIGNOF(_Tp) == sizeof(_Tp)>{});
}

template<class _Tp, int _Sco>
__host__ __device__ inline void __cxx_atomic_init(__cxx_atomic_base_impl_default<_Tp, _Sco> volatile* __a, _Tp __val) {
    auto __tmp = __cxx_atomic_alignment_wrap(__val);
    __cxx_atomic_assign_volatile(__a->__a_value, __tmp);
}
template<class _Tp, int _Sco>
__host__ __device__ inline void __cxx_atomic_store(__cxx_atomic_base_impl_default<_Tp, _Sco> volatile* __a, _Tp __val, int __order) {
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            detail::__atomic_store_n_cuda(&__a->__a_value, __cxx_atomic_alignment_wrap(__val), __order, detail::__scope_tag<_Sco>());
        ),
        NV_IS_HOST, (
            auto __t = __cxx_atomic_alignment_wrap(__val);
            host::__atomic_store(&__a->__a_value, &__t, __order);
        )
    )
}
template<class _Tp, int _Sco>
__host__ __device__ inline _Tp __cxx_atomic_load(__cxx_atomic_base_impl_default<_Tp, _Sco> const volatile* __a, int __order) {
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            return __cxx_atomic_alignment_unwrap(detail::__atomic_load_n_cuda(&__a->__a_value, __order, detail::__scope_tag<_Sco>()));
        ),
        NV_IS_HOST, (
            alignas(_Tp) unsigned char __buf[sizeof(_Tp)];
            auto* __dest = reinterpret_cast<_Tp*>(__buf);
            host::__atomic_load(&__a->__a_value, __dest, __order);
            return __cxx_atomic_alignment_unwrap(*__dest);
        )
    )
}
template<class _Tp, int _Sco>
__host__ __device__ inline _Tp __cxx_atomic_exchange(__cxx_atomic_base_impl_default<_Tp, _Sco> volatile* __a, _Tp __val, int __order) {
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            return __cxx_atomic_alignment_unwrap(detail::__atomic_exchange_n_cuda(&__a->__a_value, __cxx_atomic_alignment_wrap(__val), __order, detail::__scope_tag<_Sco>()));
        ),
        NV_IS_HOST, (
            alignas(_Tp) unsigned char __buf[sizeof(_Tp)];
            auto* __dest = reinterpret_cast<_Tp*>(__buf);
            auto __t = __cxx_atomic_alignment_wrap(__val);
            host::__atomic_exchange(&__a->__a_value, &__t, __dest, __order);
            return __cxx_atomic_alignment_unwrap(*__dest);
        )
    )
}
template<class _Tp, int _Sco>
__host__ __device__ inline bool __cxx_atomic_compare_exchange_strong(__cxx_atomic_base_impl_default<_Tp, _Sco> volatile* __a, _Tp* __expected, _Tp __val, int __success, int __failure) {
    auto __tmp = __cxx_atomic_alignment_wrap(*__expected);
    bool __result = false;
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            __result = detail::__atomic_compare_exchange_n_cuda(&__a->__a_value, &__tmp, __cxx_atomic_alignment_wrap(__val), false, __success, __failure, detail::__scope_tag<_Sco>());
        ),
        NV_IS_HOST, (
            __result = host::__atomic_compare_exchange(&__a->__a_value, &__tmp, &__val, false, __success, __failure);
        )
    )
    *__expected = __cxx_atomic_alignment_unwrap(__tmp);
    return __result;
}
template<class _Tp, int _Sco>
__host__ __device__ inline bool __cxx_atomic_compare_exchange_weak(__cxx_atomic_base_impl_default<_Tp, _Sco> volatile* __a, _Tp* __expected, _Tp __val, int __success, int __failure) {
    auto __tmp = __cxx_atomic_alignment_wrap(*__expected);
    bool __result = false;
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            __result = detail::__atomic_compare_exchange_n_cuda(&__a->__a_value, &__tmp, __cxx_atomic_alignment_wrap(__val), true, __success, __failure, detail::__scope_tag<_Sco>());
        ),
        NV_IS_HOST, (
            __result = host::__atomic_compare_exchange(&__a->__a_value, &__tmp, &__val, true, __success, __failure);
        )
    )
    *__expected = __cxx_atomic_alignment_unwrap(__tmp);
    return __result;
}
template<class _Tp, int _Sco>
__host__ __device__ inline _Tp __cxx_atomic_fetch_add(__cxx_atomic_base_impl_default<_Tp, _Sco> volatile* __a, _Tp __delta, int __order) {
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            return detail::__atomic_fetch_add_cuda(&__a->__a_value, __delta, __order, detail::__scope_tag<_Sco>());
        ),
        NV_IS_HOST, (
            return host::__atomic_fetch_add(&__a->__a_value, __delta, __order);
        )
    )
}
template<class _Tp, int _Sco>
__host__ __device__ inline _Tp* __cxx_atomic_fetch_add(__cxx_atomic_base_impl_default<_Tp*, _Sco> volatile* __a, ptrdiff_t __delta, int __order) {
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            return detail::__atomic_fetch_add_cuda(&__a->__a_value, __delta, __order, detail::__scope_tag<_Sco>());
        ),
        NV_IS_HOST, (
            return host::__atomic_fetch_add(&__a->__a_value, __delta * __skip_amt<_Tp*>::value, __order);
        )
    )
}
template<class _Tp, int _Sco>
__host__ __device__ inline _Tp __cxx_atomic_fetch_sub(__cxx_atomic_base_impl_default<_Tp, _Sco> volatile* __a, _Tp __delta, int __order) {
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            return detail::__atomic_fetch_sub_cuda(&__a->__a_value, __delta, __order, detail::__scope_tag<_Sco>());
        ),
        NV_IS_HOST, (
            return host::__atomic_fetch_sub(&__a->__a_value, __delta, __order);
        )
    )
}
template<class _Tp, int _Sco>
__host__ __device__ inline _Tp* __cxx_atomic_fetch_sub(__cxx_atomic_base_impl_default<_Tp*, _Sco> volatile* __a, ptrdiff_t __delta, int __order) {
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            return detail::__atomic_fetch_sub_cuda(&__a->__a_value, __delta, __order, detail::__scope_tag<_Sco>());
        ),
        NV_IS_HOST, (
            return host::__atomic_fetch_sub(&__a->__a_value, __delta * __skip_amt<_Tp*>::value, __order);
        )
    )
}
template<class _Tp, int _Sco>
__host__ __device__ inline _Tp __cxx_atomic_fetch_and(__cxx_atomic_base_impl_default<_Tp, _Sco> volatile* __a, _Tp __pattern, int __order) {
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            return detail::__atomic_fetch_and_cuda(&__a->__a_value, __pattern, __order, detail::__scope_tag<_Sco>());
        ),
        NV_IS_HOST, (
            return host::__atomic_fetch_and(&__a->__a_value, __pattern, __order);
        )
    )
}
template<class _Tp, int _Sco>
__host__ __device__ inline _Tp __cxx_atomic_fetch_or(__cxx_atomic_base_impl_default<_Tp, _Sco> volatile* __a, _Tp __pattern, int __order) {
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            return detail::__atomic_fetch_or_cuda(&__a->__a_value, __pattern, __order, detail::__scope_tag<_Sco>());
        ),
        NV_IS_HOST, (
            return host::__atomic_fetch_or(&__a->__a_value, __pattern, __order);
        )
    )
}
template<class _Tp, int _Sco>
__host__ __device__ inline _Tp __cxx_atomic_fetch_xor(__cxx_atomic_base_impl_default<_Tp, _Sco> volatile* __a, _Tp __pattern, int __order) {
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            return detail::__atomic_fetch_xor_cuda(&__a->__a_value, __pattern, __order, detail::__scope_tag<_Sco>());
        ),
        NV_IS_HOST, (
            return host::__atomic_fetch_xor(&__a->__a_value, __pattern, __order);
        )
    )
}

template <typename _Tp>
using __cxx_small_proxy = typename conditional<sizeof(_Tp) == 1,
                                               uint8_t,
                                               typename conditional<sizeof(_Tp) == 2,
                                                                    uint16_t,
                                                                    void>::type >::type;

template<class _Tp>
__host__ __device__ inline uint32_t __cxx_small_to_32(_Tp __val) {
    __cxx_small_proxy<_Tp> __temp;
    memcpy(&__temp, &__val, sizeof(_Tp));
    return __temp;
}

template<class _Tp>
__host__ __device__ inline _Tp __cxx_small_from_32(uint32_t __val) {
    __cxx_small_proxy<_Tp> __temp = static_cast<__cxx_small_proxy<_Tp>>(__val);
    _Tp __result;
    memcpy(&__result, &__temp, sizeof(_Tp));
    return __result;
}

template<class _Tp, int _Sco>
__host__ __device__ inline void __cxx_atomic_init(__cxx_atomic_base_impl_small<_Tp, _Sco> volatile* __a, _Tp __val) {
    __cxx_atomic_init(&__a->__a_value, __cxx_small_to_32(__val));
}
template<class _Tp, int _Sco>
__host__ __device__ inline void __cxx_atomic_store(__cxx_atomic_base_impl_small<_Tp, _Sco> volatile* __a, _Tp __val, int __order) {
    __cxx_atomic_store(&__a->__a_value, __cxx_small_to_32(__val), __order);
}
template<class _Tp, int _Sco>
__host__ __device__ inline _Tp __cxx_atomic_load(__cxx_atomic_base_impl_small<_Tp, _Sco> const volatile* __a, int __order) {
    return __cxx_small_from_32<_Tp>(__cxx_atomic_load(&__a->__a_value, __order));
}
template<class _Tp, int _Sco>
__host__ __device__ inline _Tp __cxx_atomic_exchange(__cxx_atomic_base_impl_small<_Tp, _Sco> volatile* __a, _Tp __value, int __order) {
    return __cxx_small_from_32<_Tp>(__cxx_atomic_exchange(&__a->__a_value, __cxx_small_to_32(__value), __order));
}
__host__ __device__
inline int __cuda_memcmp(void const * __lhs, void const * __rhs, size_t __count) {
#ifdef __CUDA_ARCH__
    auto __lhs_c = reinterpret_cast<unsigned char const *>(__lhs);
    auto __rhs_c = reinterpret_cast<unsigned char const *>(__rhs);
    while (__count--) {
        auto const __lhs_v = *__lhs_c++;
        auto const __rhs_v = *__rhs_c++;
        if (__lhs_v < __rhs_v) { return -1; }
        if (__lhs_v > __rhs_v) { return 1; }
    }
    return 0;
#else
    return memcmp(__lhs, __rhs, __count);
#endif
}
template<class _Tp, int _Sco>
__host__ __device__ inline bool __cxx_atomic_compare_exchange_weak(__cxx_atomic_base_impl_small<_Tp, _Sco> volatile* __a, _Tp* __expected, _Tp __value, int __success, int __failure) {
    auto __temp = __cxx_small_to_32(*__expected);
    auto const __ret = __cxx_atomic_compare_exchange_weak(&__a->__a_value, &__temp, __cxx_small_to_32(__value), __success, __failure);
    auto const __actual = __cxx_small_from_32<_Tp>(__temp);
    if(!__ret) {
        if(0 == __cuda_memcmp(&__actual, __expected, sizeof(_Tp)))
            __cxx_atomic_fetch_and(&__a->__a_value, (1u << (8*sizeof(_Tp))) - 1, __ATOMIC_RELAXED);
        else
            *__expected = __actual;
    }
    return __ret;
}
template<class _Tp, int _Sco>
__host__ __device__ inline bool __cxx_atomic_compare_exchange_strong(__cxx_atomic_base_impl_small<_Tp, _Sco> volatile* __a, _Tp* __expected, _Tp __value, int __success, int __failure) {
    auto const __old = *__expected;
    while(1) {
        if(__cxx_atomic_compare_exchange_weak(__a, __expected, __value, __success, __failure))
            return true;
        if(0 != __cuda_memcmp(&__old, __expected, sizeof(_Tp)))
            return false;
    }
}
template<class _Tp, int _Sco>
__host__ __device__ inline _Tp __cxx_atomic_fetch_add(__cxx_atomic_base_impl_small<_Tp, _Sco> volatile* __a, _Tp __delta, int __order) {
    return __cxx_small_from_32<_Tp>(__cxx_atomic_fetch_add(&__a->__a_value, __cxx_small_to_32(__delta), __order));
}
template<class _Tp, int _Sco>
__host__ __device__ inline _Tp __cxx_atomic_fetch_sub(__cxx_atomic_base_impl_small<_Tp, _Sco> volatile* __a, _Tp __delta, int __order) {
    return __cxx_small_from_32<_Tp>(__cxx_atomic_fetch_sub(&__a->__a_value, __cxx_small_to_32(__delta), __order));
}
template<class _Tp, int _Sco>
__host__ __device__ inline _Tp __cxx_atomic_fetch_and(__cxx_atomic_base_impl_small<_Tp, _Sco> volatile* __a, _Tp __pattern, int __order) {
    return __cxx_small_from_32<_Tp>(__cxx_atomic_fetch_and(&__a->__a_value, __cxx_small_to_32(__pattern), __order));
}
template<class _Tp, int _Sco>
__host__ __device__ inline _Tp __cxx_atomic_fetch_or(__cxx_atomic_base_impl_small<_Tp, _Sco> volatile* __a, _Tp __pattern, int __order) {
    return __cxx_small_from_32<_Tp>(__cxx_atomic_fetch_or(&__a->__a_value, __cxx_small_to_32(__pattern), __order));
}
template<class _Tp, int _Sco>
__host__ __device__ inline _Tp __cxx_atomic_fetch_xor(__cxx_atomic_base_impl_small<_Tp, _Sco> volatile* __a, _Tp __pattern, int __order) {
    return __cxx_small_from_32<_Tp>(__cxx_atomic_fetch_xor(&__a->__a_value, __cxx_small_to_32(__pattern), __order));
}
