// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ATOMIC_WAIT_NOTIFY_WAIT_H
#define _LIBCUDACXX___ATOMIC_WAIT_NOTIFY_WAIT_H

#include <cuda/std/detail/__config>

#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/__atomic/order.h>

#include <cuda/std/__atomic/operations/heterogeneous.h>
#include <cuda/std/__atomic/wait/polling.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// Leaving this in to figure out if we want this.
// For now this should be dead code, as we don't support platform wait.
#ifdef _LIBCUDACXX_HAS_PLATFORM_WAIT

template <class _Tp, int _Sco, __enable_if_t<!__libcpp_platform_wait_uses_type<_Tp>::__value, int> = 1>
_LIBCUDACXX_INLINE_VISIBILITY void __cxx_atomic_notify_all(__cxx_atomic_impl<_Tp, _Sco> const volatile* __a) {
#ifndef _LIBCUDACXX_HAS_NO_THREAD_CONTENTION_TABLE
    auto * const __c = __libcpp_contention_state(__a);
    __cxx_atomic_fetch_add(__cxx_atomic_rebind<_Sco>(&__c->__version), (__libcpp_platform_wait_t)1, memory_order_relaxed);
    __cxx_atomic_thread_fence(memory_order_seq_cst);
    if (0 != __cxx_atomic_exchange(__cxx_atomic_rebind<_Sco>(&__c->__waiters), (ptrdiff_t)0, memory_order_relaxed))
        __libcpp_platform_wake(&__c->__version, true);
#endif
}
template <class _Tp, int _Sco, __enable_if_t<!__libcpp_platform_wait_uses_type<_Tp>::__value, int> = 1>
_LIBCUDACXX_INLINE_VISIBILITY void __cxx_atomic_notify_one(__cxx_atomic_impl<_Tp, _Sco> const volatile* __a) {
    __cxx_atomic_notify_all(__a);
}
template <class _Ty, class _Tp = __detail::__cxx_atomic_underlying_t<_Ty>, int _Sco = _Ty::__sco, __enable_if_t<!__libcpp_platform_wait_uses_type<_Tp>::__value, int> = 1>
_LIBCUDACXX_INLINE_VISIBILITY void __cxx_atomic_try_wait_slow(_Ty const volatile* __a, _Tp const __val, memory_order __order) {
#ifndef _LIBCUDACXX_HAS_NO_THREAD_CONTENTION_TABLE
    auto * const __c = __libcpp_contention_state(__a);
    __cxx_atomic_store(__cxx_atomic_rebind<_Sco>(&__c->__waiters), (ptrdiff_t)1, memory_order_relaxed);
    __cxx_atomic_thread_fence(memory_order_seq_cst);
    auto const __version = __cxx_atomic_load(__cxx_atomic_rebind<_Sco>(&__c->__version), memory_order_relaxed);
    if (!__cxx_nonatomic_compare_equal(__cxx_atomic_load(__a, __order), __val))
        return;
    if(sizeof(__libcpp_platform_wait_t) < 8) {
        constexpr timespec __timeout = { 2, 0 }; // Hedge on rare 'int version' aliasing.
        __libcpp_platform_wait(&__c->__version, __version, &__timeout);
    }
    else
        __libcpp_platform_wait(&__c->__version, __version, nullptr);
#else
    __cxx_atomic_try_wait_slow_fallback(__a, __val, __order);
#endif // _LIBCUDACXX_HAS_NO_THREAD_CONTENTION_TABLE
}

template <class _Tp, int _Sco, __enable_if_t<__libcpp_platform_wait_uses_type<_Tp>::__value, int> = 1>
_LIBCUDACXX_INLINE_VISIBILITY void __cxx_atomic_try_wait_slow(__cxx_atomic_impl<_Tp, _Sco> const volatile* __a, _Tp __val, memory_order) {
#ifndef _LIBCUDACXX_HAS_NO_THREAD_CONTENTION_TABLE
    auto * const __c = __libcpp_contention_state(__a);
    __cxx_atomic_fetch_add(__cxx_atomic_rebind<_Sco>(&__c->__waiters), (ptrdiff_t)1, memory_order_relaxed);
    __cxx_atomic_thread_fence(memory_order_seq_cst);
#endif
    __libcpp_platform_wait((_Tp*)__a, __val, nullptr);
#ifndef _LIBCUDACXX_HAS_NO_THREAD_CONTENTION_TABLE
    __cxx_atomic_fetch_sub(__cxx_atomic_rebind<_Sco>(&__c->__waiters), (ptrdiff_t)1, memory_order_relaxed);
#endif
}
template <class _Tp, int _Sco, __enable_if_t<__libcpp_platform_wait_uses_type<_Tp>::__value, int> = 1>
_LIBCUDACXX_INLINE_VISIBILITY void __cxx_atomic_notify_all(__cxx_atomic_impl<_Tp, _Sco> const volatile* __a) {
#ifndef _LIBCUDACXX_HAS_NO_THREAD_CONTENTION_TABLE
    auto * const __c = __libcpp_contention_state(__a);
    __cxx_atomic_thread_fence(memory_order_seq_cst);
    if (0 != __cxx_atomic_load(__cxx_atomic_rebind<_Sco>(&__c->__waiters), memory_order_relaxed))
#endif
        __libcpp_platform_wake((_Tp*)__a, true);
}
template <class _Tp, int _Sco, __enable_if_t<__libcpp_platform_wait_uses_type<_Tp>::__value, int> = 1>
_LIBCUDACXX_INLINE_VISIBILITY void __cxx_atomic_notify_one(__cxx_atomic_impl<_Tp, _Sco> const volatile* __a) {
#ifndef _LIBCUDACXX_HAS_NO_THREAD_CONTENTION_TABLE
    auto * const __c = __libcpp_contention_state(__a);
    __cxx_atomic_thread_fence(memory_order_seq_cst);
    if (0 != __cxx_atomic_load(__cxx_atomic_rebind<_Sco>(&__c->__waiters), memory_order_relaxed))
#endif
        __libcpp_platform_wake((_Tp*)__a, false);
}

// Contention table wait/notify is also not supported as above.
#elif !defined(_LIBCUDACXX_HAS_NO_THREAD_CONTENTION_TABLE)

template <class _Tp, int _Sco>
_LIBCUDACXX_INLINE_VISIBILITY void __cxx_atomic_notify_all(__cxx_atomic_impl<_Tp, _Sco> const volatile* __a) {
    auto * const __c = __libcpp_contention_state(__a);
    __cxx_atomic_thread_fence(memory_order_seq_cst);
    if(0 == __cxx_atomic_load(__cxx_atomic_rebind<_Sco>(&__c->__credit), memory_order_relaxed))
        return;
    if(0 != __cxx_atomic_exchange(__cxx_atomic_rebind<_Sco>(&__c->__credit), (ptrdiff_t)0, memory_order_relaxed)) {
        __libcpp_mutex_lock(&__c->__mutex);
        __libcpp_mutex_unlock(&__c->__mutex);
        __libcpp_condvar_broadcast(&__c->__condvar);
    }
}
template <class _Tp, int _Sco>
_LIBCUDACXX_INLINE_VISIBILITY void __cxx_atomic_notify_one(__cxx_atomic_impl<_Tp, _Sco> const volatile* __a) {
    __cxx_atomic_notify_all(__a);
}
template <class _Tp, int _Sco>
_LIBCUDACXX_INLINE_VISIBILITY void __cxx_atomic_try_wait_slow(__cxx_atomic_impl<_Tp, _Sco> const volatile* __a, _Tp const __val, memory_order __order) {
    auto * const __c = __libcpp_contention_state(__a);
    __libcpp_mutex_lock(&__c->__mutex);
    __cxx_atomic_store(__cxx_atomic_rebind<_Sco>(&__c->__credit), (ptrdiff_t)1, memory_order_relaxed);
    __cxx_atomic_thread_fence(memory_order_seq_cst);
    if (__cxx_nonatomic_compare_equal(__cxx_atomic_load(__a, __order), __val))
        __libcpp_condvar_wait(&__c->__condvar, &__c->__mutex);
    __libcpp_mutex_unlock(&__c->__mutex);
}

#else

// Heterogeneous atomic impl begins here
extern "C" _CCCL_DEVICE void __atomic_try_wait_unsupported_before_SM_70__();

template <typename _Tp, typename _Sco>
_LIBCUDACXX_INLINE_VISIBILITY void __atomic_try_wait_slow(_Tp const volatile* __a, __atomic_underlying_t<_Tp> __val, memory_order __order, _Sco) {
    NV_DISPATCH_TARGET(
        NV_PROVIDES_SM_70,
            __atomic_try_wait_slow_fallback(__a, __val, __order, _Sco{});,
        NV_IS_HOST,
            __atomic_try_wait_slow_fallback(__a, __val, __order, _Sco{});,
        NV_ANY_TARGET,
            __atomic_try_wait_unsupported_before_SM_70__();
    );
}

template <typename _Tp, typename _Sco>
_LIBCUDACXX_INLINE_VISIBILITY void __atomic_notify_one(_Tp const volatile*, _Sco) {
    NV_DISPATCH_TARGET(
        NV_PROVIDES_SM_70,,
        NV_IS_HOST,,
        NV_ANY_TARGET,
            __atomic_try_wait_unsupported_before_SM_70__();
    );
}

template <typename _Tp, typename _Sco>
_LIBCUDACXX_INLINE_VISIBILITY void __atomic_notify_all(_Tp const volatile*, _Sco) {
    NV_DISPATCH_TARGET(
        NV_PROVIDES_SM_70,,
        NV_IS_HOST,,
        NV_ANY_TARGET,
            __atomic_try_wait_unsupported_before_SM_70__();
    );
}

#endif // _LIBCUDACXX_HAS_PLATFORM_WAIT || !defined(_LIBCUDACXX_HAS_NO_THREAD_CONTENTION_TABLE)

template <typename _Tp> _LIBCUDACXX_INLINE_VISIBILITY
bool __nonatomic_compare_equal(_Tp const& __lhs, _Tp const& __rhs) {
#if defined(_CCCL_CUDA_COMPILER)
    return __lhs == __rhs;
#else
    return memcmp(&__lhs, &__rhs, sizeof(_Tp)) == 0;
#endif
}

template <typename _Tp, typename _Sco>
_LIBCUDACXX_INLINE_VISIBILITY void __atomic_wait(_Tp const volatile* __a, __atomic_underlying_t<_Tp> const __val, memory_order __order, _Sco = {}) {
    for(int __i = 0; __i < _LIBCUDACXX_POLLING_COUNT; ++__i) {
        if(!__nonatomic_compare_equal(__atomic_load_dispatch(*__a, __order, _Sco{}, __atomic_tag_t<_Tp>{}), __val))
            return;
        if(__i < 12)
            __libcpp_thread_yield_processor();
        else
            __libcpp_thread_yield();
    }
    while(__nonatomic_compare_equal(__atomic_load_dispatch(*__a, __order, _Sco{}, __atomic_tag_t<_Tp>{}), __val))
        __atomic_try_wait_slow(__a, __val, __order, _Sco{});
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif _LIBCUDACXX___ATOMIC_WAIT_NOTIFY_WAIT_H
