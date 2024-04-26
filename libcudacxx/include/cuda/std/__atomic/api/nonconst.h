//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __LIBCUDACXX___ATOMIC_API_NONCONST_H
#define __LIBCUDACXX___ATOMIC_API_NONCONST_H

#include <cuda/std/detail/__config>

#include <cuda/std/__atomic/api/atomic_crtp.h>

#include <cuda/std/__atomic/wait/polling.h>
#include <cuda/std/__atomic/wait/notify_wait.h>

#include <cuda/std/__atomic/order.h>
#include <cuda/std/__atomic/scopes.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <typename _Tp, typename _Crtp, typename _Sco = __thread_scope_system_tag>
struct __atomic_common : public _Crtp {
#if defined(_LIBCUDACXX_ATOMIC_ALWAYS_LOCK_FREE)
    static constexpr bool is_always_lock_free = _LIBCUDACXX_ATOMIC_ALWAYS_LOCK_FREE(sizeof(_Tp), 0);
#endif // defined(_LIBCUDACXX_ATOMIC_ALWAYS_LOCK_FREE)

    _CCCL_HOST_DEVICE inline
    bool is_lock_free() const volatile noexcept
        {return _LIBCUDACXX_ATOMIC_IS_LOCK_FREE(sizeof(_Tp));}
    _CCCL_HOST_DEVICE inline
    bool is_lock_free() const noexcept
        {return _LIBCUDACXX_ATOMIC_IS_LOCK_FREE(sizeof(_Tp));}

    _CCCL_HOST_DEVICE inline
    void store(_Tp __d, memory_order __m = memory_order_seq_cst) volatile noexcept
      _LIBCUDACXX_CHECK_STORE_MEMORY_ORDER(__m)
        {__atomic_store_dispatch(this->__this_atom(), __d, __m, _Sco{});}
    _CCCL_HOST_DEVICE inline
    void store(_Tp __d, memory_order __m = memory_order_seq_cst) noexcept
      _LIBCUDACXX_CHECK_STORE_MEMORY_ORDER(__m)
        {__atomic_store_dispatch(this->__this_atom(), __d, __m, _Sco{});}

    _CCCL_HOST_DEVICE inline
    _Tp load(memory_order __m = memory_order_seq_cst) const volatile noexcept
      _LIBCUDACXX_CHECK_LOAD_MEMORY_ORDER(__m)
        {return __atomic_load_dispatch(this->__this_atom(), __m, _Sco{});}
    _CCCL_HOST_DEVICE inline
    _Tp load(memory_order __m = memory_order_seq_cst) const noexcept
      _LIBCUDACXX_CHECK_LOAD_MEMORY_ORDER(__m)
        {return __atomic_load_dispatch(this->__this_atom(), __m, _Sco{});}

    _CCCL_HOST_DEVICE inline
    operator _Tp() const volatile noexcept {return load();}
    _CCCL_HOST_DEVICE inline
    operator _Tp() const noexcept          {return load();}

    _CCCL_HOST_DEVICE inline
    _Tp exchange(_Tp __d, memory_order __m = memory_order_seq_cst) volatile noexcept
        {return __atomic_exchange_dispatch(this->__this_atom(), __d, __m, _Sco{});}
    _CCCL_HOST_DEVICE inline
    _Tp exchange(_Tp __d, memory_order __m = memory_order_seq_cst) noexcept
        {return __atomic_exchange_dispatch(this->__this_atom(), __d, __m, _Sco{});}

    _CCCL_HOST_DEVICE inline
    bool compare_exchange_weak(_Tp& __e, _Tp __d,
                               memory_order __s, memory_order __f) volatile noexcept
      _LIBCUDACXX_CHECK_EXCHANGE_MEMORY_ORDER(__s, __f)
        {return __atomic_compare_exchange_weak_dispatch(this->__this_atom(), &__e, __d, __s, __f, _Sco{});}
    _CCCL_HOST_DEVICE inline
    bool compare_exchange_weak(_Tp& __e, _Tp __d,
                               memory_order __s, memory_order __f) noexcept
      _LIBCUDACXX_CHECK_EXCHANGE_MEMORY_ORDER(__s, __f)
        {return __atomic_compare_exchange_weak_dispatch(this->__this_atom(), &__e, __d, __s, __f, _Sco{});}

    _CCCL_HOST_DEVICE inline
    bool compare_exchange_strong(_Tp& __e, _Tp __d,
                                 memory_order __s, memory_order __f) volatile noexcept
      _LIBCUDACXX_CHECK_EXCHANGE_MEMORY_ORDER(__s, __f)
        {return __atomic_compare_exchange_strong_dispatch(this->__this_atom(), &__e, __d, __s, __f, _Sco{});}
    _CCCL_HOST_DEVICE inline
    bool compare_exchange_strong(_Tp& __e, _Tp __d,
                                 memory_order __s, memory_order __f) noexcept
      _LIBCUDACXX_CHECK_EXCHANGE_MEMORY_ORDER(__s, __f)
        {return __atomic_compare_exchange_strong_dispatch(this->__this_atom(), &__e, __d, __s, __f, _Sco{});}

    _CCCL_HOST_DEVICE inline
    bool compare_exchange_weak(_Tp& __e, _Tp __d,
                              memory_order __m = memory_order_seq_cst) volatile noexcept {
        if (memory_order_acq_rel == __m)
            return __atomic_compare_exchange_weak_dispatch(this->__this_atom(), &__e, __d, __m, memory_order_acquire, _Sco{});
        else if (memory_order_release == __m)
            return __atomic_compare_exchange_weak_dispatch(this->__this_atom(), &__e, __d, __m, memory_order_relaxed, _Sco{});
        else
            return __atomic_compare_exchange_weak_dispatch(this->__this_atom(), &__e, __d, __m, __m, _Sco{});
    }
    _CCCL_HOST_DEVICE inline
    bool compare_exchange_weak(_Tp& __e, _Tp __d,
                               memory_order __m = memory_order_seq_cst) noexcept {
        if(memory_order_acq_rel == __m)
            return __atomic_compare_exchange_weak_dispatch(this->__this_atom(), &__e, __d, __m, memory_order_acquire, _Sco{});
        else if(memory_order_release == __m)
            return __atomic_compare_exchange_weak_dispatch(this->__this_atom(), &__e, __d, __m, memory_order_relaxed, _Sco{});
        else
            return __atomic_compare_exchange_weak_dispatch(this->__this_atom(), &__e, __d, __m, __m, _Sco{});
    }

    _CCCL_HOST_DEVICE inline
    bool compare_exchange_strong(_Tp& __e, _Tp __d,
                              memory_order __m = memory_order_seq_cst) volatile noexcept {
        if (memory_order_acq_rel == __m)
            return __atomic_compare_exchange_strong_dispatch(this->__this_atom(), &__e, __d, __m, memory_order_acquire, _Sco{});
        else if (memory_order_release == __m)
            return __atomic_compare_exchange_strong_dispatch(this->__this_atom(), &__e, __d, __m, memory_order_relaxed, _Sco{});
        else
            return __atomic_compare_exchange_strong_dispatch(this->__this_atom(), &__e, __d, __m, __m, _Sco{});
    }
    _CCCL_HOST_DEVICE inline
    bool compare_exchange_strong(_Tp& __e, _Tp __d,
                                 memory_order __m = memory_order_seq_cst) noexcept {
        if (memory_order_acq_rel == __m)
            return __atomic_compare_exchange_strong_dispatch(this->__this_atom(), &__e, __d, __m, memory_order_acquire, _Sco{});
        else if (memory_order_release == __m)
            return __atomic_compare_exchange_strong_dispatch(this->__this_atom(), &__e, __d, __m, memory_order_relaxed, _Sco{});
        else
            return __atomic_compare_exchange_strong_dispatch(this->__this_atom(), &__e, __d, __m, __m, _Sco{});
    }

    _CCCL_HOST_DEVICE inline void wait(_Tp __v, memory_order __m = memory_order_seq_cst) const volatile noexcept
        {__atomic_wait(this->__this_atom(), __v, __m, _Sco{});}
    _CCCL_HOST_DEVICE inline void wait(_Tp __v, memory_order __m = memory_order_seq_cst) const noexcept
        {__atomic_wait(this->__this_atom(), __v, __m, _Sco{});}

    _CCCL_HOST_DEVICE inline void notify_one() volatile noexcept
        {__atomic_notify_one(this->__this_atom(), _Sco{});}
    _CCCL_HOST_DEVICE inline void notify_one() noexcept
        {__atomic_notify_one(this->__this_atom(), _Sco{});}

    _CCCL_HOST_DEVICE inline void notify_all() volatile noexcept
        {__atomic_notify_all(this->__this_atom(), _Sco{});}
    _CCCL_HOST_DEVICE inline void notify_all() noexcept
        {__atomic_notify_all(this->__this_atom(), _Sco{});}
};

template <typename _Tp, typename _Crtp, typename _Sco = __thread_scope_system_tag>
struct __atomic_arithmetic : public __atomic_common<_Tp, _Crtp, _Sco> {
    _CCCL_HOST_DEVICE inline
    _Tp fetch_add(_Tp __op, memory_order __m = memory_order_seq_cst) volatile noexcept
        {return __atomic_fetch_add_dispatch(this->__this_atom(), __op, __m, _Sco{});}
    _CCCL_HOST_DEVICE inline
    _Tp fetch_add(_Tp __op, memory_order __m = memory_order_seq_cst) noexcept
        {return __atomic_fetch_add_dispatch(this->__this_atom(), __op, __m, _Sco{});}

    _CCCL_HOST_DEVICE inline
    _Tp fetch_sub(_Tp __op, memory_order __m = memory_order_seq_cst) volatile noexcept
        {return __atomic_fetch_sub_dispatch(this->__this_atom(), __op, __m, _Sco{});}
    _CCCL_HOST_DEVICE inline
    _Tp fetch_sub(_Tp __op, memory_order __m = memory_order_seq_cst) noexcept
        {return __atomic_fetch_sub_dispatch(this->__this_atom(), __op, __m, _Sco{});}

    _CCCL_HOST_DEVICE inline
    _Tp operator++(int) volatile noexcept      {return fetch_add(_Tp(1));}
    _CCCL_HOST_DEVICE inline
    _Tp operator++(int) noexcept               {return fetch_add(_Tp(1));}

    _CCCL_HOST_DEVICE inline
    _Tp operator--(int) volatile noexcept      {return fetch_sub(_Tp(1));}
    _CCCL_HOST_DEVICE inline
    _Tp operator--(int) noexcept               {return fetch_sub(_Tp(1));}

    _CCCL_HOST_DEVICE inline
    _Tp operator++() volatile noexcept         {return fetch_add(_Tp(1)) + _Tp(1);}
    _CCCL_HOST_DEVICE inline
    _Tp operator++() noexcept                  {return fetch_add(_Tp(1)) + _Tp(1);}

    _CCCL_HOST_DEVICE inline
    _Tp operator--() volatile noexcept         {return fetch_sub(_Tp(1)) - _Tp(1);}
    _CCCL_HOST_DEVICE inline
    _Tp operator--() noexcept                  {return fetch_sub(_Tp(1)) - _Tp(1);}

    _CCCL_HOST_DEVICE inline
    _Tp operator+=(_Tp __op) volatile noexcept {return fetch_add(__op) + __op;}
    _CCCL_HOST_DEVICE inline
    _Tp operator+=(_Tp __op) noexcept          {return fetch_add(__op) + __op;}

    _CCCL_HOST_DEVICE inline
    _Tp operator-=(_Tp __op) volatile noexcept {return fetch_sub(__op) - __op;}
    _CCCL_HOST_DEVICE inline
    _Tp operator-=(_Tp __op) noexcept          {return fetch_sub(__op) - __op;}
};

template <typename _Tp, typename _Crtp, typename _Sco = __thread_scope_system_tag>
struct __atomic_bitwise : public __atomic_arithmetic<_Tp, _Crtp, _Sco> {
    _CCCL_HOST_DEVICE inline
    _Tp fetch_and(_Tp __op, memory_order __m = memory_order_seq_cst) volatile noexcept
        {return __atomic_fetch_and_dispatch(this->__this_atom(), __op, __m, _Sco{});}
    _CCCL_HOST_DEVICE inline
    _Tp fetch_and(_Tp __op, memory_order __m = memory_order_seq_cst) noexcept
        {return __atomic_fetch_and_dispatch(this->__this_atom(), __op, __m, _Sco{});}

    _CCCL_HOST_DEVICE inline
    _Tp fetch_or(_Tp __op, memory_order __m = memory_order_seq_cst) volatile noexcept
        {return __atomic_fetch_or_dispatch(this->__this_atom(), __op, __m, _Sco{});}
    _CCCL_HOST_DEVICE inline
    _Tp fetch_or(_Tp __op, memory_order __m = memory_order_seq_cst) noexcept
        {return __atomic_fetch_or_dispatch(this->__this_atom(), __op, __m, _Sco{});}

    _CCCL_HOST_DEVICE inline
    _Tp fetch_xor(_Tp __op, memory_order __m = memory_order_seq_cst) volatile noexcept
        {return __atomic_fetch_xor_dispatch(this->__this_atom(), __op, __m, _Sco{});}
    _CCCL_HOST_DEVICE inline
    _Tp fetch_xor(_Tp __op, memory_order __m = memory_order_seq_cst) noexcept
        {return __atomic_fetch_xor_dispatch(this->__this_atom(), __op, __m, _Sco{});}

    _CCCL_HOST_DEVICE inline
    _Tp operator&=(_Tp __op) volatile noexcept {return fetch_and(__op) & __op;}
    _CCCL_HOST_DEVICE inline
    _Tp operator&=(_Tp __op) noexcept          {return fetch_and(__op) & __op;}

    _CCCL_HOST_DEVICE inline
    _Tp operator|=(_Tp __op) volatile noexcept {return fetch_or(__op) | __op;}
    _CCCL_HOST_DEVICE inline
    _Tp operator|=(_Tp __op) noexcept          {return fetch_or(__op) | __op;}

    _CCCL_HOST_DEVICE inline
    _Tp operator^=(_Tp __op) volatile noexcept {return fetch_xor(__op) ^ __op;}
    _CCCL_HOST_DEVICE inline
    _Tp operator^=(_Tp __op) noexcept          {return fetch_xor(__op) ^ __op;}
};

_LIBCUDACXX_END_NAMESPACE_STD

#endif __LIBCUDACXX___ATOMIC_API_NONCONST_H
