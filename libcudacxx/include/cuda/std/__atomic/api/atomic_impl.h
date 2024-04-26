//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __LIBCUDACXX___ATOMIC_API_ATOMIC_IMPL_H
#define __LIBCUDACXX___ATOMIC_API_ATOMIC_IMPL_H

#include <cuda/std/detail/__config>

#include <cuda/std/__atomic/api/const.h>
#include <cuda/std/__atomic/api/nonconst.h>

#include <cuda/std/__atomic/storage/base.h>
#include <cuda/std/__atomic/storage/reference.h>
#include <cuda/std/__atomic/storage/small.h>
#include <cuda/std/__atomic/storage/locked.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <typename _Tp>
struct __atomic_traits {
    static constexpr bool __atomic_requires_lock = !__atomic_is_always_lock_free<_Tp>::__value;
    static constexpr bool __atomic_requires_small = sizeof(_Tp) < 4;
    static constexpr bool __atomic_supports_reference = __atomic_is_always_lock_free<_Tp>::__value && (sizeof(_Tp) >= 4 && sizeof(_Tp) <= 8);
};

template <typename _Tp>
using __atomic_get_storage_t = typename __conditional_t<__atomic_traits<_Tp>::__atomic_requires_small,
                                            __atomic_small_storage<_Tp>,
                                            __conditional_t<__atomic_traits<_Tp>::__atomic_requires_lock,
                                                __atomic_locked_storage<_Tp>,
                                                __atomic_storage<_Tp>
                                                >>;

template <typename _Tp, typename _Crtp, typename _Sco = __thread_scope_system_tag>
using __atomic_impl_t = __conditional_t<is_floating_point<_Tp>::value,
                                            __atomic_arithmetic<_Tp, _Crtp, _Sco>,
                                            __conditional_t<is_integral<_Tp>::value,
                                                __atomic_bitwise<_Tp, _Crtp, _Sco>,
                                                __atomic_common<_Tp, _Crtp, _Sco> >>;

template <typename _Tp, typename _Crtp, typename _Sco = __thread_scope_system_tag>
using __atomic_const_impl_t = __conditional_t<is_floating_point<_Tp>::value,
                                            __atomic_arithmetic_const<_Tp, _Crtp, _Sco>,
                                            __conditional_t<is_integral<_Tp>::value,
                                                __atomic_bitwise_const<_Tp, _Crtp, _Sco>,
                                                __atomic_common_const<_Tp, _Crtp, _Sco> >>;


template <typename _Tp, thread_scope _Sco>
struct __atomic_impl :
    public __atomic_impl_t<_Tp, __atomic_crtp_accessor<__atomic_impl<_Tp,_Sco>, __atomic_get_storage_t<_Tp>>, __scope_to_tag<_Sco>> {

    using __storage = __atomic_get_storage_t<_Tp>;
    __storage __a;

    _CCCL_HOST_DEVICE constexpr inline
    __atomic_impl(_Tp __v) noexcept : __a(__v) {}

    _CCCL_HOST_DEVICE constexpr inline
    __storage* __get_atom() {
        return &__a;
    }
    _CCCL_HOST_DEVICE constexpr inline
    const __storage* __get_atom() const {
        return &__a;
    }
    _CCCL_HOST_DEVICE constexpr inline
    volatile __storage* __get_atom() volatile {
        return &__a;
    }
    _CCCL_HOST_DEVICE constexpr inline
    const volatile __storage* __get_atom() const volatile {
        return &__a;
    }

    constexpr inline
    __atomic_impl() noexcept = default;
    constexpr inline
    __atomic_impl(const __atomic_impl&) = delete;
    constexpr inline
    __atomic_impl(__atomic_impl&&) = delete;

    constexpr inline
    __atomic_impl& operator=(const __atomic_impl&) = delete;
    constexpr inline
    __atomic_impl& operator=(__atomic_impl&&) = delete;
};

template <typename _Tp, thread_scope _Sco>
struct __atomic_ref_impl :
    public __atomic_const_impl_t<_Tp, __atomic_crtp_accessor<__atomic_ref_impl<_Tp,_Sco>, __atomic_ref_storage<_Tp>>, __scope_to_tag<_Sco>> {

    using __storage = __atomic_ref_storage<_Tp>;
    __storage __a;

    _CCCL_HOST_DEVICE constexpr inline
    __storage* __get_atom() {
        return &__a;
    }
    _CCCL_HOST_DEVICE constexpr inline
    const __storage* __get_atom() const {
        return &__a;
    }
    _CCCL_HOST_DEVICE constexpr inline
    volatile __storage* __get_atom() volatile {
        return &__a;
    }
    _CCCL_HOST_DEVICE constexpr inline
    const volatile __storage* __get_atom() const volatile {
        return &__a;
    }

    _CCCL_HOST_DEVICE constexpr inline
    __atomic_ref_impl(_Tp& __v) : __a(&__v) {}

    constexpr inline
    __atomic_ref_impl() = delete;
    constexpr inline
    __atomic_ref_impl(const __atomic_ref_impl&) noexcept = default;
    constexpr inline
    __atomic_ref_impl(__atomic_ref_impl&&) = delete;

    constexpr inline
    __atomic_ref_impl& operator=(const __atomic_ref_impl&) = delete;
    constexpr inline
    __atomic_ref_impl& operator=(__atomic_ref_impl&&) = delete;
};

_LIBCUDACXX_END_NAMESPACE_STD

#endif __LIBCUDACXX___ATOMIC_API_ATOMIC_IMPL_H
