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

#ifndef _LIBCUDACXX___ATOMIC_STORAGE_SMALL_H
#define _LIBCUDACXX___ATOMIC_STORAGE_SMALL_H

#include <cuda/std/detail/__config>

#include <cuda/std/type_traits>

#include <cuda/std/__atomic/storage/base.h>
#include <cuda/std/__atomic/order.h>
#include <cuda/std/__atomic/scopes.h>

#include <cuda/std/__atomic/operations/heterogeneous.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// Atomic small types require conversion to/from a proxy type that can be
// manipulated by PTX without any performance overhead
struct __atomic_small_tag {};

template <typename _Tp>
using __atomic_small_proxy_t = __conditional_t<is_signed<_Tp>::value, int32_t, uint32_t>;

// Arithmetic conversions to/from proxy types
template<class _Tp, __enable_if_t<is_arithmetic<_Tp>::value, int> = 0>
constexpr _CCCL_HOST_DEVICE inline __atomic_small_proxy_t<_Tp> __atomic_small_to_32(_Tp __val) {
    return static_cast<__atomic_small_proxy_t<_Tp>>(__val);
}

template<class _Tp, __enable_if_t<is_arithmetic<_Tp>::value, int> = 0>
constexpr _CCCL_HOST_DEVICE inline _Tp __atomic_small_from_32(__atomic_small_proxy_t<_Tp> __val) {
    return static_cast<_Tp>(__val);
}

// Non-arithmetic conversion to/from proxy types
template<class _Tp, __enable_if_t<!is_arithmetic<_Tp>::value, int> = 0>
_CCCL_HOST_DEVICE inline __atomic_small_proxy_t<_Tp> __atomic_small_to_32(_Tp __val) {
    __atomic_small_proxy_t<_Tp> __temp{};
    memcpy(&__temp, &__val, sizeof(_Tp));
    return __temp;
}

template<class _Tp, __enable_if_t<!is_arithmetic<_Tp>::value, int> = 0>
_CCCL_HOST_DEVICE inline _Tp __atomic_small_from_32(__atomic_small_proxy_t<_Tp> __val) {
    _Tp __temp{};
    memcpy(&__temp, &__val, sizeof(_Tp));
    return __temp;
}

template <typename _Tp>
struct __atomic_small_storage {
    using __underlying_t = _Tp;
    using __tag_t = __atomic_small_tag;
    using __proxy_t = __atomic_small_proxy_t<_Tp>;

    __atomic_small_storage() noexcept = default;

    _CCCL_HOST_DEVICE
    constexpr explicit __atomic_small_storage(_Tp __value) : __a_value(__atomic_small_to_32(__value)) {}

    __atomic_storage<__proxy_t> __a_value;
};

template <typename _Tp, typename _Sco>
_CCCL_HOST_DEVICE
void __atomic_init_dispatch(_Tp& __a, __atomic_underlying_t<_Tp> __val, _Sco, __atomic_small_tag) {
    __atomic_init_dispatch(__a.__a_value, __atomic_small_to_32(__val), _Sco{});
}

template <typename _Tp, typename _Sco>
_CCCL_HOST_DEVICE inline void __atomic_store_dispatch(_Tp& __a, __atomic_underlying_t<_Tp> __val, memory_order __order, _Sco, __atomic_small_tag) {
    __atomic_store_dispatch(__a.__a_value, __atomic_small_to_32(__val), __order, _Sco{});
}

template <typename _Tp, typename _Sco>
_CCCL_HOST_DEVICE inline _Tp __atomic_load_dispatch(_Tp const& __a, memory_order __order, _Sco, __atomic_small_tag) {
    return __atomic_small_from_32<_Tp>(__atomic_load_dispatch(__a.__a_value, __order, _Sco{}));
}

template <typename _Tp, typename _Sco>
_CCCL_HOST_DEVICE inline _Tp __atomic_exchange_dispatch(_Tp& __a, __atomic_underlying_t<_Tp> __value, memory_order __order, _Sco, __atomic_small_tag) {
    return __atomic_small_from_32<_Tp>(__atomic_exchange_dispatch(__a.__a_value, __atomic_small_to_32(__value), __order, _Sco{}));
}
_CCCL_HOST_DEVICE
inline int __cuda_memcmp(void const * __lhs, void const * __rhs, size_t __count) {
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            auto __lhs_c = reinterpret_cast<unsigned char const *>(__lhs);
            auto __rhs_c = reinterpret_cast<unsigned char const *>(__rhs);
            while (__count--) {
                auto const __lhs_v = *__lhs_c++;
                auto const __rhs_v = *__rhs_c++;
                if (__lhs_v < __rhs_v) { return -1; }
                if (__lhs_v > __rhs_v) { return 1; }
            }
            return 0;
        ),
        NV_IS_HOST, (
            return memcmp(__lhs, __rhs, __count);
        )
    )
}

template <typename _Tp, typename _Sco>
_CCCL_HOST_DEVICE inline bool __atomic_compare_exchange_weak_dispatch(_Tp& __a, __atomic_underlying_t<_Tp>* __expected, __atomic_underlying_t<_Tp> __value, memory_order __success, memory_order __failure, _Sco, __atomic_small_tag) {
    auto __temp_expected = __atomic_small_to_32(*__expected);
    auto const __ret = __atomic_compare_exchange_weak_dispatch(__a.__a_value, &__temp_expected, __atomic_small_to_32(__value), __success, __failure, _Sco{});
    auto const __actual = __atomic_small_from_32<__atomic_underlying_t<_Tp>>(__temp_expected);
    constexpr auto __mask = static_cast<decltype(__temp_expected)>((1u << (8*sizeof(__atomic_underlying_t<_Tp>))) - 1);
    if(!__ret) {
        if(0 == __cuda_memcmp(&__actual, __expected, sizeof(__atomic_underlying_t<_Tp>)))
            __atomic_fetch_and_dispatch(__a.__a_value, __mask, memory_order_relaxed, _Sco{});
        else
            *__expected = __actual;
    }
    return __ret;
}

template <typename _Tp, typename _Sco>
_CCCL_HOST_DEVICE inline bool __atomic_compare_exchange_strong_dispatch(_Tp& __a, __atomic_underlying_t<_Tp>* __expected, __atomic_underlying_t<_Tp> __value, memory_order __success, memory_order __failure, _Sco, __atomic_small_tag) {
    auto const __old = *__expected;
    while(1) {
        if(__atomic_compare_exchange_weak_dispatch(__a, __expected, __value, __success, __failure, _Sco{}, __atomic_small_tag{}))
            return true;
        if(0 != __cuda_memcmp(&__old, __expected, sizeof(__atomic_underlying_t<_Tp>)))
            return false;
    }
}

template <typename _Tp, typename _Sco>
_CCCL_HOST_DEVICE inline _Tp __atomic_fetch_add_dispatch(_Tp& __a, __atomic_underlying_t<_Tp> __delta, memory_order __order, _Sco, __atomic_small_tag) {
    return __atomic_small_from_32<_Tp>(__atomic_fetch_add_dispatch(__a.__a_value, __atomic_small_to_32(__delta), __order, _Sco{}));
}

template <typename _Tp, typename _Sco>
_CCCL_HOST_DEVICE inline _Tp __atomic_fetch_sub_dispatch(_Tp& __a, __atomic_underlying_t<_Tp> __delta, memory_order __order, _Sco, __atomic_small_tag) {
    return __atomic_small_from_32<_Tp>(__atomic_fetch_sub_dispatch(__a.__a_value, __atomic_small_to_32(__delta), __order, _Sco{}));
}

template <typename _Tp, typename _Sco>
_CCCL_HOST_DEVICE inline _Tp __atomic_fetch_and_dispatch(_Tp& __a, __atomic_underlying_t<_Tp> __pattern, memory_order __order, _Sco, __atomic_small_tag) {
    return __atomic_small_from_32<_Tp>(__atomic_fetch_and_dispatch(__a.__a_value, __atomic_small_to_32(__pattern), __order, _Sco{}));
}

template <typename _Tp, typename _Sco>
_CCCL_HOST_DEVICE inline _Tp __atomic_fetch_or_dispatch(_Tp& __a, __atomic_underlying_t<_Tp> __pattern, memory_order __order, _Sco, __atomic_small_tag) {
    return __atomic_small_from_32<_Tp>(__atomic_fetch_or_dispatch(__a.__a_value, __atomic_small_to_32(__pattern), __order, _Sco{}));
}

template <typename _Tp, typename _Sco>
_CCCL_HOST_DEVICE inline _Tp __atomic_fetch_xor_dispatch(_Tp& __a, __atomic_underlying_t<_Tp> __pattern, memory_order __order, _Sco, __atomic_small_tag) {
    return __atomic_small_from_32<_Tp>(__atomic_fetch_xor_dispatch(__a.__a_value, __atomic_small_to_32(__pattern), __order, _Sco{}));
}

template <typename _Tp, typename _Delta, typename _Sco>
_CCCL_HOST_DEVICE inline _Tp __atomic_fetch_max_dispatch(_Tp& __a, __atomic_underlying_t<_Tp> __val, memory_order __order, _Sco, __atomic_small_tag) {
    return __atomic_small_from_32<_Tp>(__atomic_fetch_max_dispatch(__a.__a_value, __atomic_small_to_32(__val), __order, _Sco{}));
}

template <typename _Tp, typename _Delta, typename _Sco>
_CCCL_HOST_DEVICE inline _Tp __atomic_fetch_min_dispatch(_Tp& __a, __atomic_underlying_t<_Tp> __val, memory_order __order, _Sco, __atomic_small_tag) {
    return __atomic_small_from_32<_Tp>(__atomic_fetch_min_dispatch(__a.__a_value, __atomic_small_to_32(__val), __order, _Sco{}));
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ATOMIC_STORAGE_SMALL_H
