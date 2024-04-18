//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __LIBCUDACXX___ATOMIC_DISPATCH_H
#define __LIBCUDACXX___ATOMIC_DISPATCH_H

#include <cuda/std/detail/__config>

#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/__atomic/order.h>
#include <cuda/std/__atomic/storage/common.h>

#include <cuda/std/__atomic/operations/host.h>
#include <cuda/std/__atomic/operations/atomic_cuda_ptx_generated.h>
#include <cuda/std/__atomic/operations/atomic_cuda_ptx_derived.h>

// Dispatch directly calls PTX/Host backends for atomic objects.
// By default these objects support extracting the address contained with operator()()
// this provides some amount of syntactic sugar to avoid duplicating every function that requires `volatile`.
// `_Tp` is able to be volatile and will simply be instatiated into a new function.
// It is up to the underlying backends to implement the correct volatile behavior

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_HOST_DEVICE
inline
 void __atomic_thread_fence_dispatch(memory_order __order) {
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            __atomic_thread_fence_cuda(static_cast<__memory_order_underlying_t>(__order), __thread_scope_system_tag());
        ),
        NV_IS_HOST, (
            __atomic_thread_fence_host(__order);
        )
    )
}

_CCCL_HOST_DEVICE
inline
 void __atomic_signal_fence_dispatch(memory_order __order) {
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            __atomic_signal_fence_cuda(static_cast<__memory_order_underlying_t>(__order));
        ),
        NV_IS_HOST, (
            __atomic_signal_fence_host(__order);
        )
    )
}

// Regarding __atomic_base_Tag
// It *is* possible to define it as:
// _Tag = __atomic_enable_if_default_base_t<_Tp> and make all tag types default to the 'base' backend
// I don't know if it's necessary to do that though. For now, this just adds some kind of protection
// preventing access to the functions with the wrong tag type.
template <typename _Tp>
using __atomic_enable_if_default_base_t = __enable_if_t<is_same<__atomic_tag_t<_Tp>, __atomic_base_tag>::value, __atomic_tag_t<_Tp>>;

template <typename _Tp, typename _Tag = __atomic_enable_if_default_base_t<_Tp>>
_CCCL_HOST_DEVICE
 void __atomic_init_dispatch(_Tp& __a, __atomic_underlying_t<_Tp> __val, _Tag = {}) {
    __atomic_assign_volatile(__a.get(), __val);
}

template <typename _Tp, typename _Sco = __thread_scope_system_tag, typename _Tag = __atomic_enable_if_default_base_t<_Tp>>
_CCCL_HOST_DEVICE
 void __atomic_store_dispatch(_Tp& __a, __atomic_underlying_t<_Tp> __val, memory_order __order, _Sco = {}, _Tag = {}) {
    alignas(_Tp) auto __tmp = __val;
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            __atomic_store_n_cuda(__a.get(), __val, static_cast<__memory_order_underlying_t>(__order),  _Sco{});
        ),
        NV_IS_HOST, (
            __atomic_store_host(__a.get(), __val, __order);
        )
    )
}

template <typename _Tp, typename _Sco = __thread_scope_system_tag, typename _Tag = __atomic_enable_if_default_base_t<_Tp>>
_CCCL_HOST_DEVICE
 auto __atomic_load_dispatch(_Tp const& __a, memory_order __order, _Sco = {}, _Tag = {}) -> __atomic_underlying_t<_Tp> {
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            return __atomic_load_n_cuda(__a.get(), static_cast<__memory_order_underlying_t>(__order),  _Sco{});
        ),
        NV_IS_HOST, (
            return __atomic_load_host(__a.get(), __order);
        )
    )
}

template <typename _Tp, typename _Sco = __thread_scope_system_tag, typename _Tag = __atomic_enable_if_default_base_t<_Tp>>
_CCCL_HOST_DEVICE
__atomic_underlying_t<_Tp> __atomic_exchange_dispatch(_Tp& __a, __atomic_underlying_t<_Tp> __value, memory_order __order, _Sco = {}, _Tag = {}) {
    alignas(_Tp) auto __tmp = __value;
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            return __atomic_exchange_n_cuda(__a.get(), __tmp, static_cast<__memory_order_underlying_t>(__order), _Sco{});
        ),
        NV_IS_HOST, (
            return __atomic_exchange_host(__a.get(), __tmp, __order);
        )
    )
}

template <typename _Tp, typename _Sco = __thread_scope_system_tag, typename _Tag = __atomic_enable_if_default_base_t<_Tp>>
_CCCL_HOST_DEVICE
 bool __atomic_compare_exchange_strong_dispatch(_Tp& __a, __atomic_underlying_t<_Tp>* __expected, __atomic_underlying_t<_Tp> __val, memory_order __success, memory_order __failure, _Sco = {}, _Tag = {}) {
    bool __result = false;
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            __result = __atomic_compare_exchange_cuda(__a.get(), __expected, __val, false, static_cast<__memory_order_underlying_t>(__success), static_cast<__memory_order_underlying_t>(__failure), _Sco{});
        ),
        NV_IS_HOST, (
            __result = __atomic_compare_exchange_strong_host(__a.get(), __expected, __val, __success, __failure);
        )
    )
    return __result;
}

template <typename _Tp, typename _Sco = __thread_scope_system_tag, typename _Tag = __atomic_enable_if_default_base_t<_Tp>>
_CCCL_HOST_DEVICE
 bool __atomic_compare_exchange_weak_dispatch(_Tp& __a, __atomic_underlying_t<_Tp>* __expected, __atomic_underlying_t<_Tp> __val, memory_order __success, memory_order __failure, _Sco = {}, _Tag = {}) {
    bool __result = false;
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            __result = __atomic_compare_exchange_cuda(__a.get(), __expected, __val,  true, static_cast<__memory_order_underlying_t>(__success), static_cast<__memory_order_underlying_t>(__failure), _Sco{});
        ),
        NV_IS_HOST, (
            __result = __atomic_compare_exchange_weak_host(__a.get(), __expected, __val, __success, __failure);
        )
    )
    return __result;
}

template <typename _Tp>
using __atomic_enable_if_ptr = __enable_if_t<is_pointer<__atomic_underlying_t<_Tp>>::value, __atomic_underlying_t<_Tp>>;
template <typename _Tp>
using __atomic_enable_if_not_ptr = __enable_if_t<!is_pointer<__atomic_underlying_t<_Tp>>::value, __atomic_underlying_t<_Tp>>;

template <typename _Tp, typename _Sco = __thread_scope_system_tag, typename _Tag = __atomic_enable_if_default_base_t<_Tp>>
_CCCL_HOST_DEVICE
 __atomic_enable_if_not_ptr<_Tp> __atomic_fetch_add_dispatch(_Tp& __a, __atomic_underlying_t<_Tp> __delta, memory_order __order, _Sco = {}, _Tag = {}) {
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            return __atomic_fetch_add_cuda(__a.get(), __delta, static_cast<__memory_order_underlying_t>(__order), _Sco{});
        ),
        NV_IS_HOST, (
            return __atomic_fetch_add_host(__a.get(), __delta, __order);
        )
    )
}

template <typename _Tp, typename _Sco = __thread_scope_system_tag, typename _Tag = __atomic_enable_if_default_base_t<_Tp>>
_CCCL_HOST_DEVICE
 __atomic_enable_if_ptr<_Tp> __atomic_fetch_add_dispatch(_Tp& __a, ptrdiff_t __delta, memory_order __order, _Sco = {}, _Tag = {}) {
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            return __atomic_fetch_add_cuda(__a.get(), __delta, static_cast<__memory_order_underlying_t>(__order), _Sco{});
        ),
        NV_IS_HOST, (
            return __atomic_fetch_add_host(__a.get(), __delta, __order);
        )
    )
}

template <typename _Tp, typename _Sco = __thread_scope_system_tag, typename _Tag = __atomic_enable_if_default_base_t<_Tp>>
_CCCL_HOST_DEVICE
 __atomic_enable_if_not_ptr<_Tp> __atomic_fetch_sub_dispatch(_Tp& __a, __atomic_underlying_t<_Tp> __delta, memory_order __order, _Sco = {}, _Tag = {}) {
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            return __atomic_fetch_sub_cuda(__a.get(), __delta, static_cast<__memory_order_underlying_t>(__order), _Sco{});
        ),
        NV_IS_HOST, (
            return __atomic_fetch_sub_cuda(__a.get(), __delta, __order);
        )
    )
}

template <typename _Tp, typename _Sco = __thread_scope_system_tag, typename _Tag = __atomic_enable_if_default_base_t<_Tp>>
_CCCL_HOST_DEVICE
 __atomic_enable_if_ptr<_Tp> __atomic_fetch_sub_dispatch(_Tp& __a, ptrdiff_t __delta, memory_order __order, _Sco = {}, _Tag = {}) {
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            return __atomic_fetch_sub_cuda(__a.get(), __delta, static_cast<__memory_order_underlying_t>(__order), _Sco{});
        ),
        NV_IS_HOST, (
            return __atomic_fetch_sub_host(__a.get(), __delta, __order);
        )
    )
}

template <typename _Tp, typename _Sco = __thread_scope_system_tag, typename _Tag = __atomic_enable_if_default_base_t<_Tp>>
_CCCL_HOST_DEVICE
 __atomic_underlying_t<_Tp> __atomic_fetch_and_dispatch(_Tp& __a, __atomic_underlying_t<_Tp> __pattern, memory_order __order, _Sco = {}, _Tag = {}) {
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            return __atomic_fetch_and_cuda(__a.get(), __pattern, static_cast<__memory_order_underlying_t>(__order), _Sco{});
        ),
        NV_IS_HOST, (
            return __atomic_fetch_and_host(__a.get(), __pattern, __order);
        )
    )
}

template <typename _Tp, typename _Sco = __thread_scope_system_tag, typename _Tag = __atomic_enable_if_default_base_t<_Tp>>
_CCCL_HOST_DEVICE
 __atomic_underlying_t<_Tp> __atomic_fetch_or_dispatch(_Tp& __a, __atomic_underlying_t<_Tp> __pattern, memory_order __order, _Sco = {}, _Tag = {}) {
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            return __atomic_fetch_or_cuda(__a.get(), __pattern, static_cast<__memory_order_underlying_t>(__order), _Sco{});
        ),
        NV_IS_HOST, (
            return __atomic_fetch_or_host(__a.get(), __pattern, __order);
        )
    )
}

template <typename _Tp, typename _Sco = __thread_scope_system_tag, typename _Tag = __atomic_enable_if_default_base_t<_Tp>>
_CCCL_HOST_DEVICE
 __atomic_underlying_t<_Tp> __atomic_fetch_xor_dispatch(_Tp& __a, __atomic_underlying_t<_Tp> __pattern, memory_order __order, _Sco = {}, _Tag = {}) {
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            return __atomic_fetch_xor_cuda(__a.get(), __pattern, static_cast<__memory_order_underlying_t>(__order), _Sco{});
        ),
        NV_IS_HOST, (
            return __atomic_fetch_xor_host(__a.get(), __pattern, __order);
        )
    )
}

template <typename _Tp, typename _Sco = __thread_scope_system_tag, typename _Tag = __atomic_enable_if_default_base_t<_Tp>>
_CCCL_HOST_DEVICE
 __atomic_underlying_t<_Tp> __atomic_fetch_max_dispatch(_Tp& __a, __atomic_underlying_t<_Tp> __val, memory_order __order, _Sco = {}, _Tag = {}) {
    NV_IF_TARGET(
        NV_IS_DEVICE, (
            return __atomic_fetch_max_cuda(__a.get(), __val, static_cast<__memory_order_underlying_t>(__order), _Sco{});
        ), (
            return __atomic_fetch_max_host(__a.get(), __val, __order);
        )
    )
}

template <typename _Tp, typename _Sco = __thread_scope_system_tag, typename _Tag = __atomic_enable_if_default_base_t<_Tp>>
_CCCL_HOST_DEVICE
 __atomic_underlying_t<_Tp> __atomic_fetch_min_dispatch(_Tp& __a, __atomic_underlying_t<_Tp> __val, memory_order __order, _Sco = {}, _Tag = {}) {
    NV_IF_TARGET(
        NV_IS_DEVICE, (
            return __atomic_fetch_min_cuda(__a.get(), __val, static_cast<__memory_order_underlying_t>(__order), _Sco{});
        ), (
            return __atomic_fetch_min_host(__a.get(), __val, __order);
        )
    )
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // __LIBCUDACXX___ATOMIC_DISPATCH_H
