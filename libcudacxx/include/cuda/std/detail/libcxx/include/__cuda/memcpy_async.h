// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX__CUDA_MEMCPY_ASYNC_H
#define _LIBCUDACXX__CUDA_MEMCPY_ASYNC_H

#include "../cstdlib"
#include "../bit"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

struct __single_thread_group {
    _LIBCUDACXX_INLINE_VISIBILITY
    void sync() const {}
    _LIBCUDACXX_INLINE_VISIBILITY
    constexpr _CUDA_VSTD::size_t size() const { return 1; };
    _LIBCUDACXX_INLINE_VISIBILITY
    constexpr _CUDA_VSTD::size_t thread_rank() const { return 0; };
};

template<_CUDA_VSTD::size_t _Alignment>
struct aligned_size_t {
    static constexpr _CUDA_VSTD::size_t align = _Alignment;
    _CUDA_VSTD::size_t value;
    _LIBCUDACXX_INLINE_VISIBILITY
    explicit aligned_size_t(size_t __s) : value(__s) { }
    _LIBCUDACXX_INLINE_VISIBILITY
    operator size_t() const { return value; }
};

// Type only used for logging purpose
enum async_contract_fulfillment {
    none,
    async
};

enum class __tx_api {
    __yes,
    __no
};

enum class __space {
    __local,
    __shared,
    __cluster,
    __global,
    __constant,
    __grid_constant
};

template<__space _Sp>
using __space_constant = _CUDA_VSTD::integral_constant<__space, _Sp>;

_LIBCUDACXX_DEVICE
bool __is_cluster_shared(const void * __p) {
    NV_IF_TARGET(NV_PROVIDES_SM_90,
        (return __isClusterShared(__p);),
        ((void)__p; return false;)
    )
}

_LIBCUDACXX_DEVICE
bool __is_grid_constant(const void * __p) {
#ifdef _LIBCUDACXX_COMPILER_NVCC_BELOW_11_7
    return false;
#else
    return __isGridConstant(__p);
#endif
}

#define _LIBCUDACXX_HANDLE_SPACE(intrinsic, enum_v, name, ...) \
    if (intrinsic(name)) { \
        using name ## _space_t = __space_constant<__space::enum_v>; \
        __VA_ARGS__ \
    } else

#define _LIBCUDACXX_HANDLE_POINTER_SPACE(name, ...) \
    NV_IF_ELSE_TARGET( \
        NV_IS_DEVICE, ( \
            _LIBCUDACXX_HANDLE_SPACE(__is_cluster_shared,__cluster, name, __VA_ARGS__) \
            _LIBCUDACXX_HANDLE_SPACE(__isShared, __shared, name, __VA_ARGS__) \
            _LIBCUDACXX_HANDLE_SPACE(__isGlobal, __global, name, __VA_ARGS__) \
            _LIBCUDACXX_HANDLE_SPACE(__is_grid_constant, __grid_constant, name, __VA_ARGS__) \
            _LIBCUDACXX_HANDLE_SPACE(__isConstant, __constant, name, __VA_ARGS__) \
            _LIBCUDACXX_HANDLE_SPACE(/* __isLocal when not broken */ bool, __local, name, __VA_ARGS__) \
            {} \
        ), ( \
            _LIBCUDACXX_HANDLE_SPACE(bool, __global, name, __VA_ARGS__) \
            {} \
        ) \
    )

template<typename _Tag, _CUDA_VSTD::size_t _Value>
struct __down_convertible_constant {
    template<_CUDA_VSTD::size_t _OtherValue, typename = _CUDA_VSTD::__enable_if_t<(_OtherValue < _Value)>>
    _LIBCUDACXX_INLINE_VISIBILITY
    constexpr operator __down_convertible_constant<_Tag, _OtherValue>() const {
        return {};
    }

    static const constexpr _CUDA_VSTD::size_t value = _Value;
};

struct __alignment_tag {};
template<_CUDA_VSTD::size_t _Alignment>
using __alignment = __down_convertible_constant<__alignment_tag, _Alignment>;

template<_CUDA_VSTD::size_t _GuaranteedAlignment,
    _CUDA_VSTD::size_t _MaxInterestingAlignment,
    _CUDA_VSTD::size_t _MinInterestingAlignment,
    _CUDA_VSTD::size_t _Alignment,
    typename = void>
struct __memcpy_async_invoke_if_applicable {
    template<typename _Fn>
    _LIBCUDACXX_INLINE_VISIBILITY
    static async_contract_fulfillment __invoke(_Fn && __f) {
        return _CUDA_VSTD::forward<_Fn>(__f)(__alignment<_Alignment>());
    }
};

template<_CUDA_VSTD::size_t _GuaranteedAlignment,
    _CUDA_VSTD::size_t _MaxInterestingAlignment,
    _CUDA_VSTD::size_t _MinInterestingAlignment,
    _CUDA_VSTD::size_t _Alignment>
struct __memcpy_async_invoke_if_applicable<_GuaranteedAlignment,
    _MaxInterestingAlignment,
    _MinInterestingAlignment,
    _Alignment,
    _CUDA_VSTD::__enable_if_t<
        // These are the cases in which we want to _not_ generate code in the switch below.
        // If alignment is greater than max interesting, there's an if in front of the switch for that.
        // If alignment is less than min interesting, there's also an if that sets the alignment to 1
        // (so that non-interesting cases can all be handled with just 1 as the value, avoiding instantiating
        // them a bunch of times).
        // If alignment is less than guaranteed alignment, then that switch case will never be taken, because
        // the alignment is guaranteed to be above the value it represents.
        _Alignment >= _MaxInterestingAlignment
            || _Alignment < _MinInterestingAlignment
            || _Alignment < _GuaranteedAlignment
>> {
    template<typename _Fn>
    _LIBCUDACXX_INLINE_VISIBILITY
    static async_contract_fulfillment __invoke(_Fn && __f) {
        _LIBCUDACXX_UNREACHABLE();
    }
};

template<bool _ShouldInvoke>
struct __memcpy_async_invoke_if_true : _CUDA_VSTD::true_type {
    template<typename _Fn>
    _LIBCUDACXX_INLINE_VISIBILITY
    static async_contract_fulfillment __invoke(_Fn && __f) {
        return _CUDA_VSTD::forward<_Fn>(__f)(__alignment<1>());
    }
};

template<>
struct __memcpy_async_invoke_if_true<false> : _CUDA_VSTD::false_type {
    template<typename _Fn>
    _LIBCUDACXX_INLINE_VISIBILITY
    static async_contract_fulfillment __invoke(_Fn && __f) {
        _LIBCUDACXX_UNREACHABLE();
    }
};

template<_CUDA_VSTD::size_t _MaxInterestingAlignment,
    _CUDA_VSTD::size_t _MinInterestingAlignment,
    _CUDA_VSTD::size_t _GuaranteedAlignment,
    typename _Fn>
_LIBCUDACXX_INLINE_VISIBILITY async_contract_fulfillment __dispatch_alignment_bit(_Fn && __f, _CUDA_VSTD::size_t __alignment_fsb) {
    const _CUDA_VSTD::size_t __alignment_v = 1ull << (__alignment_fsb - 1);

    if (__builtin_expect(__alignment_v >= _MaxInterestingAlignment, true)) {
        return _CUDA_VSTD::forward<_Fn>(__f)(__alignment<_MaxInterestingAlignment>());
    }

    using __not_guaranteed_interesting = __memcpy_async_invoke_if_true<_GuaranteedAlignment < _MinInterestingAlignment>;
    if (__not_guaranteed_interesting::value) {
        if (__builtin_expect(__alignment_v < _MinInterestingAlignment, false)) {
            __not_guaranteed_interesting::__invoke(_CUDA_VSTD::forward<_Fn>(__f));
        }
    }

    switch (__alignment_fsb) {
#define _ADD_CASE(val)                                                                                      \
    case val:                                                                                               \
        return __memcpy_async_invoke_if_applicable<                                                                      \
            _GuaranteedAlignment,                                                                           \
            _MaxInterestingAlignment,                                                                       \
            _MinInterestingAlignment,                                                                       \
            1ull << (val - 1)>::__invoke(_CUDA_VSTD::forward<_Fn>(__f))

        _ADD_CASE(12);
        _ADD_CASE(11);
        _ADD_CASE(10);
        _ADD_CASE(9);
        _ADD_CASE(8);
        _ADD_CASE(7);
        _ADD_CASE(6);
        _ADD_CASE(5);
        _ADD_CASE(4);
        _ADD_CASE(3);
        _ADD_CASE(2);
        _ADD_CASE(1);

#undef _ADD_CASE
    }

    _LIBCUDACXX_UNREACHABLE();
}

_LIBCUDACXX_INLINE_VISIBILITY _CUDA_VSTD::size_t __get_size(_CUDA_VSTD::size_t __size) {
    return __size;
}

template<_CUDA_VSTD::size_t _Alignment>
_LIBCUDACXX_INLINE_VISIBILITY _CUDA_VSTD::size_t __get_size(aligned_size_t<_Alignment> __size) {
    return __size.value;
}

_LIBCUDACXX_INLINE_VISIBILITY
constexpr _CUDA_VSTD::integral_constant<_CUDA_VSTD::size_t, 1> __get_alignment(_CUDA_VSTD::size_t);

template<_CUDA_VSTD::size_t _Alignment>
_LIBCUDACXX_INLINE_VISIBILITY
constexpr _CUDA_VSTD::integral_constant<_CUDA_VSTD::size_t, _Alignment> __get_alignment(aligned_size_t<_Alignment>);

namespace __arch
{
struct __cuda_tag {};
template<_CUDA_VSTD::size_t _ProvidedSM>
using __cuda = __down_convertible_constant<__cuda_tag, _ProvidedSM>;

template<typename _Tp, _CUDA_VSTD::size_t _RequestedSM>
struct __is_cuda_provides_sm : _CUDA_VSTD::false_type {
};

template<_CUDA_VSTD::size_t _ProvidedSM, _CUDA_VSTD::size_t _RequestedSM>
struct __is_cuda_provides_sm<__cuda<_ProvidedSM>, _RequestedSM> : _CUDA_VSTD::_LIBCUDACXX_BOOL_CONSTANT((_ProvidedSM >= _RequestedSM)) {
};

struct __host {};
}

_LIBCUDACXX_INLINE_VISIBILITY
async_contract_fulfillment __strided_memcpy(_CUDA_VSTD::size_t __rank, _CUDA_VSTD::size_t __group_size, char * __out_ptr, const char * __in_ptr, _CUDA_VSTD::size_t __size, _CUDA_VSTD::size_t __alignment) {
    if (__group_size == 1) {
        memcpy(__out_ptr, __in_ptr, __size);
    }
    else {
        for (_CUDA_VSTD::size_t __offset = __rank * __alignment; __offset < __size; __offset += __group_size * __alignment) {
            memcpy(__out_ptr + __offset, __in_ptr + __offset, __alignment);
        }
    }
    return async_contract_fulfillment::none;
}

template<typename _Arch, __tx_api _Tx, _CUDA_VSTD::size_t _Alignment, __space _OutSpace, __space InSpace, __space _SyncSpace, typename = void>
struct __memcpy_async_default_aligned_impl {
    template<typename _Group, typename _Sync>
    _LIBCUDACXX_INLINE_VISIBILITY static async_contract_fulfillment __memcpy_async(
        _Arch,
        __alignment<_Alignment>,
        _Group & __g,
        char *__out_ptr,
        const char *__in_ptr,
        _CUDA_VSTD::size_t __size,
        _Sync & __sync
    ) {
        return __strided_memcpy(__g.thread_rank(), __g.size(), __out_ptr, __in_ptr, __size, _Alignment);
    }
};

template<_CUDA_VSTD::size_t _Alignment>
__device__
async_contract_fulfillment __cp_async_cg_shared_global(_CUDA_VSTD::size_t __rank, _CUDA_VSTD::size_t __group_size, char * __out_ptr, const char * __in_ptr, _CUDA_VSTD::size_t __size) {
    _CUDA_VSTD::size_t __offset = __rank * _Alignment;
    auto __shptr = __cvta_generic_to_shared(__out_ptr) + __offset;
    __in_ptr += __offset;
    const auto __stride = __group_size * _Alignment;

    for (; __offset < __size; __offset += __stride) {
        asm volatile ("cp.async.cg.shared.global [%0], [%1], %2, %2;"
            :: "l"(__shptr),
                "l"(__in_ptr),
                "n"(_Alignment)
            : "memory");
        __shptr += __stride;
        __in_ptr += __stride;
    }
    return async_contract_fulfillment::async;
}

template<_CUDA_VSTD::size_t _Alignment>
__device__
async_contract_fulfillment __cp_async_ca_shared_global(_CUDA_VSTD::size_t __rank, _CUDA_VSTD::size_t __group_size, char * __out_ptr, const char * __in_ptr, _CUDA_VSTD::size_t __size) {
    _CUDA_VSTD::size_t __offset = __rank * _Alignment;
    auto __shptr = __cvta_generic_to_shared(__out_ptr) + __offset;
    __in_ptr += __offset;
    const auto __stride = __group_size * _Alignment;

    for (; __offset < __size; __offset += __stride) {
        asm volatile ("cp.async.ca.shared.global [%0], [%1], %2, %2;"
            :: "l"(__shptr),
                "l"(__in_ptr),
                "n"(_Alignment)
            : "memory");
        __shptr += __stride;
        __in_ptr += __stride;
    }
    return async_contract_fulfillment::async;
}

template<_CUDA_VSTD::size_t _ProvidedSM, _CUDA_VSTD::size_t _Alignment, __space _SyncSpace>
struct __memcpy_async_default_aligned_impl<
    __arch::__cuda<_ProvidedSM>, __tx_api::__no, _Alignment,
    __space::__shared, __space::__global, _SyncSpace,
    _CUDA_VSTD::__enable_if_t<_Alignment >= 4>
> {
    template<typename _Group, typename _Sync>
    __device__ static async_contract_fulfillment __memcpy_async(
        __arch::__cuda<80>,
        __alignment<16>,
        _Group & __g,
        char *__out_ptr,
        const char *__in_ptr,
        _CUDA_VSTD::size_t __size,
        _Sync & __sync
    ) {
        return __cp_async_cg_shared_global<16>(__g.thread_rank(), __g.size(), __out_ptr, __in_ptr, __size);
    }

    template<typename _Group, typename _Sync>
    __device__ static async_contract_fulfillment __memcpy_async(
        __arch::__cuda<80>,
        __alignment<4>,
        _Group & __g,
        char *__out_ptr,
        const char *__in_ptr,
        _CUDA_VSTD::size_t __size,
        _Sync & __sync
    ) {
        return __cp_async_ca_shared_global<_Alignment>(__g.thread_rank(), __g.size(), __out_ptr, __in_ptr, __size);
    }
};

_LIBCUDACXX_INLINE_VISIBILITY
_CUDA_VSTD::size_t __memcpy_async_ffs(_CUDA_VSTD::size_t __val) {
    NV_IF_ELSE_TARGET(
        NV_IS_DEVICE,
        (return __ffsll(__val);),
        (return _CUDA_VSTD::__libcpp_ctz(__val) + 1;)
    )
}

struct __proto_hooks {
    template<typename _Group, typename _Size, typename _Sync>
    _LIBCUDACXX_INLINE_VISIBILITY static _CUDA_VSTD::size_t __compute_alignment_bit(const _Group & __g,
        char * __out_ptr,
        const char *__in_ptr,
        _Size __size,
        const _Sync & __sync
    ) {
        auto __out_addr = reinterpret_cast<_CUDA_VSTD::uintptr_t>(__out_ptr);
        auto __in_addr = reinterpret_cast<_CUDA_VSTD::uintptr_t>(__in_ptr);
        auto __size_val = __get_size(__size);

        auto __bit_pattern = (__out_addr | __in_addr | __size_val) & (__max_interesting_alignment - 1);

        if (__bit_pattern == 0) {
            return __memcpy_async_ffs(__max_interesting_alignment);
        }
        else {
            return __memcpy_async_ffs(__bit_pattern);
        }
    }

    static const constexpr _CUDA_VSTD::size_t __max_interesting_alignment = 16;
    static const constexpr _CUDA_VSTD::size_t __min_interesting_alignment = 4;
};

template<__tx_api _Tx, typename _Arch, __space _OutSpace, __space _InSpace, __space _SyncSpace, typename = void>
struct __memcpy_async_hooks : __proto_hooks {
    using __unspecialized = void;
};

template<__tx_api _Tx, _CUDA_VSTD::size_t _ProvidedSM, __space _SyncSpace>
struct __memcpy_async_hooks<_Tx, __arch::__cuda<_ProvidedSM>, __space::__shared, __space::__global, _SyncSpace, _CUDA_VSTD::__enable_if_t<_ProvidedSM >= 80>>
    : __proto_hooks {
    template<_CUDA_VSTD::size_t _Alignment>
    using __aligned = __memcpy_async_default_aligned_impl<__arch::__cuda<_ProvidedSM>, _Tx, _Alignment, __space::__shared, __space::__global, _SyncSpace>;
};

template<typename _Tp>
struct __dependent_false : std::false_type {};

template<typename _Sync, __tx_api _Tx, typename _Arch, __space _OutSpace, __space _InSpace, __space _SyncSpace, typename = void>
struct __memcpy_async_sync_hooks {
    static_assert(
        __dependent_false<_Sync>::value,
        "the provided synchronization object is not a valid synchronization object for this invocation of memcpy_async");

    _LIBCUDACXX_INLINE_VISIBILITY
    static async_contract_fulfillment __synchronize(_Arch, const _Sync &, async_contract_fulfillment);
};

struct __noop_sync_hooks {
    template<typename _Arch, typename _Sync>
    _LIBCUDACXX_INLINE_VISIBILITY
    static async_contract_fulfillment __synchronize(_Arch, _Sync &&, async_contract_fulfillment __acf) {
        _LIBCUDACXX_ASSERT(__acf == async_contract_fulfillment::none, "error in memcpy_async: asynchronous copy invoked a noop sync");
        return __acf;
    }
};

template<__tx_api _Tx, typename _Arch, typename _InSpace, typename _OutSpace, typename _SyncSpace, typename = void>
struct __are_memcpy_async_hooks_specialized : _CUDA_VSTD::true_type {
    template<typename _Fn>
    _LIBCUDACXX_INLINE_VISIBILITY
    static async_contract_fulfillment __invoke(_Fn && __f) {
        return _CUDA_VSTD::forward<_Fn>(__f)(_InSpace{}, _OutSpace{}, _SyncSpace{});
    }
};

template<__tx_api _Tx, typename _Arch, typename _InSpace, typename _OutSpace, typename _SyncSpace>
struct __are_memcpy_async_hooks_specialized<_Tx, _Arch, _InSpace, _OutSpace, _SyncSpace,
    _CUDA_VSTD::__void_t<
        typename __memcpy_async_hooks<_Tx, _Arch, _InSpace::value, _OutSpace::value, _SyncSpace::value>::__unspecialized
    >
> : _CUDA_VSTD::false_type {
    template<typename _Fn>
    _LIBCUDACXX_INLINE_VISIBILITY
    static async_contract_fulfillment __invoke(_Fn &&) {
        _LIBCUDACXX_UNREACHABLE();
    }
};

template<typename _Ret, typename _Fn>
_LIBCUDACXX_INLINE_VISIBILITY _Ret __dispatch_architecture(_Fn && __f)
{
    NV_DISPATCH_TARGET(
        NV_PROVIDES_SM_90,
        (return _CUDA_VSTD::forward<_Fn>(__f)(__arch::__cuda<90>());),
        NV_PROVIDES_SM_80,
        (return _CUDA_VSTD::forward<_Fn>(__f)(__arch::__cuda<80>());),
        NV_PROVIDES_SM_70,
        (return _CUDA_VSTD::forward<_Fn>(__f)(__arch::__cuda<70>());),
        NV_IS_HOST,
        (return _CUDA_VSTD::forward<_Fn>(__f)(__arch::__host());))
}

template<typename _Hooks, _CUDA_VSTD::size_t _NativeAlignment, typename _Arch, typename _Group, typename _Size, typename _SyncObject>
struct __memcpy_async_alignment_dispatcher_t {
    _Group && __g;
    char * __out_ptr;
    const char * __in_ptr;
    _Size __size;
    _SyncObject & __sync;

    template<typename _Alignment>
    _LIBCUDACXX_INLINE_VISIBILITY
    async_contract_fulfillment operator()(_Alignment) {
        return _Hooks::template __aligned<_Alignment::value>::__memcpy_async(
            _Arch{}, _Alignment{}, __g, __out_ptr, __in_ptr, __get_size(__size), __sync);
    }
};

template<typename _Hooks, _CUDA_VSTD::size_t _NativeAlignment, typename _Arch, typename _Group, typename _Size, typename _SyncObject>
_LIBCUDACXX_INLINE_VISIBILITY
__memcpy_async_alignment_dispatcher_t<_Hooks, _NativeAlignment, _Arch, _Group, _Size, _SyncObject> __memcpy_async_alignment_dispatcher(_Group && __g, char * __out_ptr, const char * __in_ptr, _Size __size, _SyncObject & __sync) {
    return { _CUDA_VSTD::forward<_Group>(__g), __out_ptr, __in_ptr, __size, __sync };
}

template<__tx_api _Tx, _CUDA_VSTD::size_t _NativeAlignment, typename _Arch, typename _Group, typename _Size, typename _SyncObject>
struct __memcpy_async_space_dispatcher_t {
    _Group && __g;
    char * __out_ptr;
    const char * __in_ptr;
    _Size __size;
    _SyncObject & __sync;

    template<typename _OutSpace, typename _InSpace, typename _SyncSpace>
    _LIBCUDACXX_INLINE_VISIBILITY
    async_contract_fulfillment operator()(_OutSpace, _InSpace, _SyncSpace) {
        using __hooks = __memcpy_async_hooks<_Tx,
            _Arch,
            _OutSpace::value,
            _InSpace::value,
            _SyncSpace::value>;

        auto __f = __memcpy_async_alignment_dispatcher<__hooks, _NativeAlignment, _Arch>(_CUDA_VSTD::forward<_Group>(__g), __out_ptr, __in_ptr, __size, __sync);

        auto __alignment_bit_v = __hooks::__compute_alignment_bit(__g, __out_ptr, __in_ptr, __size, __sync);

        auto __acf = __dispatch_alignment_bit<
            __hooks::__max_interesting_alignment,
            __hooks::__min_interesting_alignment,
            (_NativeAlignment > decltype(__get_alignment(__size))::value ? _NativeAlignment : decltype(__get_alignment(__size))::value)
        >(__f, __alignment_bit_v);

        using __sync_hooks = __memcpy_async_sync_hooks<_CUDA_VSTD::__remove_cvref_t<_SyncObject>,
            _Tx,
            _Arch,
            _OutSpace::value,
            _InSpace::value,
            _SyncSpace::value>;
        return __sync_hooks::__synchronize(_Arch{}, __sync, __acf);
    }
};

template<__tx_api _Tx, _CUDA_VSTD::size_t _NativeAlignment, typename _Arch, typename _Group, typename _Size, typename _SyncObject>
_LIBCUDACXX_INLINE_VISIBILITY
__memcpy_async_space_dispatcher_t<_Tx, _NativeAlignment, _Arch, _Group, _Size, _SyncObject> __memcpy_async_space_dispatcher(_Group && __g, char * __out_ptr, const char * __in_ptr, _Size __size, _SyncObject & __sync) {
    return { _CUDA_VSTD::forward<_Group>(__g), __out_ptr, __in_ptr, __size, __sync };
}

template<__tx_api _Tx, _CUDA_VSTD::size_t _NativeAlignment, typename _Group, typename _Size, typename _SyncObject>
struct __memcpy_async_arch_dispatcher_t {
    _Group && __g;
    char * __out_ptr;
    const char * __in_ptr;
    _Size __size;
    _SyncObject & __sync;

    template<typename _Arch>
    _LIBCUDACXX_INLINE_VISIBILITY
    async_contract_fulfillment operator()(_Arch) {
        auto __f = __memcpy_async_space_dispatcher<_Tx, _NativeAlignment, _Arch>(_CUDA_VSTD::forward<_Group>(__g), __out_ptr, __in_ptr, __size, __sync);

        auto __sync_ptr = &__sync;
        _LIBCUDACXX_HANDLE_POINTER_SPACE(__out_ptr, _LIBCUDACXX_HANDLE_POINTER_SPACE(__in_ptr, _LIBCUDACXX_HANDLE_POINTER_SPACE(__sync_ptr,
            using __ahs = __are_memcpy_async_hooks_specialized<_Tx, _Arch, __out_ptr_space_t, __in_ptr_space_t, __sync_ptr_space_t>;
            if _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 (__ahs::value) {
                return __ahs::__invoke(__f);
            }
        )))

        // fallback to unspecialized implementation
        auto __alignment_bit_v = __proto_hooks::__compute_alignment_bit(__g, __out_ptr, __in_ptr, __size, __sync);
        __strided_memcpy(__g.thread_rank(), __g.size(), __out_ptr, __in_ptr, __get_size(__size), 1ull << (__alignment_bit_v - 1));
        return async_contract_fulfillment::none;
    }
};

template<__tx_api _Tx, _CUDA_VSTD::size_t _NativeAlignment, typename _Group, typename _Size, typename _SyncObject>
_LIBCUDACXX_INLINE_VISIBILITY
__memcpy_async_arch_dispatcher_t<_Tx, _NativeAlignment, _Group, _Size, _SyncObject> __memcpy_async_arch_dispatcher(_Group && __g, char * __out_ptr, const char * __in_ptr, _Size __size, _SyncObject & __sync) {
    return { _CUDA_VSTD::forward<_Group>(__g), __out_ptr, __in_ptr, __size, __sync };
}

template<__tx_api _Tx, _CUDA_VSTD::size_t _NativeAlignment, typename _Group, typename _Size, typename _SyncObject>
_LIBCUDACXX_INLINE_VISIBILITY async_contract_fulfillment __memcpy_async(
    _Group && __g,
    char * __out_ptr,
    const char *__in_ptr,
    _Size __size,
    _SyncObject & __sync)
{
    return __dispatch_architecture<async_contract_fulfillment>(__memcpy_async_arch_dispatcher<_Tx, _NativeAlignment>(_CUDA_VSTD::forward<_Group>(__g), __out_ptr, __in_ptr, __size, __sync));
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _LIBCUDACXX__CUDA_MEMCPY_ASYNC_H
