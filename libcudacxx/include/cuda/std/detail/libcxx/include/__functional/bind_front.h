// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FUNCTIONAL_BIND_FRONT_H
#define _LIBCUDACXX___FUNCTIONAL_BIND_FRONT_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include "../__concepts/__concept_macros.h"
#include "../__functional/invoke.h"
#include "../__functional/perfect_forward.h"
#include "../__type_traits/decay.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_constructible.h"
#include "../__type_traits/is_move_constructible.h"
#include "../__type_traits/is_nothrow_constructible.h"
#include "../__utility/forward.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_STD_VER > 2014

struct __bind_front_op {
    template <class ..._Args>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto operator()(_Args&& ...__args) const
        noexcept(noexcept(_CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Args>(__args)...)))
        -> decltype(      _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Args>(__args)...))
        { return          _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Args>(__args)...); }
};

template <class _Fn, class ..._BoundArgs>
struct __bind_front_t : __perfect_forward<__bind_front_op, _Fn, _BoundArgs...> {
    using __base = __perfect_forward<__bind_front_op, _Fn, _BoundArgs...>;
#if defined(_CCCL_COMPILER_NVRTC)
    constexpr __bind_front_t() noexcept = default;

    template<class... _Args>
    _LIBCUDACXX_INLINE_VISIBILITY constexpr
    __bind_front_t(_Args&&... __args) noexcept(noexcept(__base(_CUDA_VSTD::declval<_Args>()...)))
        : __base(_CUDA_VSTD::forward<_Args>(__args)...)
    {}
#else
    using __base::__base;
#endif
};

template<class _Fn, class... _Args>
_LIBCUDACXX_CONCEPT __can_bind_front = is_constructible_v<decay_t<_Fn>, _Fn> &&
                                       is_move_constructible_v<decay_t<_Fn>> &&
                                       (is_constructible_v<decay_t<_Args>, _Args> && ...) &&
                                       (is_move_constructible_v<decay_t<_Args>> && ... );

_LIBCUDACXX_TEMPLATE(class _Fn, class... _Args)
  _LIBCUDACXX_REQUIRES( __can_bind_front<_Fn, _Args...>)
_LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
constexpr auto bind_front(_Fn&& __f, _Args&&... __args) noexcept(is_nothrow_constructible_v<tuple<decay_t<_Args>...>, _Args&&...>) {
    return __bind_front_t<decay_t<_Fn>, decay_t<_Args>...>(_CUDA_VSTD::forward<_Fn>(__f), _CUDA_VSTD::forward<_Args>(__args)...);
}

#endif // _CCCL_STD_VER > 2014

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___FUNCTIONAL_BIND_FRONT_H
