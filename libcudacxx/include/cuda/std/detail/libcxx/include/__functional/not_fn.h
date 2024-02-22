// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FUNCTIONAL_NOT_FN_H
#define _LIBCUDACXX___FUNCTIONAL_NOT_FN_H

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

#include "../__functional/invoke.h"
#include "../__functional/perfect_forward.h"
#include "../__type_traits/decay.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_constructible.h"
#include "../__type_traits/is_move_constructible.h"
#include "../__utility/forward.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_STD_VER > 2014

struct __not_fn_op {
    template <class... _Args>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 auto operator()(_Args&&... __args) const
        noexcept(noexcept(!_CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Args>(__args)...)))
        -> decltype(      !_CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Args>(__args)...))
        { return          !_CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Args>(__args)...); }
};

template <class _Fn>
struct __not_fn_t : __perfect_forward<__not_fn_op, _Fn> {
    using __base = __perfect_forward<__not_fn_op, _Fn>;
#if defined(_CCCL_COMPILER_NVRTC) // nvbug 3961621
    constexpr __not_fn_t() noexcept = default;

    _LIBCUDACXX_TEMPLATE(class _OrigFn)
        _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_same, _Fn, __decay_t<_OrigFn>))
    _LIBCUDACXX_INLINE_VISIBILITY constexpr
    __not_fn_t(_OrigFn&& __fn) noexcept(noexcept(__base(_CUDA_VSTD::declval<_OrigFn>())))
        : __base(_CUDA_VSTD::forward<_OrigFn>(__fn))
    {}
#else
    using __base::__base;
#endif
};

template <class _Fn, class = enable_if_t<
    is_constructible_v<decay_t<_Fn>, _Fn> &&
    is_move_constructible_v<decay_t<_Fn>>
>>
inline _LIBCUDACXX_INLINE_VISIBILITY
_LIBCUDACXX_CONSTEXPR_AFTER_CXX17 auto not_fn(_Fn&& __f) {
    return __not_fn_t<decay_t<_Fn>>(_CUDA_VSTD::forward<_Fn>(__f));
}

#endif // _CCCL_STD_VER > 2014

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___FUNCTIONAL_NOT_FN_H
