// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FUNCTIONAL_COMPOSE_H
#define _LIBCUDACXX___FUNCTIONAL_COMPOSE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__concepts/__concept_macros.h"
#include "../__functional/invoke.h"
#include "../__functional/perfect_forward.h"
#include "../__type_traits/decay.h"
#include "../__type_traits/is_same.h"
#include "../__utility/declval.h"
#include "../__utility/forward.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 14

struct __compose_op {
    template<class _Fn1, class _Fn2, class ..._Args>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto operator()(_Fn1&& __f1, _Fn2&& __f2, _Args&&... __args) const
        noexcept(noexcept(_CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fn1>(__f1), _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fn2>(__f2), _CUDA_VSTD::forward<_Args>(__args)...))))
        -> decltype(      _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fn1>(__f1), _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fn2>(__f2), _CUDA_VSTD::forward<_Args>(__args)...)))
        { return          _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fn1>(__f1), _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fn2>(__f2), _CUDA_VSTD::forward<_Args>(__args)...)); }
};

template <class _Fn1, class _Fn2>
struct __compose_t : __perfect_forward<__compose_op, _Fn1, _Fn2> {
    using __base = __perfect_forward<__compose_op, _Fn1, _Fn2>;
#if defined(_LIBCUDACXX_COMPILER_NVRTC)
    constexpr __compose_t() noexcept = default;

    _LIBCUDACXX_TEMPLATE(class _OrigFn1, class _OrigFn2)
      (requires _LIBCUDACXX_TRAIT(is_same, _Fn1, __decay_t<_OrigFn1>) _LIBCUDACXX_AND
                _LIBCUDACXX_TRAIT(is_same, _Fn2, __decay_t<_OrigFn2>)
      )
    _LIBCUDACXX_INLINE_VISIBILITY constexpr
    __compose_t(_OrigFn1&& __fn1, _OrigFn2&& __fn2)
      noexcept(noexcept(__base(cuda::std::declval<_OrigFn1>(), cuda::std::declval<_OrigFn2>())))
      : __base(_CUDA_VSTD::forward<_OrigFn1>(__fn1), _CUDA_VSTD::forward<_OrigFn2>(__fn2))
    {}
#else
    using __base::__base;
#endif
};

template <class _Fn1, class _Fn2>
_LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
constexpr auto __compose(_Fn1&& __f1, _Fn2&& __f2)
    noexcept(noexcept(__compose_t<decay_t<_Fn1>, decay_t<_Fn2>>(_CUDA_VSTD::forward<_Fn1>(__f1), _CUDA_VSTD::forward<_Fn2>(__f2))))
    -> decltype(      __compose_t<decay_t<_Fn1>, decay_t<_Fn2>>(_CUDA_VSTD::forward<_Fn1>(__f1), _CUDA_VSTD::forward<_Fn2>(__f2)))
    { return          __compose_t<decay_t<_Fn1>, decay_t<_Fn2>>(_CUDA_VSTD::forward<_Fn1>(__f1), _CUDA_VSTD::forward<_Fn2>(__f2)); }

#endif // _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___FUNCTIONAL_COMPOSE_H
