//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_VALID_EXPANSION_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_VALID_EXPANSION_H

#ifndef __cuda_std__
#  include <__config>
#endif // __cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/detail/libcxx/include/__type_traits/integral_constant.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <template <class...> class _Templ, class... _Args, class = _Templ<_Args...>>
_LIBCUDACXX_INLINE_VISIBILITY true_type __sfinae_test_impl(int);
template <template <class...> class, class...>
_LIBCUDACXX_INLINE_VISIBILITY false_type __sfinae_test_impl(...);

template <template <class...> class _Templ, class... _Args>
using _IsValidExpansion _LIBCUDACXX_NODEBUG_TYPE = decltype(_CUDA_VSTD::__sfinae_test_impl<_Templ, _Args...>(0));

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_VALID_EXPANSION_H
