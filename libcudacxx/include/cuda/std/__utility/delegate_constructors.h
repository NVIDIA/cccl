//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___UTILITY_DELEGATE_CONSTRUCTORS_H
#define _CUDA_STD___UTILITY_DELEGATE_CONSTRUCTORS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__utility/forward.h>

#include <cuda/std/__cccl/prologue.h>

// NVRTC has a bug that prevented the use of delegated constructors, as it did not accept execution space annotations.
// This creates a whole lot of boilerplate that we can avoid through a macro (see nvbug3961621)
#if _CCCL_COMPILER(NVRTC, <, 12, 6)
#  define _CCCL_DELEGATE_CONSTRUCTORS(__class, __baseclass, ...)                                                       \
    using __base = __baseclass<__VA_ARGS__>;                                                                           \
    _CCCL_TEMPLATE(class... _Args)                                                                                     \
    _CCCL_REQUIRES(::cuda::std::is_constructible_v<__base, _Args...>)                                                  \
    _CCCL_API constexpr __class(_Args&&... __args) noexcept(::cuda::std::is_nothrow_constructible_v<__base, _Args...>) \
        : __base(::cuda::std::forward<_Args>(__args)...)                                                               \
    {}                                                                                                                 \
    _CCCL_HIDE_FROM_ABI constexpr __class() noexcept(::cuda::std::is_nothrow_default_constructible_v<__base>) = default;
#else // ^^^ workaround ^^^ / vvv no workaround vvv
#  define _CCCL_DELEGATE_CONSTRUCTORS(__class, __baseclass, ...) \
    using __base = __baseclass<__VA_ARGS__>;                     \
    using __base::__base;                                        \
    _CCCL_HIDE_FROM_ABI constexpr __class() noexcept(::cuda::std::is_nothrow_default_constructible_v<__base>) = default;
#endif // ^^^ no workaround ^^^

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___UTILITY_DELEGATE_CONSTRUCTORS_H
