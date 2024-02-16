//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___UTILITY_UNREACHABLE_H
#define _LIBCUDACXX___UTILITY_UNREACHABLE_H

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

#include "../cstdlib"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_LIBCUDACXX_NORETURN _LIBCUDACXX_INLINE_VISIBILITY
inline void __libcpp_unreachable()
{
  _LIBCUDACXX_UNREACHABLE();
}

#if _CCCL_STD_VER > 2020

[[noreturn]] _LIBCUDACXX_INLINE_VISIBILITY
inline void unreachable() { _LIBCUDACXX_UNREACHABLE(); }

#endif // _CCCL_STD_VER > 2020

_LIBCUDACXX_END_NAMESPACE_STD

#endif
