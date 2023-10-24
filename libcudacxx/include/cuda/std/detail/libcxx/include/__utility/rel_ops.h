//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___UTILITY_REL_OPS_H
#define _LIBCUDACXX___UTILITY_REL_OPS_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__utility/forward.h"
#include "../__utility/move.h"

#if defined(_CCCL_COMPILER_NVHPC) && defined(_CCCL_USE_IMPLICIT_SYSTEM_DEADER)
#pragma GCC system_header
#else // ^^^ _CCCL_COMPILER_NVHPC ^^^ / vvv !_CCCL_COMPILER_NVHPC vvv
_CCCL_IMPLICIT_SYSTEM_HEADER
#endif // !_CCCL_COMPILER_NVHPC

_LIBCUDACXX_BEGIN_NAMESPACE_STD

namespace rel_ops
{

template<class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI
bool
operator!=(const _Tp& __x, const _Tp& __y)
{
    return !(__x == __y);
}

template<class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI
bool
operator> (const _Tp& __x, const _Tp& __y)
{
    return __y < __x;
}

template<class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI
bool
operator<=(const _Tp& __x, const _Tp& __y)
{
    return !(__y < __x);
}

template<class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI
bool
operator>=(const _Tp& __x, const _Tp& __y)
{
    return !(__x < __y);
}

} // namespace rel_ops

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___UTILITY_REL_OPS_H
