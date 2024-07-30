//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___UTILITY_AS_CONST_H
#define _LIBCUDACXX___UTILITY_AS_CONST_H

#include <cuda/std/detail/__config>

_CCCL_IMPLICIT_SYSTEM_HEADER

#include <cuda/std/__type_traits/add_const.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_STD_VER > 2011
template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_INLINE_VISIBILITY constexpr add_const_t<_Tp>& as_const(_Tp& __t) noexcept
{
  return __t;
}

template <class _Tp>
void as_const(const _Tp&&) = delete;
#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___UTILITY_AS_CONST_H
