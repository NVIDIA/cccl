// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___UTILITY_TO_UNDERLYING_H
#define _LIBCUDACXX___UTILITY_TO_UNDERLYING_H

#include <cuda/std/detail/__config>

_CCCL_IMPLICIT_SYSTEM_HEADER

#include <cuda/std/__type_traits/underlying_type.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY constexpr typename underlying_type<_Tp>::type __to_underlying(_Tp __val) noexcept
{
  return static_cast<typename underlying_type<_Tp>::type>(__val);
}

#if _CCCL_STD_VER > 2020
template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_INLINE_VISIBILITY constexpr underlying_type_t<_Tp> to_underlying(_Tp __val) noexcept
{
  return _CUDA_VSTD::__to_underlying(__val);
}
#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___UTILITY_TO_UNDERLYING_H
