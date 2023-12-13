// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___RANGES_ENABLE_BORROWED_RANGE_H
#define _LIBCUDACXX___RANGES_ENABLE_BORROWED_RANGE_H

// These customization variables are used in <span> and <string_view>. The
// separate header is used to avoid including the entire <ranges> header in
// <span> and <string_view>.

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

#if _CCCL_STD_VER > 2014

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

// [range.range], ranges

template <class>
_LIBCUDACXX_INLINE_VAR constexpr bool enable_borrowed_range = false;

_LIBCUDACXX_END_NAMESPACE_RANGES

#endif // _CCCL_STD_VER > 2014

#endif // _LIBCUDACXX___RANGES_ENABLE_BORROWED_RANGE_H
