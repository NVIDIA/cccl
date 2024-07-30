// -*- C++ -*-
//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===---------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FWD_SPAN_H
#define _LIBCUDACXX___FWD_SPAN_H

#include <cuda/std/detail/__config>

_CCCL_IMPLICIT_SYSTEM_HEADER

#include <cuda/std/cstddef>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_STD_VER > 2011

_LIBCUDACXX_INLINE_VAR constexpr size_t dynamic_extent = static_cast<size_t>(-1);
template <typename _Tp, size_t _Extent = dynamic_extent>
class span;

#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___FWD_SPAN_H
