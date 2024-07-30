//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___EXPECTED_UNEXPECT_H
#define _LIBCUDACXX___EXPECTED_UNEXPECT_H

#include <cuda/std/detail/__config>

_CCCL_IMPLICIT_SYSTEM_HEADER

#if _CCCL_STD_VER > 2011

_LIBCUDACXX_BEGIN_NAMESPACE_STD

struct unexpect_t
{
  explicit unexpect_t() = default;
};

_CCCL_GLOBAL_CONSTANT unexpect_t unexpect{};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _CCCL_STD_VER > 2011

#endif // _LIBCUDACXX___EXPECTED_UNEXPECT_H
