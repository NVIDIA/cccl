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

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

_CCCL_IMPLICIT_SYSTEM_HEADER

#if _LIBCUDACXX_STD_VER > 11

_LIBCUDACXX_BEGIN_NAMESPACE_STD

struct unexpect_t {
  explicit unexpect_t() = default;
};

_LIBCUDACXX_CPO_ACCESSIBILITY unexpect_t unexpect{};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX_STD_VER > 11

#endif // _LIBCUDACXX___EXPECTED_UNEXPECT_H
