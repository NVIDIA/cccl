//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_NAT_H
#define _LIBCUDACXX___TYPE_TRAITS_NAT_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

_CCCL_IMPLICIT_SYSTEM_HEADER

_LIBCUDACXX_BEGIN_NAMESPACE_STD

struct __nat
{
    __nat() = delete;
    __nat(const __nat&) = delete;
    __nat& operator=(const __nat&) = delete;
    ~__nat() = delete;
};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_NAT_H
