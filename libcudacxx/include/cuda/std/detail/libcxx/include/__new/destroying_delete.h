// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___NEW_DESTROYING_DELETEL_H
#define _LIBCUDACXX___NEW_DESTROYING_DELETEL_H

#ifndef __cuda_std__
#  include <__config>
#endif //__cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if defined(__cpp_impl_destroying_delete) && defined(__cpp_lib_destroying_delete)

_LIBCUDACXX_BEGIN_NAMESPACE_STD_NOVERSION // purposefully not using versioning namespace

using ::std::destroying_delete_t;
using ::std::destroying_delete;

_LIBCUDACXX_END_NAMESPACE_STD_NOVERSION

#endif // __cpp_impl_destroying_delete && __cpp_lib_destroying_delete

#endif // _LIBCUDACXX___NEW_DESTROYING_DELETEL_H
