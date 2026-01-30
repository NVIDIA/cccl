//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___OPTIONAL_BAD_OPTIONAL_ACCESS_H
#define _CUDA_STD___OPTIONAL_BAD_OPTIONAL_ACCESS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_EXCEPTIONS()

#  include <optional>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_NOVERSION

using bad_optional_access CCCL_DEPRECATED_BECAUSE("use std::bad_optional_access instead") = ::std::bad_optional_access;

_CCCL_END_NAMESPACE_CUDA_STD_NOVERSION

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_EXCEPTIONS()

#endif // _CUDA_STD___OPTIONAL_BAD_OPTIONAL_ACCESS_H
