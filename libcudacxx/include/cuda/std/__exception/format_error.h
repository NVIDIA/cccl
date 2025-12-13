// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___EXCEPTION_FORMAT_ERROR_H
#define _CUDA_STD___EXCEPTION_FORMAT_ERROR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__exception/msg_storage.h>
#include <cuda/std/__string/string_view.h>
#include <cuda/std/source_location>

#if !_CCCL_COMPILER(NVRTC)
#  include <cstdio>
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

#if !_CCCL_COMPILER(NVRTC)

inline char* __format_error(__msg_storage& __msg_buffer,
                            ::cuda::std::__string_view __type_name,
                            const char* __msg,
                            ::cuda::std::source_location __loc = ::cuda::std::source_location::current()) noexcept
{
  (void) ::snprintf(
    __msg_buffer.__buffer,
    __msg_buffer.__size,
    "%s:%u: %.*s: %s",
    __loc.file_name(),
    __loc.line(),
    static_cast<int>(__type_name.size()),
    (__type_name.data() != nullptr) ? __type_name.data() : "<unknown_type>",
    (__msg != nullptr) ? __msg : "<unknown_error>");
  return __msg_buffer.__buffer;
}

#endif // !_CCCL_COMPILER(NVRTC)

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___EXCEPTION_FORMAT_ERROR_H
