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

#ifndef _CUDA_STD___NEW_BAD_ALLOC_H
#define _CUDA_STD___NEW_BAD_ALLOC_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__exception/terminate.h>

#if _CCCL_HAS_EXCEPTIONS()
#  include <new>
#endif // !_CCCL_HAS_EXCEPTIONS()

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

[[noreturn]] _CCCL_API inline void __throw_bad_alloc()
{
#if _CCCL_HAS_EXCEPTIONS()
  NV_IF_ELSE_TARGET(NV_IS_HOST, (throw ::std::bad_alloc();), (::cuda::std::terminate();))
#else // ^^^ _CCCL_HAS_EXCEPTIONS() ^^^ / vvv !_CCCL_HAS_EXCEPTIONS() vvv
  ::cuda::std::terminate();
#endif // !_CCCL_HAS_EXCEPTIONS()
}

[[noreturn]] _CCCL_API inline void __throw_bad_array_new_length()
{
#if _CCCL_HAS_EXCEPTIONS()
  NV_IF_ELSE_TARGET(NV_IS_HOST, (throw ::std::bad_array_new_length();), (::cuda::std::terminate();))
#else // ^^^ _CCCL_HAS_EXCEPTIONS() ^^^ / vvv !_CCCL_HAS_EXCEPTIONS() vvv
  ::cuda::std::terminate();
#endif // !_CCCL_HAS_EXCEPTIONS()
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___NEW_BAD_ALLOC_H
