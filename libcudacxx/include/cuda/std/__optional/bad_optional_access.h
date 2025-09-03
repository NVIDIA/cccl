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
#  ifdef __cpp_lib_optional
#    include <optional>
#  else // ^^^ __cpp_lib_optional ^^^ / vvv !__cpp_lib_optional vvv
#    include <exception>
#  endif // !__cpp_lib_optional
#endif // _CCCL_HAS_EXCEPTIONS()

#include <cuda/std/__exception/terminate.h>

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_HAS_EXCEPTIONS()
_CCCL_BEGIN_NAMESPACE_CUDA_STD_NOVERSION

#  ifdef __cpp_lib_optional

using ::std::bad_optional_access;

#  else // ^^^ __cpp_lib_optional ^^^ / vvv !__cpp_lib_optional vvv
class _CCCL_TYPE_VISIBILITY_DEFAULT bad_optional_access : public ::std::exception
{
public:
  const char* what() const noexcept override
  {
    return "bad access to cuda::std::optional";
  }
};
#  endif // !__cpp_lib_optional

_CCCL_END_NAMESPACE_CUDA_STD_NOVERSION
#endif // _CCCL_HAS_EXCEPTIONS()

_CCCL_BEGIN_NAMESPACE_CUDA_STD

[[noreturn]] _CCCL_API inline void __throw_bad_optional_access()
{
#if _CCCL_HAS_EXCEPTIONS()
  NV_IF_ELSE_TARGET(NV_IS_HOST, (throw ::cuda::std::bad_optional_access();), (::cuda::std::terminate();))
#else // ^^^ !_CCCL_HAS_EXCEPTIONS() ^^^ / vvv _CCCL_HAS_EXCEPTIONS() vvv
  ::cuda::std::terminate();
#endif // _CCCL_HAS_EXCEPTIONS()
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___OPTIONAL_BAD_OPTIONAL_ACCESS_H
