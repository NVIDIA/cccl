//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___VARIANT_BAD_VARIANT_ACCESS_H
#define _CUDA_STD___VARIANT_BAD_VARIANT_ACCESS_H

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

#  ifdef __cpp_lib_variant
#    include <variant>
#  else // ^^^ __cpp_lib_variant ^^^ / vvv !__cpp_lib_variant vvv
#    include <exception>
#  endif // !__cpp_lib_variant
#endif // _CCCL_HAS_EXCEPTIONS()

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_HAS_EXCEPTIONS()

_CCCL_BEGIN_NAMESPACE_CUDA_STD_NOVERSION

#  ifdef __cpp_lib_variant

using ::std::bad_variant_access;

#  else // ^^^ __cpp_lib_variant ^^^ / vvv !__cpp_lib_variant vvv
class _CCCL_TYPE_VISIBILITY_DEFAULT bad_variant_access : public ::std::exception
{
public:
  const char* what() const noexcept override
  {
    return "bad access to cuda::std::variant";
  }
};
#  endif // !__cpp_lib_variant

_CCCL_END_NAMESPACE_CUDA_STD_NOVERSION

#endif // _CCCL_HAS_EXCEPTIONS()

_CCCL_BEGIN_NAMESPACE_CUDA_STD

[[noreturn]] _CCCL_API inline void __throw_bad_variant_access()
{
#if _CCCL_HAS_EXCEPTIONS()
  NV_IF_ELSE_TARGET(NV_IS_HOST, (throw ::cuda::std::bad_variant_access();), (::cuda::std::terminate();))
#else // ^^^ !_CCCL_HAS_EXCEPTIONS() ^^^ / vvv _CCCL_HAS_EXCEPTIONS() vvv
  ::cuda::std::terminate();
#endif // _CCCL_HAS_EXCEPTIONS()
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___VARIANT_BAD_VARIANT_ACCESS_H
