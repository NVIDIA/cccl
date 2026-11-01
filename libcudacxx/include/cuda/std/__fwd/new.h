//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FWD_NEW_H
#define _CUDA_STD___FWD_NEW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>

#include <cuda/std/__cccl/prologue.h>

// std:: forward declarations

#if _CCCL_HAS_HOST_STD_LIB()

// libc++ puts align_val_t to unversioned namespace. Since libc++-21, they use _LIBCPP_BEGIN_UNVERSIONED_NAMESPACE_STD,
// before that it was just plain `namespace std`.
#  if _CCCL_HOST_STD_LIB(LIBCXX)
#    if _CCCL_HOST_STD_LIB(LIBCXX, >=, 21)
_LIBCPP_BEGIN_UNVERSIONED_NAMESPACE_STD
#    else // ^^^ _CCCL_HOST_STD_LIB(LIBCXX, >=, 21) ^^^ / vvv _CCCL_HOST_STD_LIB(LIBCXX, <, 21) vvv
namespace std
{
#    endif // ^^^ _CCCL_HOST_STD_LIB(LIBCXX, <, 21) ^^^
#  else // ^^^ _CCCL_HOST_STD_LIB(LIBCXX) ^^^ / vvv !_CCCL_HOST_STD_LIB(LIBCXX) vvv
_CCCL_BEGIN_NAMESPACE_STD
#  endif // ^^^ !_CCCL_HOST_STD_LIB(LIBCXX) ^^^

// We can always forward declare align_val_t, because we know its underlying type.
enum class align_val_t : size_t;

#  if _CCCL_HOST_STD_LIB(LIBCXX)
#    if _CCCL_HOST_STD_LIB(LIBCXX, >=, 21)
_LIBCPP_END_UNVERSIONED_NAMESPACE_STD
#    else // ^^^ _CCCL_HOST_STD_LIB(LIBCXX, >=, 21) ^^^ / vvv _CCCL_HOST_STD_LIB(LIBCXX, <, 21) vvv
} // namespace std
#    endif // ^^^ _CCCL_HOST_STD_LIB(LIBCXX, <, 21) ^^^
#  else // ^^^ _CCCL_HOST_STD_LIB(LIBCXX) ^^^ / vvv !_CCCL_HOST_STD_LIB(LIBCXX) vvv
_CCCL_END_NAMESPACE_STD
#  endif // ^^^ !_CCCL_HOST_STD_LIB(LIBCXX) ^^^
#endif // _CCCL_HAS_HOST_STD_LIB()

// cuda::std:: forward declarations

_CCCL_BEGIN_NAMESPACE_CUDA_STD

using ::std::align_val_t;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FWD_NEW_H
