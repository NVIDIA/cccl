//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_IS_VALID_ALIGNMENT_H
#define _CUDA___MEMORY_IS_VALID_ALIGNMENT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/pow2.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/is_void.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <class _Tp = void>
[[nodiscard]] _CCCL_API constexpr bool __is_valid_alignment(::cuda::std::size_t __alignment) noexcept
{
  if constexpr (::cuda::std::is_void_v<_Tp>)
  {
    return __alignment > 0 && ::cuda::is_power_of_two(__alignment);
  }
  else
  {
    return __alignment >= alignof(_Tp) && ::cuda::is_power_of_two(__alignment);
  }
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MEMORY_IS_VALID_ALIGNMENT_H
