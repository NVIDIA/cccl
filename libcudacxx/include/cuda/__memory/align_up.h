//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_ALIGN_UP_H
#define _CUDA___MEMORY_ALIGN_UP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/pow2.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <typename _Tp>
[[nodiscard]] _CCCL_API _Tp* align_up(_Tp* __ptr, _CUDA_VSTD::size_t __alignment) noexcept
{
  _CCCL_ASSERT(::cuda::is_power_of_two(__alignment), "alignment must be a power of two");
  _CCCL_ASSERT(__alignment >= alignof(_Tp), "wrong alignment");
  _CCCL_ASSERT(reinterpret_cast<_CUDA_VSTD::uintptr_t>(__ptr) % alignof(_Tp) == 0, "ptr is not aligned");
  auto __tmp = static_cast<_CUDA_VSTD::uintptr_t>(__alignment - 1);
  auto __ret = reinterpret_cast<_Tp*>((reinterpret_cast<_CUDA_VSTD::uintptr_t>(__ptr) + __tmp) & ~__tmp);
#if defined(_CCCL_BUILTIN_ASSUME_ALIGNED)
  switch (__alignment)
  {
    case 1:
      return static_cast<_Tp*>(_CCCL_BUILTIN_ASSUME_ALIGNED(__ret, 1));
    case 2:
      return static_cast<_Tp*>(_CCCL_BUILTIN_ASSUME_ALIGNED(__ret, 2));
    case 4:
      return static_cast<_Tp*>(_CCCL_BUILTIN_ASSUME_ALIGNED(__ret, 4));
    case 8:
      return static_cast<_Tp*>(_CCCL_BUILTIN_ASSUME_ALIGNED(__ret, 8));
    case 16:
      return static_cast<_Tp*>(_CCCL_BUILTIN_ASSUME_ALIGNED(__ret, 16));
    default:
      return static_cast<_Tp*>(_CCCL_BUILTIN_ASSUME_ALIGNED(__ret, 32));
  }
#else
  return __ret;
#endif // defined(_CCCL_BUILTIN_ASSUME_ALIGNED)
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MEMORY_ALIGN_UP_H
