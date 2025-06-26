//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_PTR_CAST_H
#define _CUDA___MEMORY_PTR_CAST_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <typename _Up, typename _Tp>
[[nodiscard]] _CCCL_API _Up* ptr_cast(_Tp* __ptr) noexcept
{
  constexpr auto __max_alignment = alignof(_Tp) > alignof(_Up) ? alignof(_Tp) : alignof(_Up);
  _CCCL_ASSERT(reinterpret_cast<_CUDA_VSTD::uintptr_t>(__ptr) % __max_alignment == 0, "ptr is not aligned");
  return reinterpret_cast<_Up*>(_CCCL_BUILTIN_ASSUME_ALIGNED(__ptr, __max_alignment));
}

template <typename _Up, typename _Tp>
[[nodiscard]] _CCCL_API inline const _Up* ptr_cast(const _Tp* __ptr) noexcept
{
  return ::cuda::ptr_cast<const _Up>(__ptr);
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MEMORY_PTR_CAST_H
