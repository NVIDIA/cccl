//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_UNINITIALIZED_ARRAY_H
#define _CUDA___MEMORY_UNINITIALIZED_ARRAY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__new/launder.h>
#include <cuda/std/__type_traits/is_trivially_copyable.h>
#include <cuda/std/__type_traits/is_trivially_move_constructible.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <class _Tp, size_t _Size, size_t _Alignment = alignof(_Tp)>
struct uninitialized_array
{
  static_assert(::cuda::std::is_trivially_copyable<_Tp>::value
                  && ::cuda::std::is_trivially_move_constructible<_Tp>::value,
                "uninitialized_array requires trivially copyable and trivially move constructible types");

  alignas(_Alignment) unsigned char __data[_Size * sizeof(_Tp)];

  [[nodiscard]] _CCCL_API _Tp* data() noexcept
  {
    return ::cuda::std::launder(reinterpret_cast<_Tp*>(__data));
  }

  [[nodiscard]] _CCCL_API const _Tp* data() const noexcept
  {
    return ::cuda::std::launder(reinterpret_cast<const _Tp*>(__data));
  }

  [[nodiscard]] _CCCL_API _Tp& operator[](const size_t __idx) noexcept
  {
    _CCCL_ASSERT(__idx < _Size, "out of bounds access in uninitialized_array::operator[]");
    return data()[__idx];
  }

  [[nodiscard]] _CCCL_API const _Tp& operator[](const size_t __idx) const noexcept
  {
    _CCCL_ASSERT(__idx < _Size, "out of bounds access in uninitialized_array::operator[]");
    return data()[__idx];
  }
};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MEMORY_UNINITIALIZED_ARRAY_H
