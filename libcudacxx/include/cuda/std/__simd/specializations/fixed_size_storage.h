//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_SPECIALIZATIONS_FIXED_SIZE_STORAGE_H
#define _CUDA_STD___SIMD_SPECIALIZATIONS_FIXED_SIZE_STORAGE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/in_range.h>
#include <cuda/std/__fwd/simd.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

template <__simd_size_type _Np>
struct __fixed_size
{
  static_assert(_Np > 0, "_Np must be greater than 0");

  static constexpr __simd_size_type __simd_size = _Np;
};

// Element-per-slot simd storage for fixed_size ABI
template <typename _Tp, __simd_size_type _Np>
struct __simd_storage<_Tp, __fixed_size<_Np>>
{
  using value_type = _Tp;

  _Tp __data[_Np]{};

  [[nodiscard]] _CCCL_API constexpr _Tp __get(const __simd_size_type __idx) const noexcept
  {
    _CCCL_ASSERT(::cuda::in_range(__idx, __simd_size_type{0}, _Np), "Index is out of bounds");
    return __data[__idx];
  }

  _CCCL_API constexpr void __set(const __simd_size_type __idx, const _Tp __v) noexcept
  {
    _CCCL_ASSERT(::cuda::in_range(__idx, __simd_size_type{0}, _Np), "Index is out of bounds");
    __data[__idx] = __v;
  }
};

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_SPECIALIZATIONS_FIXED_SIZE_STORAGE_H
