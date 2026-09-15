//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___COMPLEX_TUPLE_H
#define _CUDA___COMPLEX_TUPLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__complex/complex.h>
#include <cuda/std/__fwd/get.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <size_t _Index, class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp& get(::cuda::complex<_Tp>& __z) noexcept
{
  static_assert(_Index < 2, "Index value is out of range");
  return (_Index == 0) ? __z.__re_ : __z.__im_;
}

template <size_t _Index, class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp&& get(::cuda::complex<_Tp>&& __z) noexcept
{
  static_assert(_Index < 2, "Index value is out of range");
  return ::cuda::std::move((_Index == 0) ? __z.__re_ : __z.__im_);
}

template <size_t _Index, class _Tp>
[[nodiscard]] _CCCL_API constexpr const _Tp& get(const ::cuda::complex<_Tp>& __z) noexcept
{
  static_assert(_Index < 2, "Index value is out of range");
  return (_Index == 0) ? __z.__re_ : __z.__im_;
}

template <size_t _Index, class _Tp>
[[nodiscard]] _CCCL_API constexpr const _Tp&& get(const ::cuda::complex<_Tp>&& __z) noexcept
{
  static_assert(_Index < 2, "Index value is out of range");
  return ::cuda::std::move((_Index == 0) ? __z.__re_ : __z.__im_);
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___COMPLEX_TUPLE_H
