//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___COMPLEX_GET_REAL_IMAG_H
#define _CUDA___COMPLEX_GET_REAL_IMAG_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__fwd/complex.h>
#include <cuda/std/__fwd/complex.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __get_real(const complex<_Tp>& __c) noexcept
{
  return __c.real();
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __get_imag(const complex<_Tp>& __c) noexcept
{
  return __c.imag();
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __get_real(const ::cuda::std::complex<_Tp>& __c) noexcept
{
  return __c.real();
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __get_imag(const ::cuda::std::complex<_Tp>& __c) noexcept
{
  return __c.imag();
}

#if !_CCCL_COMPILER(NVRTC)

// Unless `--expt-relaxed-constexpr` is specified, obtaining values from std::complex is not constexpr :(
#  if defined(__CUDACC_RELAXED_CONSTEXPR__)
template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __get_real(const ::std::complex<_Tp>& __c) noexcept
{
  return __c.real();
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __get_imag(const ::std::complex<_Tp>& __c) noexcept
{
  return __c.imag();
}
#  else // ^^^ __CUDACC_RELAXED_CONSTEXPR__ ^^^ / vvv !__CUDACC_RELAXED_CONSTEXPR__ vvv
template <class _Tp>
[[nodiscard]] _CCCL_API _Tp __get_real(const ::std::complex<_Tp>& __c) noexcept
{
  return reinterpret_cast<const _Tp(&)[2]>(__c)[0];
}

template <class _Tp>
[[nodiscard]] _CCCL_API _Tp __get_imag(const ::std::complex<_Tp>& __c) noexcept
{
  return reinterpret_cast<const _Tp(&)[2]>(__c)[1];
}
#  endif // ^^^ !__CUDACC_RELAXED_CONSTEXPR__ ^^^
#endif // !_CCCL_COMPILER(NVRTC)

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___COMPLEX_GET_REAL_IMAG_H
