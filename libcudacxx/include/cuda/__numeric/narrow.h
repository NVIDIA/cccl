//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___NARROW_NARROW_H
#define _CUDA___NARROW_NARROW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)
#  include <stdexcept>
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_signed.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

#if !_CCCL_COMPILER(NVRTC)
struct narrowing_error : ::std::runtime_error
{
  narrowing_error()
      : ::std::runtime_error("Narrowing error")
  {}
};
#endif // !_CCCL_COMPILER(NVRTC)

//! Casts a value \p __from to type \p _To and checks whether the value has changed. Throws in host code and traps in
//! device code. Modelled after `gsl::narrow`.
template <class _To, class _From>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr _To narrow(_From __from)
{
  const auto __converted = static_cast<_To>(__from);
  if constexpr (_CUDA_VSTD::is_arithmetic_v<_From>)
  {
    constexpr bool __is_different_signedness = _CUDA_VSTD::is_signed_v<_From> != _CUDA_VSTD::is_signed_v<_To>;
    _CCCL_NV_DIAG_SUPPRESS(186) // pointless comparison of unsigned integer with zero
    if (static_cast<_From>(__converted) != __from
        || (__is_different_signedness && (__converted < _To{} != __from < _From{})))
    {
      NV_IF_TARGET(NV_IS_HOST, throw narrowing_error{};, __trap(););
    }
    _CCCL_NV_DIAG_DEFAULT()
  }
  else if (static_cast<_From>(__converted) != __from)
  {
    NV_IF_TARGET(NV_IS_HOST, throw narrowing_error{};, __trap(););
  }
  return __converted;
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___NARROW_NARROW_H
