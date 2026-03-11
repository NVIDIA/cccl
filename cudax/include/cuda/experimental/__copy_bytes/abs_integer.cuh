//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_COPY_ABS_INTEGER_H
#define __CUDAX_COPY_ABS_INTEGER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__cstdlib/abs.h>
#  include <cuda/std/__type_traits/is_integer.h>
#  include <cuda/std/__type_traits/is_signed.h>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief Returns the absolute value of an integer. Identity for unsigned types.
//!
//! @param[in] __value Integer value
//! @return Absolute value of @p __value
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Tp __abs_integer(_Tp __value) noexcept
{
  if constexpr (::cuda::std::is_signed_v<_Tp>)
  {
    return ::cuda::std::abs(__value);
  }
  else
  {
    return __value;
  }
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // __CUDAX_COPY_ABS_INTEGER_H
