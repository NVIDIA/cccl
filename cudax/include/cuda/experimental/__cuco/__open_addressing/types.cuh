//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO___OPEN_ADDRESSING_TYPES_CUH
#define _CUDAX___CUCO___OPEN_ADDRESSING_TYPES_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__cuco/__utility/strong_type.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::__open_addressing
{
//! @brief Strong type wrapper for an empty key sentinel.
//!
//! @tparam _Key The key type
template <class _Key>
struct __empty_key : public ::cuda::experimental::cuco::__strong_type<_Key>
{
  _CCCL_API explicit constexpr __empty_key(_Key __value)
      : ::cuda::experimental::cuco::__strong_type<_Key>(__value)
  {}
};

//! @brief Strong type wrapper for an empty value sentinel.
//!
//! @tparam _T The mapped value type
template <class _T>
struct __empty_value : public ::cuda::experimental::cuco::__strong_type<_T>
{
  _CCCL_API explicit constexpr __empty_value(_T __value)
      : ::cuda::experimental::cuco::__strong_type<_T>(__value)
  {}
};

//! @brief Strong type wrapper for an erased key sentinel.
//!
//! @tparam _Key The key type
template <class _Key>
struct __erased_key : public ::cuda::experimental::cuco::__strong_type<_Key>
{
  _CCCL_API explicit constexpr __erased_key(_Key __value)
      : ::cuda::experimental::cuco::__strong_type<_Key>(__value)
  {}
};
} // namespace cuda::experimental::cuco::__open_addressing

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO___OPEN_ADDRESSING_TYPES_CUH
