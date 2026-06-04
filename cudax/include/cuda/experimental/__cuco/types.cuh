//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO_TYPES_CUH
#define _CUDAX___CUCO_TYPES_CUH

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

namespace cuda::experimental::cuco
{
//! @brief Strong type wrapper for an empty key sentinel.
//!
//! @tparam _Key The key type
template <class _Key>
struct empty_key : public ::cuda::experimental::cuco::__strong_type<_Key>
{
  _CCCL_HOST_DEVICE_API explicit constexpr empty_key(_Key __value) noexcept
      : ::cuda::experimental::cuco::__strong_type<_Key>(__value)
  {}
};

//! @brief Strong type wrapper for an empty value sentinel.
//!
//! @tparam _Tp The mapped value type
template <class _Tp>
struct empty_value : public ::cuda::experimental::cuco::__strong_type<_Tp>
{
  _CCCL_HOST_DEVICE_API explicit constexpr empty_value(_Tp __value) noexcept
      : ::cuda::experimental::cuco::__strong_type<_Tp>(__value)
  {}
};

//! @brief Strong type wrapper for an erased key sentinel.
//!
//! @tparam _Key The key type
template <class _Key>
struct erased_key : public ::cuda::experimental::cuco::__strong_type<_Key>
{
  _CCCL_HOST_DEVICE_API explicit constexpr erased_key(_Key __value) noexcept
      : ::cuda::experimental::cuco::__strong_type<_Key>(__value)
  {}
};
} // namespace cuda::experimental::cuco

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_TYPES_CUH
