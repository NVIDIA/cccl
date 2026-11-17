//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO_BLOOM_FILTER_MODES_CUH
#define _CUDAX___CUCO_BLOOM_FILTER_MODES_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco
{
//! @brief Selects whether `bloom_filter::add` reads a word before issuing the atomic OR.
enum class conditional_add_mode : bool
{
  off = false, ///< Always issue the atomic OR
  on  = true ///< Skip the atomic OR when the required bits are already set
};

//! @brief Selects whether `bloom_filter::contains` short-circuits on a missing fingerprint slice.
enum class early_exit_contains_mode : bool
{
  off = false, ///< Always evaluate every fingerprint slice
  on  = true ///< Return as soon as a slice misses
};
} // namespace cuda::experimental::cuco

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_BLOOM_FILTER_MODES_CUH
