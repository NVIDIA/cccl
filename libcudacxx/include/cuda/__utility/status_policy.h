//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___UTILITY_STATUS_POLICY_H
#define _CUDA___UTILITY_STATUS_POLICY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief Tag requesting a status object from a status-capable operation.
struct _CCCL_TYPE_VISIBILITY_DEFAULT return_status_t
{
  _CCCL_HIDE_FROM_ABI explicit return_status_t() = default;
};

//! @brief Tag requesting that a status-capable operation ignore any produced status report.
struct _CCCL_TYPE_VISIBILITY_DEFAULT ignore_status_t
{
  _CCCL_HIDE_FROM_ABI explicit ignore_status_t() = default;
};

//! @brief Selects how an operation status report should be interpreted.
enum class status_source
{
  //! Generic fabric report payload produced through an mbarrier layout-v1 status report.
  generic_fabric,
};

//! @brief Recommended action for a classified status report.
enum class status_action
{
  //! Retry the operation that produced the status report.
  retry,
  //! Abort the operation that produced the status report.
  abort,
};

//! @brief Constant of type `cuda::return_status_t`.
_CCCL_GLOBAL_CONSTANT return_status_t return_status{};
//! @brief Constant of type `cuda::ignore_status_t`.
_CCCL_GLOBAL_CONSTANT ignore_status_t ignore_status{};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___UTILITY_STATUS_POLICY_H
