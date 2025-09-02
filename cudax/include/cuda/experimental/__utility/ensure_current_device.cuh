//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__UTILITY_ENSURE_CURRENT_DEVICE_CUH
#define _CUDAX__UTILITY_ENSURE_CURRENT_DEVICE_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__runtime/ensure_current_context.h>

#include <cuda/experimental/__device/logical_device.cuh>
#include <cuda/experimental/__graph/concepts.cuh>

#include <cuda/std/__cccl/prologue.h>

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

namespace cuda::experimental
{
//! TODO we might want to change the comments to indicate it operates on contexts for certains differences
//! with green context, but it depends on how exactly green context internals end up being

//! @brief RAII helper which on construction sets the current device to the specified one or one a
//! stream was created under. It sets the state back on destruction.
//!
struct [[maybe_unused]] __ensure_current_device : ::cuda::__ensure_current_context
{
  using __ensure_current_context::__ensure_current_context;

  //! @brief Construct a new `__ensure_current_device` object and switch to the specified
  //!        device.
  //!
  //! Note: if this logical device contains a green_context the device under which the green
  //! context was created will be set to current
  //!
  //! @param new_device The device to switch to
  //!
  //! @throws cuda_error if the device switch fails
  explicit __ensure_current_device(logical_device __new_device)
      : __ensure_current_context(__new_device.context())
  {}

  _CCCL_TEMPLATE(typename _GraphInserter)
  _CCCL_REQUIRES(graph_inserter<_GraphInserter>)
  explicit __ensure_current_device(const _GraphInserter& __inserter)
      : __ensure_current_device(__inserter.get_device())
  {}
};
} // namespace cuda::experimental
#endif // _CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__UTILITY_ENSURE_CURRENT_DEVICE_CUH
