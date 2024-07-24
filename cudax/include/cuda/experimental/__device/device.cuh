//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__DEVICE_DEVICE
#define _CUDAX__DEVICE_DEVICE

#include <cuda/std/detail/__config>
#include <cuda_runtime_api.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

namespace cuda::experimental
{
// Dummy struct for now to be able to reference it in other places
// TODO this might be device_ref instead
// TODO proper implementation
//! @brief A non-owning representation of a CUDA device
struct device
{
  int __id = 0;

  //! @brief Retrieve the native ordinal of the device
  //!
  //! @return int The native device ordinal held by the device object
  _CCCL_NODISCARD constexpr int get() const noexcept
  {
    return __id;
  }
};

} // namespace cuda::experimental
#endif
