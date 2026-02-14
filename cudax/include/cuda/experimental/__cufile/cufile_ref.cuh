//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>

#include <cufile.h>

namespace cuda::experimental
{
using __cufile_os_native_type = int;

//! @brief A non-owning wrapper of \c CUfileHandle_t.
class cufile_ref
{
protected:
  ::CUfileHandle_t __cufile_handle_{}; //!< The cuFile file handle.

  _CCCL_HIDE_FROM_ABI cufile_ref() noexcept = default;

public:
  using off_type = ::off_t;

  //! @brief Constructs the object from a \c CUfileHandle_t handle.
  _CCCL_HOST_API cufile_ref(::CUfileHandle_t __cufile_handle) noexcept
      : __cufile_handle_{__cufile_handle}
  {}

  //! @brief Disallow construction from nullptr.
  cufile_ref(::cuda::std::nullptr_t) = delete;

  _CCCL_HIDE_FROM_ABI cufile_ref(const cufile_ref&) noexcept = default;

  _CCCL_HIDE_FROM_ABI cufile_ref& operator=(const cufile_ref&) noexcept = default;

  //! @brief Retrieve the \c CUfileHandle_t handle.
  //!
  //! @returns The handle being held by the object.
  [[nodiscard]] _CCCL_HOST_API ::CUfileHandle_t get() const noexcept
  {
    return __cufile_handle_;
  }
};
} // namespace cuda::experimental
