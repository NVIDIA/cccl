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

#include <cuda/std/__utility/swap.h>

#include <cufile.h>

namespace cuda::experimental
{

class cufile_ref
{
public:
  using off_type           = ::off_t;
  using native_handle_type = int;

protected:
  static constexpr native_handle_type __invalid_native_handle = -1;

public:
  cufile_ref(const cufile_ref&) noexcept = default;

  cufile_ref& operator=(const cufile_ref&) noexcept = default;

  //! @brief Queries whether the file is opened.
  //!
  //! @return True, if opened, false otherwise.
  [[nodiscard]] bool is_open() const noexcept
  {
    return __native_handle_ != __invalid_native_handle;
  }

  //! @brief Gets the OS native handle.
  //!
  //! @return The native handle.
  [[nodiscard]] native_handle_type native_handle() const noexcept
  {
    return __native_handle_;
  }

  //! @brief Swaps contents of two cufile references.
  //!
  //! @param __lhs First instance.
  //! @param __rhs Second instance.
  [[nodiscard]] friend void swap(cufile_ref& __lhs, cufile_ref& __rhs) noexcept
  {
    ::cuda::std::swap(__lhs.__native_handle_, __rhs.__native_handle_);
    ::cuda::std::swap(__lhs.__cufile_handle_, __rhs.__cufile_handle_);
  }

protected:
  cufile_ref() noexcept = default;

  cufile_ref(native_handle_type __native_handle, ::CUfileHandle_t __cufile_handle) noexcept
      : __native_handle_{__native_handle}
      , __cufile_handle_{__cufile_handle_}
  {}

  native_handle_type __native_handle_{__invalid_native_handle};
  ::CUfileHandle_t __cufile_handle_{};
};

} // namespace cuda::experimental
