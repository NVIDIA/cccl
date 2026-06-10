//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___LIBRARY_LIBRARY_CUH
#define _CUDAX___LIBRARY_LIBRARY_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__driver/driver_api.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__exception/cuda_error.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__utility/exchange.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/swap.h>

#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__library/library_ref.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief An owning wrapper for a CUDA library handle
struct library : public library_ref
{
  //! @brief Construct an `library` object from a native `CUlibrary`/`cudaLibrary_t` handle
  //!
  //! @param __handle The native handle
  //!
  //! @return The constructed `library` object
  //!
  //! @note The constructed `library` object takes ownership of the native handle
  [[nodiscard]] static library from_native_handle(value_type __handle) noexcept
  {
    return library{__handle};
  }

  //! @brief Disallow construction from a null pointer
  static library from_native_handle(::cuda::std::nullptr_t) = delete;

  //! @brief Construct a new `library` object into the moved-from state
  //!
  //! @post `get()` will return an invalid `CUlibrary` handle
  explicit constexpr library(no_init_t) noexcept
      : library{value_type{}}
  {}

  library(const library&) = delete;

  //! @brief Move-construct a new 'library' object
  //!
  //! @param __other The `library` to move from
  //!
  //! @post `__other` is in the moved-from state
  library(library&& __other) noexcept
      : library{__other.release()}
  {}

  //! @brief Destroy the `library` object
  //!
  //! @note If the library fails to unload, the error is silently ignored
  ~library()
  {
    if (__library_ != value_type{})
    {
      [[maybe_unused]] const auto __status = ::cuda::__driver::__libraryUnloadNoThrow(__library_);
    }
  }

  library& operator=(const library&) = delete;

  //! @brief Move-assign a new `library` object
  //!
  //! @param __other The `library` to move from
  //!
  //! @post `__other` is in the moved-from state
  library& operator=(library&& __other) noexcept
  {
    if (this != ::cuda::std::addressof(__other))
    {
      library __tmp{::cuda::std::move(__other)};
      ::cuda::std::swap(__library_, __tmp.__library_);
    }
    return *this;
  }

  //! @brief Retrieve the native `CUlibrary`/`cudaLibrary_t` handle and give up ownership
  //!
  //! @return The native handle being held by the `library` object
  //!
  //! @post The library object is in a moved-from state
  [[nodiscard]] constexpr value_type release() noexcept
  {
    return ::cuda::std::exchange(__library_, value_type{});
  }

private:
  constexpr explicit library(value_type __handle) noexcept
      : library_ref{__handle}
  {}
};
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___LIBRARY_LIBRARY_CUH
