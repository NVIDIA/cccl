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

#include <cuda/std/__utility/exchange.h>
#include <cuda/std/__utility/swap.h>
#include <cuda/std/detail/libcxx/include/stdexcept>

#include <cuda/experimental/__cufile/cufile_ref.cuh>
#include <cuda/experimental/__cufile/exception.cuh>
#include <cuda/experimental/__cufile/open_mode.cuh>

#include <cufile.h>
#include <fcntl.h>
#include <unistd.h>

namespace cuda::experimental
{

class cufile : public cufile_ref
{
  [[nodiscard]] static constexpr int __make_oflags(cufile_open_mode __om) noexcept
  {
    int __ret = O_CREAT | O_DIRECT;

    if ((__om & (cufile_open_mode::in | cufile_open_mode::out)) == (cufile_open_mode::in | cufile_open_mode::out))
    {
      __ret |= O_RDWR;
    }
    else if ((__om & cufile_open_mode::in) == cufile_open_mode::in)
    {
      __ret |= O_RDONLY;
    }
    else if ((__om & cufile_open_mode::out) == cufile_open_mode::out)
    {
      __ret |= O_WRONLY;
    }

    __ret |= ((__om & cufile_open_mode::trunc) == cufile_open_mode::trunc) ? O_TRUNC : 0;
    __ret |= ((__om & cufile_open_mode::noreplace) == cufile_open_mode::noreplace) ? O_EXCL : 0;
    return __ret;
  }

  [[nodiscard]] static native_handle_type __open_file(const char* __filename, cufile_open_mode __open_mode)
  {
    int __fd = ::open(__filename, __make_oflags(__open_mode));

    if (__fd == -1)
    {
      ::cuda::std::__throw_runtime_error("Failed to open file.");
    }

    return __fd;
  }

  [[nodiscard]] static ::CUfileHandle_t __register_cufile_handle(native_handle_type __native_handle)
  {
    ::CUfileDescr_t __desc{};
    __desc.type      = ::CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    __desc.handle.fd = __native_handle;

    ::CUfileHandle_t __handle{};
    _CCCL_TRY_CUFILE_API(::cuFileHandleRegister, "Failed to register cuFile handle", &__handle, &__desc);
    return __handle;
  }

  [[nodiscard]] static bool __close_file_no_throw(native_handle_type __native_handle) noexcept
  {
    return ::close(__native_handle) == 0;
  }

  static void __close_file(native_handle_type __native_handle)
  {
    if (!__close_file_no_throw(__native_handle))
    {
      ::cuda::std::__throw_runtime_error("Failed to close file.");
    }
  }

public:
  //! @brief Make a cufile object from already existing native handle.
  //!
  //! @param __native_handle The native handle.
  //!
  //! @return The created cufile object.
  [[nodiscard]] static cufile from_native_handle(native_handle_type __native_handle)
  {
    _CCCL_ASSERT(__native_handle != __invalid_native_handle, "invalid native handle");

    ::CUfileHandle_t __cufile_handle = __register_cufile_handle(__native_handle);

    cufile __ret{};
    __ret.__native_handle_ = __native_handle;
    __ret.__cufile_handle_ = __cufile_handle;
    return __ret;
  }

  cufile() noexcept = default;

  //! @brief Constructs the object by opening file @c __filename in mode @c __open_mode.
  //!
  //! @param __filename Path to the file.
  //! @param __open_mode Open mode to open the file with.
  //!
  //! @throws cuda::std::runtime_error if the file cannot be opened.
  //! @throws cuda::cuda_error if a CUDA driver error occurs.
  //! @throws cuda::cufile_error if a cuFile driver error occurs.
  cufile(const char* __filename, cufile_open_mode __open_mode)
      : cufile{}
  {
    __native_handle_ = __open_file(__filename, __open_mode);
    ::CUfileHandle_t __cufile_handle{};

    try
    {
      __cufile_handle_ = __register_cufile_handle(__native_handle_);
    }
    catch (...)
    {
      __close_file(__native_handle_);
      throw;
    }
  }

  cufile(const cufile&) = delete;

  //! @brief Move-construct a new @c cufile.
  //!
  //! @param __other The other @c cufile.
  //!
  //! @post `__other` is in moved-from state.
  cufile(cufile&& __other) noexcept
      : cufile_ref{::cuda::std::exchange(__other.__native_handle_, __invalid_native_handle),
                   ::cuda::std::exchange(__other.__cufile_handle_, nullptr)}
  {}

  cufile& operator=(const cufile&) = delete;

  //! @brief Move-assign from a @c cufile object.
  //!
  //! @param __other The other @c cufile.
  //!
  //! @post `__other` is in moved-from state.
  //!
  //! @throws cuda::std::runtime_error if the currently opened file fails to close.
  cufile& operator=(cufile&& __other)
  {
    if (this != ::cuda::std::addressof(__other))
    {
      close();
      __native_handle_ = ::cuda::std::exchange(__other.__native_handle_, __invalid_native_handle);
      __cufile_handle_ = ::cuda::std::exchange(__other.__cufile_handle_, nullptr);
    }
  }

  ~cufile()
  {
    if (is_open())
    {
      ::cuFileHandleDeregister(__cufile_handle_);
      [[maybe_unused]] const auto __ignore_close_retval = __close_file_no_throw(__native_handle_);
    }
  }

  //! @brief Opens file @c __filename in mode @c __open_mode.
  //!
  //! @param __filename Path to the file.
  //! @param __open_mode Open mode to open the file with.
  //!
  //! @throws cuda::std::runtime_error if the file cannot be opened or if a file is already opened.
  //! @throws cuda::cuda_error if a CUDA driver error occurs.
  //! @throws cuda::cufile_error if a cuFile driver error occurs.
  void open(const char* __filename, cufile_open_mode __open_mode)
  {
    if (is_open())
    {
      ::cuda::std::__throw_runtime_error("File is already opened.");
    }

    __native_handle_ = __open_file(__filename, __open_mode);

    try
    {
      __cufile_handle_ = __register_cufile_handle(__native_handle_);
    }
    catch (...)
    {
      __close_file(::cuda::std::exchange(__native_handle_, __invalid_native_handle));
      throw;
    }
  }

  //! @brief Closes the currently opened file. If there is no opened file, no action is taken.
  //!
  //! @throws cuda::std::runtime_error if the file fails to close.
  //! @throws cuda::cuda_error if a CUDA driver error occurs.
  //! @throws cuda::cufile_error if a cuFile driver error occurs.
  void close()
  {
    if (!is_open())
    {
      return;
    }

    ::cuFileHandleDeregister(::cuda::std::exchange(__cufile_handle_, nullptr));
    __close_file(::cuda::std::exchange(__native_handle_, __invalid_native_handle));
  }

  //! @brief Swaps contents of two cufiles.
  //!
  //! @param __lhs First instance.
  //! @param __rhs Second instance.
  [[nodiscard]] friend void swap(cufile& __lhs, cufile& __rhs) noexcept
  {
    ::cuda::std::swap(__lhs.__native_handle_, __rhs.__native_handle_);
    ::cuda::std::swap(__lhs.__cufile_handle_, __rhs.__cufile_handle_);
  }
};

} // namespace cuda::experimental
