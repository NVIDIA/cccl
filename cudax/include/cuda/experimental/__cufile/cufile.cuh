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

#include <cuda/std/__exception/throw_error.h>
#include <cuda/std/__utility/exchange.h>
#include <cuda/std/string_view>

#include <cuda/experimental/__cufile/cufile_ref.cuh>
#include <cuda/experimental/__cufile/driver.cuh>
#include <cuda/experimental/__cufile/exception.cuh>
#include <cuda/experimental/__cufile/open_mode.cuh>

#include <string>

#include <cufile.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

namespace cuda::experimental
{
//! @brief An owning wrapper of \c CUfileHandle_t and the OS specific native file handle.
class cufile : public cufile_ref
{
public:
  using native_handle_type = __cufile_os_native_type; //!< The underlying OS native handle type.

private:
  using __oflags_type = int;

  static constexpr native_handle_type __invalid_native_handle = -1;

  native_handle_type __native_handle_{__invalid_native_handle}; //< The native handle.

  //! @brief Constructs the object from native handle and cuFile file handle.
  _CCCL_HIDE_FROM_ABI cufile(cufile_ref __cufile_handle, native_handle_type __native_handle) noexcept
      : cufile_ref{__cufile_handle}
      , __native_handle_{__native_handle}
  {}

  //! @brief Make open flags from the \c cuda::cufile_open_mode.
  //!
  //! @param __om The cuFile open mode.
  //!
  //! @return The flags mask to be passed to open function.
  [[nodiscard]] static _CCCL_HOST_API constexpr __oflags_type __make_oflags(cufile_open_mode __om) noexcept
  {
    __oflags_type __ret{};
    if ((__om & (cufile_open_mode::in | cufile_open_mode::out)) == (cufile_open_mode::in | cufile_open_mode::out))
    {
      __ret |= O_RDWR | O_CREAT;
    }
    else if ((__om & cufile_open_mode::in) == cufile_open_mode::in)
    {
      __ret |= O_RDONLY;
    }
    else if ((__om & cufile_open_mode::out) == cufile_open_mode::out)
    {
      __ret |= O_WRONLY | O_CREAT;
    }

    __ret |= ((__om & cufile_open_mode::trunc) == cufile_open_mode::trunc) ? O_TRUNC : 0;
    __ret |= ((__om & cufile_open_mode::noreplace) == cufile_open_mode::noreplace) ? O_EXCL : 0;
    __ret |= ((__om & cufile_open_mode::direct) == cufile_open_mode::direct) ? O_DIRECT : 0;
    return __ret;
  }

  //! @brief Wrapper for opening the native handle.
  [[nodiscard]] static _CCCL_HOST_API native_handle_type __open_file(const char* __filename, __oflags_type __oflags)
  {
    // if O_CREAT flag is specified, use the same mode as if opened by fopend
    ::mode_t __ocreat_mode{};
    if (__oflags & O_CREAT)
    {
      __ocreat_mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH;
    }

    int __fd = ::open(__filename, __oflags, __ocreat_mode);

    if (__fd == -1)
    {
      errno = 0; // clear errno
      ::cuda::std::__throw_runtime_error("Failed to open file.");
    }

    return __fd;
  }

  //! @brief Wrapper for retrieving the open mode.
  [[nodiscard]] static _CCCL_HOST_API cufile_open_mode __open_mode(native_handle_type __native_handle)
  {
    int __oflags = ::fcntl(__native_handle, F_GETFL);

    if (__oflags == -1)
    {
      errno = 0; // clear errno
      ::cuda::std::__throw_runtime_error("Failed to retrieve open flags.");
    }

    cufile_open_mode __om{};
    if (__oflags & O_RDWR)
    {
      __om |= cufile_open_mode::in | cufile_open_mode::out;
    }
    else if (__oflags & O_RDONLY)
    {
      __om |= cufile_open_mode::in;
    }
    else if (__oflags & O_WRONLY)
    {
      __om |= cufile_open_mode::out;
    }
    __om |= (__oflags & O_TRUNC) ? cufile_open_mode::trunc : cufile_open_mode{};
    __om |= (__oflags & O_EXCL) ? cufile_open_mode::noreplace : cufile_open_mode{};
    __om |= (__oflags & O_DIRECT) ? cufile_open_mode::direct : cufile_open_mode{};
    return __om;
  }

  //! @brief Wrapper for closing the native handle.
  [[nodiscard]] static _CCCL_HOST_API bool __close_file_no_throw(native_handle_type __native_handle) noexcept
  {
    return ::close(__native_handle) == 0;
  }

  //! @brief Wrapper for closing the native handle. Throws \c cuda::std::runtime_error if an error occurs.
  static _CCCL_HOST_API void __close_file(native_handle_type __native_handle)
  {
    if (!__close_file_no_throw(__native_handle))
    {
      errno = 0; // clear errno
      ::cuda::std::__throw_runtime_error("Failed to close file.");
    }
  }

public:
  //! @brief Make a cufile object from already existing native handle.
  //!
  //         The ownership of the handle is transferred to the object and the handle is registered by the cuFile driver.
  //!
  //! @param __native_handle The native handle.
  //!
  //! @return The created cufile object.
  [[nodiscard]] static _CCCL_HOST_API cufile from_native_handle(native_handle_type __native_handle)
  {
    return cufile{cufile_driver.register_native_handle(__native_handle), __native_handle};
  }

  _CCCL_HIDE_FROM_ABI cufile() noexcept = default;

  //! @brief Constructs the object by opening file @c __filename in mode @c __open_mode.
  //!
  //! @param __filename Path to the file. Must be a zero terminated string.
  //! @param __open_mode Open mode to open the file with.
  //!
  //! @throws cuda::std::runtime_error if the file cannot be opened.
  //! @throws cuda::cuda_error if a CUDA driver error occurs.
  //! @throws cuda::cufile_error if a cuFile driver error occurs.
  _CCCL_HOST_API cufile(const char* __filename, cufile_open_mode __open_mode)
  {
    __native_handle_ = __open_file(__filename, __make_oflags(__open_mode));
    try
    {
      __cufile_handle_ = cufile_driver.register_native_handle(__native_handle_).get();
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
  _CCCL_HOST_API cufile(cufile&& __other) noexcept
      : cufile_ref{::cuda::std::exchange(__other.__cufile_handle_, nullptr)}
      , __native_handle_{::cuda::std::exchange(__other.__native_handle_, __invalid_native_handle)}
  {}

  cufile& operator=(const cufile&) = delete;

  //! @brief Move-assign from a @c cufile object.
  //!
  //! @param __other The other @c cufile.
  //!
  //! @post `__other` is in moved-from state.
  //!
  //! @throws cuda::std::runtime_error if the currently opened file fails to close.
  _CCCL_HOST_API cufile& operator=(cufile&& __other)
  {
    if (this != ::cuda::std::addressof(__other))
    {
      close();
      __native_handle_ = ::cuda::std::exchange(__other.__native_handle_, __invalid_native_handle);
      __cufile_handle_ = ::cuda::std::exchange(__other.__cufile_handle_, nullptr);
    }
    return *this;
  }

  //! @brief Destructor. Deregisters the cuFile file handle and closes the native handle.
  _CCCL_HOST_API ~cufile()
  {
    if (is_open())
    {
      cufile_driver.deregister_native_handle(__cufile_handle_);
      [[maybe_unused]] const auto __ignore_close_retval = __close_file_no_throw(__native_handle_);
    }
  }

  //! @brief Queries whether the file is opened.
  //!
  //! @return True, if opened, false otherwise.
  [[nodiscard]] _CCCL_HOST_API bool is_open() const noexcept
  {
    return __native_handle_ != __invalid_native_handle;
  }

  //! @brief Queries the open mode the object was opened with.
  //!
  //! @return The \c cuda::cufile_open_mode value if opened, empty value otherwise.
  [[nodiscard]] _CCCL_HOST_API cufile_open_mode open_mode() const
  {
    return is_open() ? __open_mode(__native_handle_) : cufile_open_mode{};
  }

  //! @brief Opens file @c __filename in mode @c __open_mode.
  //!
  //! @param __filename Path to the file.
  //! @param __open_mode Open mode to open the file with.
  //!
  //! @throws cuda::std::runtime_error if the file cannot be opened or if a file is already opened.
  //! @throws cuda::cuda_error if a CUDA driver error occurs.
  //! @throws cuda::cufile_error if a cuFile driver error occurs.
  _CCCL_HOST_API void open(const char* __filename, cufile_open_mode __open_mode)
  {
    if (is_open())
    {
      ::cuda::std::__throw_runtime_error("File is already opened.");
    }

    __native_handle_ = __open_file(__filename, __make_oflags(__open_mode));

    try
    {
      __cufile_handle_ = cufile_driver.register_native_handle(__native_handle_).get();
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
  _CCCL_HOST_API void close()
  {
    if (!is_open())
    {
      return;
    }

    cufile_driver.deregister_native_handle(::cuda::std::exchange(__cufile_handle_, nullptr));
    __close_file(::cuda::std::exchange(__native_handle_, __invalid_native_handle));
  }

  //! @brief Gets the OS native handle.
  //!
  //! @return The native handle.
  [[nodiscard]] _CCCL_HOST_API native_handle_type native_handle() const noexcept
  {
    return __native_handle_;
  }

  //! @brief Deregisters the cuFile file handle and releases the native handle. The ownership of the native handle is
  //!        transferred to the caller.
  //!
  //! @returns The native handle.
  [[nodiscard]] _CCCL_HOST_API native_handle_type release() noexcept
  {
    cufile_driver.deregister_native_handle(::cuda::std::exchange(__cufile_handle_, nullptr));
    return ::cuda::std::exchange(__native_handle_, __invalid_native_handle);
  }
};
} // namespace cuda::experimental
