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

#include <cuda/std/utility>
#include <cuda/stream_ref>

#include <cuda/experimental/__cufile/batch_handle.hpp>
#include <cuda/experimental/__cufile/buffer_handle.hpp>
#include <cuda/experimental/__cufile/cufile.hpp>
#include <cuda/experimental/__cufile/detail/enums.hpp>
#include <cuda/experimental/__cufile/stream_handle.hpp>

#include <ios>
#include <string>

#include <fcntl.h>
#include <unistd.h>

namespace cuda::experimental::cufile
{

class file_handle;

//! Non-owning reference to a file handle for cuFILE operations
class file_handle_ref
{
protected:
  int fd_                       = -1;
  CUfileHandle_t cufile_handle_ = nullptr;

  file_handle_ref() = default; // for derived move-ctor
  void register_file();

public:
  //! Create from existing file descriptor (non-owning)
  explicit file_handle_ref(int fd);

  file_handle_ref(const file_handle_ref&)            = delete;
  file_handle_ref& operator=(const file_handle_ref&) = delete;
  file_handle_ref(file_handle_ref&&)                 = delete;
  file_handle_ref& operator=(file_handle_ref&&)      = delete;
  ~file_handle_ref() noexcept
  {
    if (cufile_handle_ != nullptr)
    {
      cuFileHandleDeregister(cufile_handle_);
      cufile_handle_ = nullptr;
    }
  }

  int get_fd() const noexcept
  {
    return fd_;
  }

  //! Read data from file using span
  template <typename T>
  size_t read(::cuda::std::span<T> buffer, off_t file_offset = 0, off_t buffer_offset = 0);

  //! Write data to file using span
  template <typename T>
  size_t write(::cuda::std::span<const T> buffer, off_t file_offset = 0, off_t buffer_offset = 0);

  //! Asynchronous read using span
  template <typename T>
  void read_async(
    ::cuda::stream_ref stream, ::cuda::std::span<T> buffer, off_t file_offset, off_t buffer_offset, ssize_t& bytes_read);

  //! Asynchronous write using span
  template <typename T>
  void write_async(::cuda::stream_ref stream,
                   ::cuda::std::span<const T> buffer,
                   off_t file_offset,
                   off_t buffer_offset,
                   ssize_t& bytes_written);

  //! Get native cuFILE handle
  CUfileHandle_t native_handle() const noexcept
  {
    return cufile_handle_;
  }

  //! Check if the handle owns a valid resource
  bool is_valid() const noexcept
  {
    return cufile_handle_ != nullptr;
  }
};

//! RAII file handle for cuFILE operations (owning)
class file_handle : public file_handle_ref
{
public:
  //! Open file for cuFILE operations
  explicit file_handle(const ::std::string& path, ::std::ios_base::openmode mode = ::std::ios_base::in);

  //! Create from existing file descriptor (owning)
  explicit file_handle(int fd);

  file_handle(file_handle&& other) noexcept;
  file_handle& operator=(file_handle&& other) noexcept;
  ~file_handle() noexcept;

private:
  static int convert_ios_mode(::std::ios_base::openmode mode);
  static int open_file_descriptor(const ::std::string& path, ::std::ios_base::openmode mode);
  static int validate_fd_or_throw(int fd);
};

} // namespace cuda::experimental::cufile

// ===================== Inline implementations =====================

namespace cuda::experimental::cufile
{

// Static method implementations
inline int file_handle::convert_ios_mode(::std::ios_base::openmode mode)
{
  int flags = 0;

  bool has_in  = (mode & ::std::ios_base::in) != 0;
  bool has_out = (mode & ::std::ios_base::out) != 0;

  if (has_in && has_out)
  {
    flags |= O_RDWR;
  }
  else if (has_out)
  {
    flags |= O_WRONLY;
  }
  else
  {
    flags |= O_RDONLY;
  }

  if (mode & ::std::ios_base::trunc)
  {
    flags |= O_TRUNC;
  }

  if (mode & ::std::ios_base::app)
  {
    flags |= O_APPEND;
  }

  if (has_out)
  {
    flags |= O_CREAT;
  }

  flags |= O_DIRECT;

  return flags;
}

inline void file_handle_ref::register_file()
{
  CUfileDescr_t desc = {};
  desc.handle.fd     = fd_;
  desc.type          = to_c_enum(cu_file_handle_type::opaque_fd);
  desc.fs_ops        = nullptr;

  CUfileHandle_t handle;
  CUfileError_t error = cuFileHandleRegister(&handle, &desc);
  ::cuda::experimental::cufile::detail::check_cufile_result(error, "cuFileHandleRegister");
  cufile_handle_ = handle;
}

inline int file_handle::open_file_descriptor(const ::std::string& path, ::std::ios_base::openmode mode)
{
  int flags = convert_ios_mode(mode);
  int fd    = open(path.c_str(), flags, 0644);
  if (fd < 0)
  {
    throw ::std::system_error(errno, ::std::system_category(), "Failed to open file: " + path);
  }
  return fd;
}

inline int file_handle::validate_fd_or_throw(int fd)
{
  if (fd < 0)
  {
    throw ::std::invalid_argument("Invalid file descriptor");
  }
  return fd;
}

// Constructor implementations
inline file_handle::file_handle(const ::std::string& path, ::std::ios_base::openmode mode)
    : file_handle_ref(open_file_descriptor(path, mode))
{}

inline file_handle::file_handle(int fd)
    : file_handle_ref(validate_fd_or_throw(fd))
{}

// Move constructor and assignment
inline file_handle::file_handle(file_handle&& other) noexcept
    : file_handle_ref()
{
  fd_                  = other.fd_;
  cufile_handle_       = other.cufile_handle_;
  other.cufile_handle_ = nullptr;
  other.fd_            = -1;
}

inline file_handle& file_handle::operator=(file_handle&& other) noexcept
{
  if (this != ::cuda::std::addressof(other))
  {
    // Clean up file descriptor
    if (fd_ >= 0)
    {
      close(fd_);
    }

    fd_ = other.fd_;
    // Deregister current handle if owned
    if (cufile_handle_ != nullptr)
    {
      cuFileHandleDeregister(cufile_handle_);
    }
    cufile_handle_       = other.cufile_handle_;
    other.cufile_handle_ = nullptr;

    other.fd_ = -1;
  }
  return *this;
}

// Destructor
inline file_handle::~file_handle() noexcept
{
  if (fd_ >= 0)
  {
    close(fd_);
  }
}

// file_handle_ref constructors
inline file_handle_ref::file_handle_ref(int fd)
{
  fd_ = fd;
  if (fd_ < 0)
  {
    throw ::std::invalid_argument("Invalid file descriptor");
  }
  register_file();
}

// Template method implementations
template <typename T>
inline size_t file_handle_ref::read(::cuda::std::span<T> buffer, off_t file_offset, off_t buffer_offset)
{
  static_assert(::cuda::std::is_trivially_copyable_v<T>, "Type must be trivially copyable for cuFile operations");

  // Convert span to void* and size for cuFile API
  void* buffer_ptr  = static_cast<void*>(buffer.data());
  size_t size_bytes = buffer.size_bytes();

  ssize_t result = cuFileRead(cufile_handle_, buffer_ptr, size_bytes, file_offset, buffer_offset);
  return static_cast<size_t>(detail::check_cufile_result(result, "cuFileRead"));
}

template <typename T>
inline size_t file_handle_ref::write(::cuda::std::span<const T> buffer, off_t file_offset, off_t buffer_offset)
{
  static_assert(::cuda::std::is_trivially_copyable_v<T>, "Type must be trivially copyable for cuFile operations");

  // Convert span to void* and size for cuFile API
  const void* buffer_ptr = static_cast<const void*>(buffer.data());
  size_t size_bytes      = buffer.size_bytes();

  ssize_t result = cuFileWrite(cufile_handle_, buffer_ptr, size_bytes, file_offset, buffer_offset);
  return static_cast<size_t>(detail::check_cufile_result(result, "cuFileWrite"));
}

template <typename T>
inline void file_handle_ref::read_async(
  ::cuda::stream_ref stream, ::cuda::std::span<T> buffer, off_t file_offset, off_t buffer_offset, ssize_t& bytes_read)
{
  static_assert(::cuda::std::is_trivially_copyable_v<T>, "Type must be trivially copyable for cuFile operations");

  // cuFile async API requires size parameters to be passed by pointer
  size_t size_bytes = buffer.size_bytes();
  void* buffer_ptr  = static_cast<void*>(buffer.data());

  CUfileError_t error =
    cuFileReadAsync(cufile_handle_, buffer_ptr, &size_bytes, &file_offset, &buffer_offset, &bytes_read, stream.get());
  detail::check_cufile_result(error, "cuFileReadAsync");
}

template <typename T>
inline void file_handle_ref::write_async(
  ::cuda::stream_ref stream,
  ::cuda::std::span<const T> buffer,
  off_t file_offset,
  off_t buffer_offset,
  ssize_t& bytes_written)
{
  static_assert(::cuda::std::is_trivially_copyable_v<T>, "Type must be trivially copyable for cuFile operations");

  // cuFile async API requires size parameters to be passed by pointer
  size_t size_bytes      = buffer.size_bytes();
  const void* buffer_ptr = static_cast<const void*>(buffer.data());

  CUfileError_t error = cuFileWriteAsync(
    cufile_handle_,
    const_cast<void*>(buffer_ptr),
    &size_bytes,
    &file_offset,
    &buffer_offset,
    &bytes_written,
    stream.get());
  ::cuda::experimental::cufile::detail::check_cufile_result(error, "cuFileWriteAsync");
}

// getters implemented inline in class

} // namespace cuda::experimental::cufile
