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

// This file provides the implementation of file_handle methods
// It's included after the class definition to avoid circular dependency issues

#include <cuda/experimental/__cufile/file_handle.hpp>

#include <ios>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>

#include <fcntl.h>
#include <unistd.h>

namespace cuda::experimental::cufile
{

// Static method implementations
inline int file_handle_base::convert_ios_mode(::std::ios_base::openmode mode)
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

inline void file_handle_base::register_file()
{
  CUfileDescr_t desc = {};
  desc.handle.fd     = fd_;
  desc.type          = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  desc.fs_ops        = nullptr;

  CUfileHandle_t handle;
  CUfileError_t error = cuFileHandleRegister(&handle, &desc);
  detail::check_cufile_result(error, "cuFileHandleRegister");

  cufile_handle_.emplace(handle, [](CUfileHandle_t h) {
    cuFileHandleDeregister(h);
  });
}

// Constructor implementations
inline file_handle::file_handle(const ::std::string& path, ::std::ios_base::openmode mode)
    : file_handle_base()
{
  int flags = convert_ios_mode(mode);
  fd_       = open(path.c_str(), flags, 0644);

  if (fd_ < 0)
  {
    throw ::std::system_error(errno, ::std::system_category(), "Failed to open file: " + path);
  }

  register_file();
}

inline file_handle::file_handle(int fd)
    : file_handle_base()
{
  fd_ = fd;
  if (fd_ < 0)
  {
    throw ::std::invalid_argument("Invalid file descriptor");
  }

  register_file();
}

// Move constructor and assignment
inline file_handle::file_handle(file_handle&& other) noexcept
    : file_handle_base()
{
  fd_            = other.fd_;
  cufile_handle_ = ::std::move(other.cufile_handle_);
  other.fd_      = -1;
}

inline file_handle& file_handle::operator=(file_handle&& other) noexcept
{
  if (this != &other)
  {
    // Clean up file descriptor
    if (fd_ >= 0)
    {
      close(fd_);
    }

    fd_            = other.fd_;
    cufile_handle_ = ::std::move(other.cufile_handle_);

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
    : file_handle_base()
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
size_t file_handle_base::read(cuda::std::span<T> buffer, off_t file_offset, off_t buffer_offset)
{
  static_assert(cuda::std::is_trivially_copyable_v<T>, "Type must be trivially copyable for cuFile operations");

  // Convert span to void* and size for cuFile API
  void* buffer_ptr  = static_cast<void*>(buffer.data());
  size_t size_bytes = buffer.size_bytes();

  ssize_t result = cuFileRead(cufile_handle_.get(), buffer_ptr, size_bytes, file_offset, buffer_offset);
  return static_cast<size_t>(detail::check_cufile_result(result, "cuFileRead"));
}

template <typename T>
size_t file_handle_base::write(cuda::std::span<const T> buffer, off_t file_offset, off_t buffer_offset)
{
  static_assert(cuda::std::is_trivially_copyable_v<T>, "Type must be trivially copyable for cuFile operations");

  // Convert span to void* and size for cuFile API
  const void* buffer_ptr = static_cast<const void*>(buffer.data());
  size_t size_bytes      = buffer.size_bytes();

  ssize_t result = cuFileWrite(cufile_handle_.get(), buffer_ptr, size_bytes, file_offset, buffer_offset);
  return static_cast<size_t>(detail::check_cufile_result(result, "cuFileWrite"));
}

template <typename T>
void file_handle_base::read_async(
  cuda::stream_ref stream, cuda::std::span<T> buffer, off_t file_offset, off_t buffer_offset, ssize_t& bytes_read)
{
  static_assert(cuda::std::is_trivially_copyable_v<T>, "Type must be trivially copyable for cuFile operations");

  // cuFile async API requires size parameters to be passed by pointer
  size_t size_bytes = buffer.size_bytes();
  void* buffer_ptr  = static_cast<void*>(buffer.data());

  CUfileError_t error = cuFileReadAsync(
    cufile_handle_.get(), buffer_ptr, &size_bytes, &file_offset, &buffer_offset, &bytes_read, stream.get());
  detail::check_cufile_result(error, "cuFileReadAsync");
}

template <typename T>
void file_handle_base::write_async(
  cuda::stream_ref stream,
  cuda::std::span<const T> buffer,
  off_t file_offset,
  off_t buffer_offset,
  ssize_t& bytes_written)
{
  static_assert(cuda::std::is_trivially_copyable_v<T>, "Type must be trivially copyable for cuFile operations");

  // cuFile async API requires size parameters to be passed by pointer
  size_t size_bytes      = buffer.size_bytes();
  const void* buffer_ptr = static_cast<const void*>(buffer.data());

  CUfileError_t error = cuFileWriteAsync(
    cufile_handle_.get(),
    const_cast<void*>(buffer_ptr),
    &size_bytes,
    &file_offset,
    &buffer_offset,
    &bytes_written,
    stream.get());
  detail::check_cufile_result(error, "cuFileWriteAsync");
}

// Simple getter implementations
inline CUfileHandle_t file_handle_base::native_handle() const noexcept
{
  return cufile_handle_.get();
}

inline bool file_handle_base::is_valid() const noexcept
{
  return cufile_handle_.has_value();
}

} // namespace cuda::experimental::cufile
