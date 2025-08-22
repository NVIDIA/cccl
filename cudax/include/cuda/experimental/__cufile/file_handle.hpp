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
#include <cuda/experimental/__cufile/detail/error_handling.hpp>
#include <cuda/experimental/__cufile/detail/raii_resource.hpp>
#include <cuda/experimental/__cufile/stream_handle.hpp>

#include <ios>
#include <iostream>
#include <string>

#include <fcntl.h>
#include <unistd.h>

namespace cuda::experimental::cufile
{

// Forward declarations
class file_handle;

/**
 * @brief Base class for file handle operations
 */
class file_handle_base
{
protected:
  int fd_;
  detail::raii_resource<CUfileHandle_t, void (*)(CUfileHandle_t)> cufile_handle_;

  static int convert_ios_mode(::std::ios_base::openmode mode);
  void register_file();

public:
  /**
   * @brief Get file descriptor
   */
  int get_fd() const noexcept
  {
    return fd_;
  }

  /**
   * @brief Read data from file using span
   * @tparam T Element type (must be trivially copyable)
   * @param buffer Span representing the destination buffer
   * @param file_offset Offset in file to read from
   * @param buffer_offset Offset in buffer to read into (in bytes)
   * @return Number of bytes read
   */
  template <typename T>
  size_t read(::cuda::std::span<T> buffer, off_t file_offset = 0, off_t buffer_offset = 0);

  /**
   * @brief Write data to file using span
   * @tparam T Element type (must be trivially copyable)
   * @param buffer Span representing the source buffer
   * @param file_offset Offset in file to write to
   * @param buffer_offset Offset in buffer to write from (in bytes)
   * @return Number of bytes written
   */
  template <typename T>
  size_t write(::cuda::std::span<const T> buffer, off_t file_offset = 0, off_t buffer_offset = 0);

  /**
   * @brief Asynchronous read using span
   * @tparam T Element type (must be trivially copyable)
   * @param stream CUDA stream for async operation
   * @param buffer Span representing the destination buffer
   * @param file_offset Offset in file to read from
   * @param buffer_offset Offset in buffer to read into (in bytes)
   * @param bytes_read Output parameter for bytes read
   */
  template <typename T>
  void read_async(
    ::cuda::stream_ref stream, ::cuda::std::span<T> buffer, off_t file_offset, off_t buffer_offset, ssize_t& bytes_read);

  /**
   * @brief Asynchronous write using span
   * @tparam T Element type (must be trivially copyable)
   * @param stream CUDA stream for async operation
   * @param buffer Span representing the source buffer
   * @param file_offset Offset in file to write to
   * @param buffer_offset Offset in buffer to write from (in bytes)
   * @param bytes_written Output parameter for bytes written
   */
  template <typename T>
  void write_async(::cuda::stream_ref stream,
                   ::cuda::std::span<const T> buffer,
                   off_t file_offset,
                   off_t buffer_offset,
                   ssize_t& bytes_written);

  /**
   * @brief Get native cuFILE handle
   */
  CUfileHandle_t native_handle() const noexcept;

  /**
   * @brief Check if the handle owns a valid resource
   */
  bool is_valid() const noexcept;
};

/**
 * @brief Non-owning reference to a file handle for cuFILE operations
 */
class file_handle_ref : public file_handle_base
{
public:
  /**
   * @brief Create from existing file descriptor (non-owning)
   * @param fd File descriptor (should be opened with O_DIRECT)
   */
  explicit file_handle_ref(int fd);

  file_handle_ref(const file_handle_ref&)            = delete;
  file_handle_ref& operator=(const file_handle_ref&) = delete;
  file_handle_ref(file_handle_ref&&)                 = delete;
  file_handle_ref& operator=(file_handle_ref&&)      = delete;
  ~file_handle_ref()                                 = default;
};

/**
 * @brief RAII file handle for cuFILE operations (owning)
 */
class file_handle : public file_handle_base
{
public:
  /**
   * @brief Open file for cuFILE operations
   * @param path File path
   * @param mode STL-compatible open mode flags
   */
  explicit file_handle(const ::std::string& path, ::std::ios_base::openmode mode = ::std::ios_base::in);

  /**
   * @brief Create from existing file descriptor (owning)
   * @param fd File descriptor (should be opened with O_DIRECT)
   */
  explicit file_handle(int fd);

  file_handle(file_handle&& other) noexcept;
  file_handle& operator=(file_handle&& other) noexcept;
  ~file_handle() noexcept;
};

} // namespace cuda::experimental::cufile

#include <cuda/experimental/__cufile/detail/batch_handle_impl.hpp>
#include <cuda/experimental/__cufile/detail/file_handle_impl.hpp>
