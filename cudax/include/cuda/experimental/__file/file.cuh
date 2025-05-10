//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__FILE_FILE
#define _CUDAX__FILE_FILE

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__utility/exchange.h>
#include <cuda/std/__utility/swap.h>

#include <cuda/experimental/__file/cufile_api.cuh>
#include <cuda/experimental/stream.cuh>

#include <cstdio>
#include <utility>

namespace cuda::experimental
{

// Provide a type-safe wrappers for offsets
enum class file_offset_t : ::off_t
{
};
enum class buffer_offset_t : ::off_t
{
};

class file
{
public:
  //! @brief Default constructor.
  file() = default;

  //! @brief Construct a file object and open the file.
  //!
  //! @param filename The name of the file to open.
  //! @param mode The mode in which to open the file.
  //!
  //! @note The mode parameter is the same as the mode parameter in the standard C library fopen function.
  //!
  //! @throws runtime_error if the file cannot be opened.
  //! @throws cuda_error if the file handle cannot be registered.
  file(const char* filename, const char* mode)
  {
    open(filename, mode);
  }

  file(const file&) = delete;

  //! @brief Move constructor.
  file(file&& other) noexcept
      : __file_{::cuda::std::exchange(other.__file_, nullptr)}
      , __handle_{::cuda::std::exchange(other.__handle_, nullptr)}
  {}

  //! @brief Destructor. Closes the file if it is open.
  //!
  //! @note If the file is open, the file handle is deregistered and the file is closed. If any errors occur during
  //!       deregistration or closing, they are silently ignored.
  ~file()
  {
    if (is_open())
    {
      detail::__cufile_handle_deregister<false>(__handle_);
      ::std::fclose(__file_);
    }
  }

  file& operator=(const file&) = delete;

  //! @brief Move assignment operator.
  file& operator=(file&& other)
  {
    close();

    __file_   = ::cuda::std::exchange(other.__file_, nullptr);
    __handle_ = ::cuda::std::exchange(other.__handle_, nullptr);
  }

  //! @brief Check if the file is open.
  //!
  //! @return true if the file is open, false otherwise.
  _CCCL_NODISCARD bool is_open() const noexcept
  {
    return __file_ != nullptr;
  }

  //! @brief Open a file.
  //!
  //! @param filename The name of the file to open.
  //! @param mode The mode in which to open the file.
  //!
  //! @throws runtime_error if the file cannot be opened.
  //! @throws cuda_error if the file handle cannot be registered.
  void open(const char* filename, const char* mode)
  {
    __file_ = ::std::fopen(filename, mode);
    if (__file_ == nullptr)
    {
      throw ::std::runtime_error{"Failed to open a cuda::file"};
    }

    __register_file_handle();
  }

  //! @brief Close the file.
  //!
  //! @note If the file is open, the file handle is deregistered and the file is closed.
  void close()
  {
    if (is_open())
    {
      detail::__cufile_handle_deregister(::cuda::std::exchange(__handle_, nullptr));
      ::std::fclose(::cuda::std::exchange(__file_, nullptr));
    }
  }

  //! @brief Swap the object data with another file object.
  //!
  //! @param other The other file object to swap with.
  void swap(file& other)
  {
    ::cuda::std::swap(__file_, other.__file_);
    ::cuda::std::swap(__handle_, other.__handle_);
  }

  //! @brief Get the underlying FILE* object.
  operator ::std::FILE*() const noexcept
  {
    return __file_;
  }

  //! @brief Read data from the file.
  //!
  //! @param buffer The buffer to read data into. Must be the base address of the buffer.
  //! @param size The size of the buffer.
  //! @param file_offset The offset in the file to read from. Defaults to 0.
  //! @param buffer_offset The offset in the buffer to read into. Defaults to 0.
  //!
  //! @return The number of bytes read, or -1 if an error occurred. In that case, errno is set.
  //!
  //! @throws cuda_error if an error occurs during the read operation on the cuFile side.
  ::ssize_t
  read(void* buffer, ::cuda::std::size_t size, file_offset_t file_offset = {}, buffer_offset_t buffer_offset = {})
  {
    return detail::__cufile_read(
      __handle_, buffer, size, ::cuda::std::to_underlying(file_offset), ::cuda::std::to_underlying(buffer_offset));
  }

  //! @brief Read data from the file asynchronously.
  //!
  //! @param[in] stream The stream to use for the asynchronous operation.
  //! @param[in] buffer The buffer to read data into. Must be the base address of the buffer.
  //! @param[in] size_ptr The size of the buffer.
  //! @param[inout] nbytes_read_ptr The number of bytes read.
  //! @param[in] file_offset_ptr The offset in the file to read from. Defaults to nullptr, which will be understood as
  //! 0.
  //! @param[in] buffer_offset_ptr The offset in the buffer to read into. Defaults to nullptr, which will be understood
  //! as 0.
  //!
  //! @note Depending on the stream flags, the size_ptr, file_offset_ptr, and buffer_offset_ptr values may be read
  //! during the
  //!       submission of the asynchronous operation or during the execution of the operation. See the cuFile
  //!       documentation for more information.
  //!
  //! @return The number of bytes read, or -1 if an error occurred in the nbytes_read_ptr. In that case, errno is set.
  //!
  //! @throws cuda_error if an error occurs during the read operation on the cuFile side.
  void read_async(stream_ref stream,
                  void* buffer,
                  const ::cuda::std::size_t* size_ptr,
                  ::ssize_t* nbytes_read_ptr,
                  const file_offset_t* file_offset_ptr     = nullptr,
                  const buffer_offset_t* buffer_offset_ptr = nullptr)
  {
    static const ::off_t __zero_offset{};

    const ::off_t* __file_offset =
      (file_offset_ptr != nullptr) ? reinterpret_cast<const ::off_t*>(file_offset_ptr) : &__zero_offset;
    const ::off_t* __buffer_offset =
      (buffer_offset_ptr != nullptr) ? reinterpret_cast<const ::off_t*>(buffer_offset_ptr) : &__zero_offset;

    detail::__cufile_read_async(
      __handle_, buffer, size_ptr, __file_offset, __buffer_offset, nbytes_read_ptr, stream.get());
  }

  //! @brief Write data to the file.
  //!
  //! @param buffer The buffer to write data from.
  //! @param size The size of the buffer.
  //! @param file_offset The offset in the file to write to. Defaults to 0.
  //! @param buffer_offset The offset in the buffer to write from. Defaults to 0.
  //!
  //! @return The number of bytes written, or -1 if an error occurred. In that case, errno is set.
  //!
  //! @throws cuda_error if an error occurs during the write operation on the cuFile side.
  ::ssize_t write(
    const void* buffer, ::cuda::std::size_t size, file_offset_t file_offset = {}, buffer_offset_t buffer_offset = {})
  {
    return detail::__cufile_write(
      __handle_, buffer, size, ::cuda::std::to_underlying(file_offset), ::cuda::std::to_underlying(buffer_offset));
  }

  //! @brief Write data to the file asynchronously.
  //!
  //! @param[in] stream The stream to use for the asynchronous operation.
  //! @param[in] buffer The buffer to write data from.
  //! @param[in] size_ptr The size of the buffer.
  //! @param[inout] bytes_written_ptr The number of bytes written.
  //! @param[in] file_offset_ptr The offset in the file to write to. Defaults to nullptr, which will be understood as 0.
  //! @param[in] buffer_offset_ptr The offset in the buffer to write from. Defaults to nullptr, which will be understood
  //! as 0.
  //!
  //! @note Depending on the stream flags, the size_ptr, file_offset_ptr, and buffer_offset_ptr values may be read
  //! during the
  //!       submission of the asynchronous operation or during the execution of the operation. See the cuFile
  //!       documentation for more information.
  //!
  //! @return The number of bytes written, or -1 if an error occurred in the bytes_written_ptr.
  //!
  //! @throws cuda_error if an error occurs during the write operation on the cuFile side.
  void write_async(stream_ref stream,
                   const void* buffer,
                   const ::cuda::std::size_t* size_ptr,
                   ::ssize_t* bytes_written_ptr,
                   const file_offset_t* file_offset_ptr     = nullptr,
                   const buffer_offset_t* buffer_offset_ptr = nullptr)
  {
    static const ::off_t __zero_offset{};

    const ::off_t* __file_offset =
      (file_offset_ptr != nullptr) ? reinterpret_cast<const ::off_t*>(file_offset_ptr) : &__zero_offset;
    const ::off_t* __buffer_offset =
      (buffer_offset_ptr != nullptr) ? reinterpret_cast<const ::off_t*>(buffer_offset_ptr) : &__zero_offset;

    detail::__cufile_write_async(
      __handle_, buffer, size_ptr, __file_offset, __buffer_offset, bytes_written_ptr, stream.get());
  }

private:
  void __register_file_handle()
  {
    int __fd = ::fileno(__file_);
    if (__fd == -1)
    {
      throw ::std::runtime_error{"Failed to get file descriptor during opening a cuda::file"};
    }

    CUfileDescr_t __descr{};
    __descr.type      = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    __descr.handle.fd = __fd;

    __handle_ = detail::__cufile_handle_register(&__descr);
  }

  ::std::FILE* __file_{};
  ::CUfileHandle_t __handle_{};
};

} // namespace cuda::experimental

void ::std::swap(::cuda::experimental::file& lhs, ::cuda::experimental::file& rhs)
{
  lhs.swap(rhs);
}

void ::cuda::std::swap(::cuda::experimental::file& lhs, ::cuda::experimental::file& rhs)
{
  lhs.swap(rhs);
}

#endif // _CUDAX__FILE_FILE
