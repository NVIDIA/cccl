//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__FILE_UTILS
#define _CUDAX__FILE_UTILS

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__utility/exchange.h>
#include <cuda/std/__utility/to_underlying.h>

#include <cuda/experimental/__file/cufile_api.cuh>
#include <cuda/experimental/__stream/stream_ref.cuh>

namespace cuda::experimental
{

enum class file_buffer_flags
{
  rdma_register         = CU_FILE_RDMA_REGISTER,
  rdma_relaxed_ordering = CU_FILE_RDMA_RELAXED_ORDERING,

  default_flags = 0,
};

_CCCL_NODISCARD file_buffer_flags operator|(file_buffer_flags lhs, file_buffer_flags rhs) noexcept
{
  return static_cast<file_buffer_flags>(::cuda::std::to_underlying(lhs) | ::cuda::std::to_underlying(rhs));
}

_CCCL_NODISCARD file_buffer_flags operator&(file_buffer_flags lhs, file_buffer_flags rhs) noexcept
{
  return static_cast<file_buffer_flags>(::cuda::std::to_underlying(lhs) & ::cuda::std::to_underlying(rhs));
}

_CCCL_NODISCARD file_buffer_flags operator^(file_buffer_flags lhs, file_buffer_flags rhs) noexcept
{
  return static_cast<file_buffer_flags>(::cuda::std::to_underlying(lhs) ^ ::cuda::std::to_underlying(rhs));
}

_CCCL_NODISCARD file_buffer_flags operator~(file_buffer_flags rhs) noexcept
{
  return static_cast<file_buffer_flags>(~::cuda::std::to_underlying(rhs));
}

_CCCL_NODISCARD file_buffer_flags& operator|=(file_buffer_flags& lhs, file_buffer_flags rhs) noexcept
{
  return lhs = lhs | rhs;
}

_CCCL_NODISCARD file_buffer_flags& operator&=(file_buffer_flags& lhs, file_buffer_flags rhs) noexcept
{
  return lhs = lhs & rhs;
}

_CCCL_NODISCARD file_buffer_flags& operator^=(file_buffer_flags& lhs, file_buffer_flags rhs) noexcept
{
  return lhs = lhs ^ rhs;
}

class file_buffer_registerer
{
public:
  //! @brief Default constructor
  file_buffer_registerer() = default;

  //! @brief Register a buffer for use with cuFile
  //!
  //! @param buffer The buffer to register
  //! @param size The size of the buffer
  //! @param flags The flags to use for registration
  //!
  //! @throw cuda_error if the registration fails
  file_buffer_registerer(
    const void* buffer, ::cuda::std::size_t size, file_buffer_flags flags = file_buffer_flags::default_flags)
  {
    do_register(buffer, size, flags);
  }

  file_buffer_registerer(const file_buffer_registerer&) = delete;

  //! @brief Move constructor
  file_buffer_registerer(file_buffer_registerer&& other)
      : __buffer_{::cuda::std::exchange(other.__buffer_, nullptr)}
  {}

  //! @brief Destructor. Deregisters the buffer from cuFile if it is registered.
  //!
  //! @note If the buffer is not registered, this function has no effect.
  //! @note If the deregistration fails, the error is silently ignored.
  ~file_buffer_registerer()
  {
    if (__buffer_ != nullptr)
    {
      // If the deregister fails, we silently ignore the error
      detail::__cufile_buf_deregister<false>(__buffer_);
    }
  }

  file_buffer_registerer& operator=(const file_buffer_registerer&) = delete;

  //! @brief Move assignment operator
  file_buffer_registerer& operator=(file_buffer_registerer&& other)
  {
    if (this != &other)
    {
      __buffer_ = ::cuda::std::exchange(other.__buffer_, nullptr);
    }
    return *this;
  }

  //! @brief Register a buffer for use with cuFile
  //!
  //! @param buffer The buffer to register
  //! @param size The size of the buffer
  //! @param flags The flags to use for registration
  //!
  //! @throw cuda_error if the registration fails
  void
  do_register(const void* buffer, ::cuda::std::size_t size, file_buffer_flags flags = file_buffer_flags::default_flags)
  {
    deregister();

    __buffer_ = buffer;
    detail::__cufile_buf_register(__buffer_, size, ::cuda::std::to_underlying(flags));
  }

  //! @brief Deregister the buffer from cuFile
  //!
  //! @note If the buffer is not registered, this function has no effect.
  //!
  //! @throw cuda_error if the deregistration fails
  void deregister()
  {
    if (__buffer_ != nullptr)
    {
      detail::__cufile_buf_deregister(::cuda::std::exchange(__buffer_, nullptr));
    }
  }

private:
  const void* __buffer_{};
};

enum class file_stream_flags
{
  fixed_buffer_offset = CU_FILE_STREAM_FIXED_BUF_OFFSET,
  fixed_file_offset   = CU_FILE_STREAM_FIXED_FILE_OFFSET,
  fixed_file_size     = CU_FILE_STREAM_FIXED_FILE_SIZE,
  page_aligned_inputs = CU_FILE_STREAM_PAGE_ALIGNED_INPUTS,

  // Prevent dangling references by making the fixed flags the default
  default_flags = CU_FILE_STREAM_FIXED_BUF_OFFSET | CU_FILE_STREAM_FIXED_FILE_OFFSET | CU_FILE_STREAM_FIXED_FILE_SIZE,
};

_CCCL_NODISCARD file_stream_flags operator|(file_stream_flags lhs, file_stream_flags rhs) noexcept
{
  return static_cast<file_stream_flags>(::cuda::std::to_underlying(lhs) | ::cuda::std::to_underlying(rhs));
}

_CCCL_NODISCARD file_stream_flags operator&(file_stream_flags lhs, file_stream_flags rhs) noexcept
{
  return static_cast<file_stream_flags>(::cuda::std::to_underlying(lhs) & ::cuda::std::to_underlying(rhs));
}

_CCCL_NODISCARD file_stream_flags operator^(file_stream_flags lhs, file_stream_flags rhs) noexcept
{
  return static_cast<file_stream_flags>(::cuda::std::to_underlying(lhs) ^ ::cuda::std::to_underlying(rhs));
}

_CCCL_NODISCARD file_stream_flags operator~(file_stream_flags rhs) noexcept
{
  return static_cast<file_stream_flags>(~::cuda::std::to_underlying(rhs));
}

_CCCL_NODISCARD file_stream_flags& operator|=(file_stream_flags& lhs, file_stream_flags rhs) noexcept
{
  return lhs = lhs | rhs;
}

_CCCL_NODISCARD file_stream_flags& operator&=(file_stream_flags& lhs, file_stream_flags rhs) noexcept
{
  return lhs = lhs & rhs;
}

_CCCL_NODISCARD file_stream_flags& operator^=(file_stream_flags& lhs, file_stream_flags rhs) noexcept
{
  return lhs = lhs ^ rhs;
}

class file_stream_registerer
{
public:
  //! @brief Default constructor
  file_stream_registerer() = default;

  //! @brief Register a stream for use with cuFile
  //!
  //! @param stream The stream to register
  //! @param flags The flags to use for registration
  //!
  //! @throw cuda_error if the registration fails
  file_stream_registerer(stream_ref stream, file_stream_flags flags = file_stream_flags::default_flags)
  {
    do_register(stream, flags);
  }

  file_stream_registerer(const file_stream_registerer&) = delete;

  //! @brief Move constructor
  file_stream_registerer(file_stream_registerer&& other)
      : __stream_{::cuda::std::exchange(other.__stream_, detail::__invalid_stream)}
  {}

  //! @brief Destructor. Deregisters the stream from cuFile if it is registered.
  //!
  //! @note If the stream is not registered, this function has no effect.
  //! @note If the deregistration fails, the error is silently ignored.
  ~file_stream_registerer()
  {
    if (__stream_ != detail::__invalid_stream)
    {
      // If the deregister fails, we silently ignore the error
      detail::__cufile_stream_deregister<false>(__stream_);
    }
  }

  file_stream_registerer& operator=(const file_stream_registerer&) = delete;

  //! @brief Move assignment operator
  file_stream_registerer& operator=(file_stream_registerer&& other)
  {
    if (this != &other)
    {
      __stream_ = ::cuda::std::exchange(other.__stream_, detail::__invalid_stream);
    }
    return *this;
  }

  //! @brief Register a stream for use with cuFile
  //!
  //! @param stream The stream to register
  //! @param flags The flags to use for registration
  //!
  //! @throw cuda_error if the registration fails
  void do_register(stream_ref stream, file_stream_flags flags = file_stream_flags::default_flags)
  {
    deregister();

    __stream_ = stream.get();
    detail::__cufile_stream_register(__stream_, ::cuda::std::to_underlying(flags));
  }

  //! @brief Deregister the stream from cuFile
  //!
  //! @note If the stream is not registered, this function has no effect.
  //!
  //! @throw cuda_error if the deregistration fails
  void deregister()
  {
    if (__stream_ != detail::__invalid_stream)
    {
      detail::__cufile_stream_deregister(::cuda::std::exchange(__stream_, detail::__invalid_stream));
    }
  }

private:
  cudaStream_t __stream_{detail::__invalid_stream};
};

} // namespace cuda::experimental

#endif // _CUDAX__FILE_UTILS
