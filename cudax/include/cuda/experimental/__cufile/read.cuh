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

#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__cstddef/byte.h>
#include <cuda/std/__type_traits/is_const.h>
#include <cuda/std/__type_traits/is_trivially_copyable.h>
#include <cuda/std/atomic>
#include <cuda/std/detail/libcxx/include/stdexcept>
#include <cuda/std/span>

#include <cuda/experimental/__cufile/cufile_ref.cuh>
#include <cuda/experimental/__cufile/exception.cuh>

#include <cuda.h>
#include <cufile.h>
#include <errno.h>

namespace cuda::experimental
{

//********************************************************************************************************************//
// Sync APIs
//********************************************************************************************************************//

//! @brief Synchronously reads data from the cuFile handle.
//!
//! @param __file The cuFile handle.
//! @param __foffset The offset of the file to read from. In bytes.
//! @param __dst The buffer to store the data in.
//! @param __nbytes The number of bytes to read.
//! @param __doffset The buffer offset in bytes. Defaults to 0.
//!
//! @returns The number of bytes successfully read.
//!
//! @throws \c cuda::std::runtime_error if an OS filesystem error occurs.
//! @throws cuda::cuda_error if a CUDA driver error occurs.
//! @throws cuda::cufile_error if a cuFile driver error occurs.
[[nodiscard]] ::cuda::std::size_t read_bytes_sync(
  cufile_ref __file,
  cufile_ref::off_type __foffset,
  ::cuda::std::byte* __dst,
  ::cuda::std::size_t __nbytes,
  ::cuda::std::ptrdiff_t __doffset = 0)
{
  _CCCL_ASSERT(__foffset >= 0, "file offset mustn't be negative");

  const auto __status = ::cuFileRead(__file.get(), __dst, __nbytes, __foffset, __doffset);

  if (__status >= 0)
  {
    return static_cast<::cuda::std::size_t>(__status);
  }

  // handle error states
  constexpr auto __msg = "Failed to read.";
  constexpr auto __api = "cuFileRead";

  const auto __error = static_cast<int>(-__status);

  if (__status == -1)
  {
    errno = 0; // todo: report errno error
    ::cuda::std::__throw_runtime_error("errno error: todo");
  }
  else if (__error < CUFILEOP_BASE_ERR)
  {
    ::cuda::__throw_cuda_error(static_cast<::cudaError_t>(__error), __msg, __api);
  }
  else
  {
    ::cuda::experimental::__throw_cufile_error(static_cast<::CUfileOpError>(__error), __msg, __api);
  }
}

//! @brief Synchronously reads n elements of type \c _Tp from the cuFile handle.
//!
//! @param __file The cuFile handle.
//! @param __foffset The offset of the file to read from. In bytes.
//! @param __dst The buffer to store the data in.
//! @param __n The number of elements to read.
//! @param __doffset The buffer offset in bytes. Defaults to 0.
//!
//! @returns The number of bytes successfully read.
//!
//! @pre \c _Tp must be non-const trivially copyable type.
//!
//! @throws \c cuda::std::runtime_error if an OS filesystem error occurs.
//! @throws cuda::cuda_error if a CUDA driver error occurs.
//! @throws cuda::cufile_error if a cuFile driver error occurs.
template <class _Tp>
[[nodiscard]] ::cuda::std::size_t read_n_sync(
  cufile_ref __file,
  cufile_ref::off_type __foffset,
  _Tp* __dst,
  ::cuda::std::size_t __n,
  ::cuda::std::ptrdiff_t __doffset = 0)
{
  static_assert(!::cuda::std::is_const_v<_Tp>);
  static_assert(::cuda::std::is_trivially_copyable_v<_Tp>);

  // todo: check alignment?

  return ::cuda::experimental::read_bytes_sync(
    __file, __foffset, reinterpret_cast<::cuda::std::byte*>(__dst), __n * sizeof(_Tp), __doffset);
}

//! @brief Synchronously reads elements of type \c _Tp from the cuFile handle.
//!
//! @param __file The cuFile handle.
//! @param __foffset The offset of the file to read from. In bytes.
//! @param __dst The span to fill the data with. Determines the number of bytes to be read.
//!
//! @returns The number of bytes successfully read.
//!
//! @pre \c _Tp must be non-const trivially copyable type.
//!
//! @throws \c cuda::std::runtime_error if an OS filesystem error occurs.
//! @throws cuda::cuda_error if a CUDA driver error occurs.
//! @throws cuda::cufile_error if a cuFile driver error occurs.
template <class _Tp>
[[nodiscard]] ::cuda::std::size_t
read_sync(cufile_ref __file, cufile_ref::off_type __foffset, ::cuda::std::span<_Tp> __dst)
{
  static_assert(!::cuda::std::is_const_v<_Tp>);
  static_assert(::cuda::std::is_trivially_copyable_v<_Tp>);

  // todo: check alignment?

  return ::cuda::experimental::read_bytes_sync(
    __file, __foffset, reinterpret_cast<::cuda::std::byte*>(__dst.data()), __dst.size() * sizeof(_Tp), __doffset);
}

//! @brief Synchronously reads elements of type \c _Tp from the cuFile handle.
//!
//! @param __file The cuFile handle.
//! @param __foffset The offset of the file to read from. In bytes.
//! @param __dst_base The address of the buffer that was registered via \c cuda::cufile_driver.register_buffer(...)
//! call.
//! @param __dst The span to fill the data with. Determines the number of bytes to be read.
//!
//! @returns The number of bytes successfully read.
//!
//! @pre \c _Tp must be non-const trivially copyable type.
//!
//! @throws \c cuda::std::runtime_error if an OS filesystem error occurs.
//! @throws cuda::cuda_error if a CUDA driver error occurs.
//! @throws cuda::cufile_error if a cuFile driver error occurs.
template <class _Tp>
[[nodiscard]] ::cuda::std::size_t
read_sync(cufile_ref __file, cufile_ref::off_type __foffset, _Tp* __dst_base, ::cuda::std::span<_Tp> __dst)
{
  static_assert(!::cuda::std::is_const_v<_Tp>);
  static_assert(::cuda::std::is_trivially_copyable_v<_Tp>);

  // todo: check alignment?

  return ::cuda::experimental::read_bytes_sync(
    __file,
    __foffset,
    reinterpret_cast<::cuda::std::byte*>(__dst_base),
    __dst.size() * sizeof(_Tp),
    __dst.data() - __dst_base);
}

// todo: add overload for a cuda::buffer?

//********************************************************************************************************************//
// Async APIs
//********************************************************************************************************************//

// todo: add overload for a cuda::buffer?
// todo: add overloads for cases when the launch parameters are not know during submission, something like:

enum class cufile_at_submission_t
{
};
inline constexpr cufile_at_submission_t at_submission;

template <class _Tp, class _Up>
inline constexpr bool __is_at_submission_or =
  ::cuda::std::is_same_v<_Tp, cufile_at_submission_t> || ::cuda::std::is_same_v<_Tp, _Up>;

class __cufile_io_submit_result_data
{
  int __state_{};

public:
  static _CCCL_HOST_API void __maybe_destroy(__cufile_io_submit_result_data* __data) noexcept
  {
    if (__data == nullptr)
    {
      return;
    }

    if (cuda::std::atomic_ref{__data->__state_}.fetch_add(1) != 0)
    {
      delete __data;
    }
  }

  static _CCCL_HOST_API void __maybe_destroy_callback(::CUstream, ::CUresult, void* __data) noexcept
  {
    __maybe_destroy(static_cast<__cufile_io_submit_result_data*>(__data));
  }

  ::off_t __foffset_{};
  ::cuda::std::size_t __nbytes_{};
  ::off_t __doffset_{};
  ::ssize_t __result_{};
};

template <class _FOffset, class _NBytes, class _DOffset>
class cufile_io_submit_result
{
  __cufile_io_submit_result_data* __data_{};

  _CCCL_HIDE_FROM_ABI cufile_io_submit_result(__cufile_io_submit_result_data* __data) noexcept
      : __data_{__data}
  {}

public:
  [[nodiscard]] static _CCCL_HOST_API cufile_io_submit_result
  __make_instance(__cufile_io_submit_result_data* __data) noexcept
  {
    return cufile_at_submission_t{__data};
  }

  cufile_io_submit_result(const cufile_io_submit_result&) = delete;

  //! @brief Move constructor.
  cufile_io_submit_result(cufile_io_submit_result&& __other) noexcept
      : __data_{::cuda::std::exchange(__other.__data_, nullptr)}
  {}

  cufile_io_submit_result& operator=(const cufile_io_submit_result&) = delete;

  //! @brief Move assignment operator.
  cufile_io_submit_result& operator=(cufile_io_submit_result&& __other) noexcept
  {
    if (this != ::cuda::std::addressof(__other))
    {
      __cufile_io_submit_result_data::__maybe_destroy(__data_);
      __data_ = ::cuda::std::exchange(__other.__data_, nullptr);
    }
    return *this;
  }

  //! @brief Destructor.
  _CCCL_HOST_API ~cufile_io_submit_result()
  {
    __cufile_io_submit_result_data::__maybe_destroy(__data_);
  }

  //! @brief Sets the file offset. Only allowed if \c cuda::at_submission was passed as the file offset argument to the
  //!        IO call.
  //!
  //! @param __offset The file offset to be used.
  _CCCL_TEMPLATE(class _FOffset2 = _FOffset)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_FOffset2, cufile_at_submission_t>)
  _CCCL_HOST_API void set_file_offset(cufile_ref::off_type __offset) const noexcept
  {
    __data_->__foffset_ = __offset;
  }

  //! @brief Sets the IO operation size in bytes. Only allowed if \c cuda::at_submission was passed as the nbytes
  //!        argument to the IO call.
  //!
  //! @param __nbytes The number of bytes to be read to be used.
  _CCCL_TEMPLATE(class _NBytes2 = _NBytes)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_NBytes2, cufile_at_submission_t>)
  _CCCL_HOST_API void set_size(::cuda::std::size_t __nbytes) const noexcept
  {
    __data_->__nbytes_ = __nbytes;
  }

  //! @brief Sets the buffer offset. Only allowed if \c cuda::at_submission was passed as the buffer offset argument to
  //! the IO call.
  //!
  //! @param __offset The buffer offset to be used.
  _CCCL_TEMPLATE(class _DOffset2 = _DOffset)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_DOffset2, cufile_at_submission_t>)
  _CCCL_HOST_API void set_buffer_offset(::cuda::std::ptrdiff_t __offset) const noexcept
  {
    __data_->__doffset_ = __offset;
  }

  //! @brief Get the number of bytes transferred by the asynchronous IO operation.
  //!
  //! @return The number of bytes transferred.
  [[nodiscard]] _CCCL_HOST_API ::cuda::std::size_t transferred_nbytes() const noexcept
  {
    // todo: check error
    return static_cast<::cuda::std::size_t>(__data_->__result);
  }
};

//! @brief Asynchronously reads data from the cuFile handle.
//!
//! @param __stream The stream to enqueue the read in.
//! @param __file The cuFile handle.
//! @param __foffset The offset of the file to read from. In bytes. If \c cuda::at_submission is passed instead of this
//!                  argument, the argument is expected to be set right before the submission.
//! @param __dst The buffer to store the data in.
//! @param __nbytes The number of bytes to read. If \c cuda::at_submission is passed instead of this argument, the
//!                 argument is expected to be set right before the submission.
//! @param __doffset The buffer offset in bytes. Defaults to 0. If \c cuda::at_submission is passed instead of this
//!                  argument, the argument is expected to be set right before the submission.
//!
//! @returns The number of bytes successfully read.
//!
//! @throws \c cuda::std::runtime_error if an OS filesystem error occurs.
//! @throws cuda::cuda_error if a CUDA driver error occurs.
//! @throws cuda::cufile_error if a cuFile driver error occurs.
_CCCL_TEMPLATE(class _FOffset, class _NBytes, class _DOffset = ::cuda::std::ptrdiff_t)
_CCCL_REQUIRES(__is_at_submission_or<_FOffset, cufile_ref::off_type> _CCCL_AND
                 __is_at_submission_or<_NBytes, ::cuda::std::size_t> _CCCL_AND
                   __is_at_submission_or<_DOffset, ::cuda::std::ptrdiff_t>)
[[nodiscard]] cufile_io_submit_result<_FOffset, _NBytes, _DOffset> read_bytes(
  stream_ref __stream,
  cufile_ref __file,
  _FOffset __foffset,
  ::cuda::std::byte* __dst,
  _NBytes __nbytes,
  _DOffset __doffset = 0)
{
  ::std::unique_ptr<__cufile_io_submit_result_data> __result_data{new __cufile_io_submit_result_data{}};

  if constexpr (::cuda::std::is_same_v<_FOffset, cufile_at_submission_t>)
  {
    __result_data->__foffset_ = __foffset;
  }
  if constexpr (::cuda::std::is_same_v<_NBytes, cufile_at_submission_t>)
  {
    __result_data->__nbytes_ = __nbytes;
  }
  if constexpr (::cuda::std::is_same_v<_DOffset, cufile_at_submission_t>)
  {
    __result_data->__doffset_ = __doffset;
  }

  _CCCL_TRY_CUFILE_API(
    ::cuFileReadAsync, "Failed to submit asynchronous read.", __file.get(), __dst, __nbytes_p, __foffset_p, __doffset_p);

  // Ignore the result of the callback. If the call fails, there will be a small memory leak. We cannot do anything
  // about it, because cuFIle will write the results there.
  // todo: replace with ::cuda::__driver:: call
  const auto __ignore = ::cuStreamAddCallback(
    __stream.get(), __cufile_io_submit_result_data::__maybe_destroy_callback, __result.__data_, 0);
  return cufile_io_submit_result<_FOffset, _NBytes, _DOffset>::__make_instance{__result_data.release()};
}

// //! @brief Asynchronously reads data from the cuFile handle.
// //!
// //! @param __stream The stream to enqueue the read in.
// //! @param __file The cuFile handle.
// //! @param __foffset The offset of the file to read from. In bytes.
// //! @param __dst The buffer to store the data in.
// //! @param __nbytes The number of bytes to read.
// //! @param __doffset The buffer offset in bytes. Defaults to 0.
// //!
// //! @returns The number of bytes successfully read.
// //!
// //! @throws \c cuda::std::runtime_error if an OS filesystem error occurs.
// //! @throws cuda::cuda_error if a CUDA driver error occurs.
// //! @throws cuda::cufile_error if a cuFile driver error occurs.
// [[nodiscard]] read_result read_bytes(
//   stream_ref __stream,
//   cufile_ref __file,
//   cufile_ref::off_type __foffset,
//   ::cuda::std::byte* __dst,
//   ::cuda::std::size_t __nbytes,
//   ::cuda::std::ptrdiff_t __doffset = 0);

// //! @brief Asynchronously reads n elements of type \c _Tp from the cuFile handle.
// //!
// //! @param __stream The stream to enqueue the read in.
// //! @param __file The cuFile handle.
// //! @param __foffset The offset of the file to read from. In bytes.
// //! @param __dst The buffer to store the data in.
// //! @param __n The number of elements to read.
// //! @param __doffset The buffer offset in bytes. Defaults to 0.
// //!
// //! @returns The number of bytes successfully read.
// //!
// //! @pre \c _Tp must be non-const trivially copyable type.
// //!
// //! @throws \c cuda::std::runtime_error if an OS filesystem error occurs.
// //! @throws cuda::cuda_error if a CUDA driver error occurs.
// //! @throws cuda::cufile_error if a cuFile driver error occurs.
// template <class _Tp>
// [[nodiscard]] ::cuda::std::size_t read_n(
//   stream_ref __stream,
//   cufile_ref __file,
//   cufile_ref::off_type __foffset,
//   _Tp* __dst,
//   ::cuda::std::size_t __n,
//   ::cuda::std::ptrdiff_t __doffset = 0);

// //! @brief Asynchronously reads elements of type \c _Tp from the cuFile handle.
// //!
// //! @param __stream The stream to enqueue the read in.
// //! @param __file The cuFile handle.
// //! @param __foffset The offset of the file to read from. In bytes.
// //! @param __dst The span to fill the data with. Determines the number of bytes to be read.
// //!
// //! @returns The number of bytes successfully read.
// //!
// //! @pre \c _Tp must be non-const trivially copyable type.
// //!
// //! @throws \c cuda::std::runtime_error if an OS filesystem error occurs.
// //! @throws cuda::cuda_error if a CUDA driver error occurs.
// //! @throws cuda::cufile_error if a cuFile driver error occurs.
// template <class _Tp>
// [[nodiscard]] ::cuda::std::size_t
// read(stream_ref __stream, cufile_ref __file, cufile_ref::off_type __foffset, ::cuda::std::span<_Tp> __dst);

// //! @brief Asynchronously reads elements of type \c _Tp from the cuFile handle.
// //!
// //! @param __stream The stream to enqueue the read in.
// //! @param __file The cuFile handle.
// //! @param __foffset The offset of the file to read from. In bytes.
// //! @param __dst_base The address of the buffer that was registered via \c cuda::cufile_driver.register_buffer(...)
// //! call.
// //! @param __dst The span to fill the data with. Determines the number of bytes to be read.
// //!
// //! @returns The number of bytes successfully read.
// //!
// //! @pre \c _Tp must be non-const trivially copyable type.
// //!
// //! @throws \c cuda::std::runtime_error if an OS filesystem error occurs.
// //! @throws cuda::cuda_error if a CUDA driver error occurs.
// //! @throws cuda::cufile_error if a cuFile driver error occurs.
// template <class _Tp>
// [[nodiscard]] ::cuda::std::size_t read(
//   stream_ref __stream, cufile_ref __file, cufile_ref::off_type __foffset, _Tp* __dst_base, ::cuda::std::span<_Tp>
//   __dst);

} // namespace cuda::experimental
