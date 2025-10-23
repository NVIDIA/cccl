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

#include <cuda/__functional/get_device_address.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__cstddef/byte.h>
#include <cuda/std/__type_traits/is_const.h>
#include <cuda/std/__type_traits/is_trivially_copyable.h>
#include <cuda/std/atomic>
#include <cuda/std/detail/libcxx/include/stdexcept>
#include <cuda/std/span>

#include <cuda/experimental/__cufile/cufile_ref.cuh>
#include <cuda/experimental/__cufile/enqueue_result.cuh>
#include <cuda/experimental/__cufile/exception.cuh>
#include <cuda/experimental/__memory_resource/pinned_memory_resource.cuh>

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

  if constexpr (!::cuda::std::is_same_v<_FOffset, cufile_at_submission_t>)
  {
    __result_data->__foffset_ = static_cast<::off_t>(__foffset);
  }
  if constexpr (!::cuda::std::is_same_v<_NBytes, cufile_at_submission_t>)
  {
    __result_data->__nbytes_ = static_cast<::cuda::std::size_t>(__nbytes);
  }
  if constexpr (!::cuda::std::is_same_v<_DOffset, cufile_at_submission_t>)
  {
    __result_data->__doffset_ = static_cast<::off_t>(__doffset);
  }

  __result_data->__pinned_result_ptr_ = ::cuda::__driver::__mallocFromPoolAsync(
    sizeof(::ssize_t), ::cuda::experimental::__get_default_host_pinned_pool(), __stream.get());

  try
  {
    _CCCL_TRY_CUFILE_API(
      ::cuFileReadAsync,
      "Failed to submit asynchronous read.",
      __file.get(),
      __dst,
      &__result_data->__nbytes_,
      &__result_data->__foffset_,
      &__result_data->__doffset_,
      reinterpret_cast<::ssize_t*>(__result_data->__pinned_result_ptr_),
      __stream.get());
  }
  catch (...)
  {
    (void) ::cuda::__driver::__freeAsyncNoThrow(__result_data->__pinned_result_ptr, __stream.get());
    throw;
  }

  (void) ::cuda::__driver::__streamAddCallbackNoThrow(
    __stream.get(),
    __cufile_io_submit_result_data::__copy_result_and_decreas_ref_count_callback,
    __result_data.get(),
    0);

  (void) ::cuda::__driver::__freeAsyncNoThrow(__result_data->__pinned_result_ptr, __stream.get());

  return cufile_io_submit_result<_FOffset, _NBytes, _DOffset>::__make_instance{__result_data.release()};
}

//! @brief Asynchronously reads n elements of type \c _Tp from the cuFile handle.
//!
//! @param __stream The stream to enqueue the read in.
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
[[nodiscard]] cufile_enqueue_result_default read_n(
  stream_ref __stream,
  cufile_ref __file,
  cufile_ref::off_type __foffset,
  _Tp* __dst,
  ::cuda::std::size_t __n,
  ::cuda::std::ptrdiff_t __doffset = 0)
{
  static_assert(!::cuda::std::is_const_v<_Tp>);
  static_assert(::cuda::std::is_trivially_copyable_v<_Tp>);

  // todo: check alignment?

  return ::cuda::experimental::read_bytes(
    __stream, __file, __foffset, reinterpret_cast<::cuda::std::byte*>(__dst), __n * sizeof(_Tp), __doffset);
}

//! @brief Asynchronously reads elements of type \c _Tp from the cuFile handle.
//!
//! @param __stream The stream to enqueue the read in.
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
[[nodiscard]] cufile_enqueue_result_default
read(stream_ref __stream, cufile_ref __file, cufile_ref::off_type __foffset, ::cuda::std::span<_Tp> __dst)
{
  static_assert(!::cuda::std::is_const_v<_Tp>);
  static_assert(::cuda::std::is_trivially_copyable_v<_Tp>);

  // todo: check alignment?

  return ::cuda::experimental::read_bytes(
    __stream,
    __file,
    __foffset,
    reinterpret_cast<::cuda::std::byte*>(__dst.data()),
    __dst.size() * sizeof(_Tp),
    __doffset);
}

//! @brief Asynchronously reads elements of type \c _Tp from the cuFile handle.
//!
//! @param __stream The stream to enqueue the read in.
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
[[nodiscard]] cufile_enqueue_result_default read(
  stream_ref __stream, cufile_ref __file, cufile_ref::off_type __foffset, _Tp* __dst_base, ::cuda::std::span<_Tp> __dst)
{
  static_assert(!::cuda::std::is_const_v<_Tp>);
  static_assert(::cuda::std::is_trivially_copyable_v<_Tp>);

  // todo: check alignment?

  return ::cuda::experimental::read_bytes(
    __stream,
    __file,
    __foffset,
    reinterpret_cast<::cuda::std::byte*>(__dst_base),
    __dst.size() * sizeof(_Tp),
    __dst.data() - __dst_base);
}

} // namespace cuda::experimental
