//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__FILE_CUFILE_API
#define _CUDAX__FILE_CUFILE_API

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__exception/cuda_error.h>
#include <cuda/std/__utility/forward.h>

#include <cstdio>

// Protect against redefinition of IS_CUDA_ERR
#pragma push_macro("IS_CUDA_ERR")

#include <cufile.h>

#pragma pop_macro("IS_CUDA_ERR")

namespace cuda::experimental::detail
{

_CCCL_NORETURN void __throw_cufile_error(::CUfileOpError __error, const char* __msg)
{
  static constexpr ::cuda::std::size_t __buffer_size = 256;

  char __buffer[__buffer_size];
  ::std::snprintf(__buffer, __buffer_size, "CUfileOpError %d: %s", __error, __msg);

  throw ::cuda::cuda_error(__buffer);
}

_CCCL_NORETURN void __throw_cufile_error(::CUfileError_t __error, const char* __msg)
{
  if (__error.err == CU_FILE_CUDA_DRIVER_ERROR)
  {
    __throw_cuda_error(static_cast<cudaError_t>(__error.cu_err), __msg);
  }

  __throw_cufile_error(__error.err, __msg);
}

void __check_cufile_error(::CUfileOpError __error, const char* __msg)
{
  if (IS_CUFILE_ERR(__error))
  {
    __throw_cufile_error(__error, __msg);
  }
}

void __check_cufile_error(::CUfileError_t __error, const char* __msg)
{
  if (IS_CUFILE_ERR(__error.err))
  {
    __throw_cufile_error(__error, __msg);
  }
}

template <bool __throw_on_error = true, class Fn, class... Args>
::CUfileError_t __call_cufile_api(Fn&& __fn, const char* __err_msg, Args&&... __args)
{
  ::CUfileError_t __error = __fn(::cuda::std::forward<Args>(__args)...);
  if constexpr (__throw_on_error)
  {
    __check_cufile_error(__error, __err_msg);
  }
  return __error;
}

::CUfileHandle_t __cufile_handle_register(::CUfileDescr_t* __descr)
{
  ::CUfileHandle_t __fh;
  __call_cufile_api(::cuFileHandleRegister, "Failed to register a CUfile file handle", &__fh, __descr);
  return __fh;
}

template <bool __throw_on_error = true>
void __cufile_handle_deregister(::CUfileHandle_t __fh)
{
  __call_cufile_api<__throw_on_error>(::cuFileHandleDeregister, "Failed to deregister a CUfile file handle", __fh);
}

void __cufile_buf_register(const void* __buf, ::cuda::std::size_t __size, int __flags)
{
  __call_cufile_api(::cuFileBufRegister, "Failed to register a CUfile buffer", __buf, __size, __flags);
}

template <bool __throw_on_error = true>
void __cufile_buf_deregister(const void* __buf)
{
  __call_cufile_api<__throw_on_error>(::cuFileBufDeregister, "Failed to deregister a CUfile buffer", __buf);
}

::ssize_t __cufile_read(
  ::CUfileHandle_t __fh, void* __buf, ::cuda::std::size_t __size, ::off_t __file_offset, ::off_t __buffer_offset)
{
  ::ssize_t __nbytes_read = ::cuFileRead(__fh, __buf, __size, __file_offset, __buffer_offset);
  if (__nbytes_read < -1)
  {
    __throw_cufile_error(static_cast<::CUfileOpError>(-__nbytes_read), "Failed to read from a CUfile file");
  }
  return __nbytes_read;
}

::ssize_t __cufile_write(
  ::CUfileHandle_t __fh, const void* __buf, ::cuda::std::size_t __size, ::off_t __file_offset, ::off_t __buffer_offset)
{
  ::ssize_t __nbytes_written = ::cuFileWrite(__fh, __buf, __size, __file_offset, __buffer_offset);
  if (__nbytes_written < -1)
  {
    __throw_cufile_error(static_cast<::CUfileOpError>(-__nbytes_written), "Failed to write to a CUfile file");
  }
  return __nbytes_written;
}

void __cufile_driver_open()
{
  __call_cufile_api(::cuFileDriverOpen, "Failed to open the CUfile driver");
}

template <bool __throw_on_error = true>
void __cufile_driver_close()
{
  __call_cufile_api<__throw_on_error>(::cuFileDriverClose, "Failed to close the CUfile driver");
}

long __cufile_use_count()
{
  return ::cuFileUseCount();
}

::CUfileDrvProps_t __cufile_driver_get_properties()
{
  ::CUfileDrvProps_t __props;
  __call_cufile_api(::cuFileDriverGetProperties, "Failed to get CUfile driver properties", &__props);
  return __props;
}

void __cufile_driver_set_poll_mode(int __poll_mode, ::cuda::std::size_t __poll_threshold_size)
{
  __call_cufile_api(
    ::cuFileDriverSetPollMode, "Failed to set CUfile driver poll mode", __poll_mode, __poll_threshold_size);
}

void __cufile_driver_set_max_direct_io_size(::cuda::std::size_t __max_direct_io_size)
{
  __call_cufile_api(
    ::cuFileDriverSetMaxDirectIOSize, "Failed to set CUfile driver maximum direct I/O size", __max_direct_io_size);
}

void __cufile_driver_set_max_cache_size(::cuda::std::size_t __max_cache_size)
{
  __call_cufile_api(::cuFileDriverSetMaxCacheSize, "Failed to set CUfile driver maximum cache size", __max_cache_size);
}

void __cufile_driver_set_max_pinned_mem_size(::cuda::std::size_t __max_pinned_mem_size)
{
  __call_cufile_api(
    ::cuFileDriverSetMaxPinnedMemSize, "Failed to set CUfile driver maximum pinned memory size", __max_pinned_mem_size);
}

// todo: add cuFileBatch API

void __check_cufile_read_async_error(void* __nbytes_read_ptr)
{
  ::ssize_t __nbytes_read = *static_cast<::ssize_t*>(__nbytes_read_ptr);
  if (__nbytes_read < -1)
  {
    __throw_cufile_error(static_cast<::CUfileOpError>(-__nbytes_read),
                         "Failed an asynchronous read from a CUfile file");
  }
}

void __cufile_read_async(
  ::CUfileHandle_t __fh,
  void* __buf,
  const ::cuda::std::size_t* __size_ptr,
  const ::off_t* __file_offset_ptr,
  const ::off_t* __buffer_offset_ptr,
  ::ssize_t* __nbytes_read_ptr,
  ::CUstream __stream)
{
  static constexpr const char* __msg = "Failed to submit asynchronous read from a CUfile file";

  __call_cufile_api(
    ::cuFileReadAsync,
    __msg,
    __fh,
    __buf,
    const_cast<::cuda::std::size_t*>(__size_ptr),
    const_cast<::off_t*>(__file_offset_ptr),
    const_cast<::off_t*>(__buffer_offset_ptr),
    __nbytes_read_ptr,
    __stream);
  auto __error =
    ::cudaLaunchHostFunc(__stream, ::cuda::experimental::detail::__check_cufile_read_async_error, __nbytes_read_ptr);
  if (__error != cudaSuccess)
  {
    ::cuda::__throw_cuda_error(__error, __msg);
  }
}

void __check_cufile_write_async_error(void* __nbytes_written_ptr)
{
  ::ssize_t __nbytes_written = *static_cast<::ssize_t*>(__nbytes_written_ptr);
  if (__nbytes_written < -1)
  {
    __throw_cufile_error(static_cast<::CUfileOpError>(-__nbytes_written),
                         "Failed an asynchronous write to a CUfile file");
  }
}

void __cufile_write_async(
  ::CUfileHandle_t __fh,
  const void* __buf,
  const ::cuda::std::size_t* __size_ptr,
  const ::off_t* __file_offset_ptr,
  const ::off_t* __buffer_offset_ptr,
  ::ssize_t* __nbytes_written_ptr,
  ::CUstream __stream)
{
  static constexpr const char* __msg = "Failed to submit asynchronous write to a CUfile file";

  __call_cufile_api(
    ::cuFileWriteAsync,
    __msg,
    __fh,
    __buf,
    const_cast<::cuda::std::size_t*>(__size_ptr),
    const_cast<::off_t*>(__file_offset_ptr),
    const_cast<::off_t*>(__buffer_offset_ptr),
    __nbytes_written_ptr,
    __stream);
  auto __error = ::cudaLaunchHostFunc(
    __stream, ::cuda::experimental::detail::__check_cufile_write_async_error, __nbytes_written_ptr);
  if (__error != cudaSuccess)
  {
    ::cuda::__throw_cuda_error(__error, __msg);
  }
}

void __cufile_stream_register(::CUstream __stream, unsigned flags)
{
  __call_cufile_api(::cuFileStreamRegister, "Failed to register a CUfile stream", __stream, flags);
}

template <bool __throw_on_error = true>
void __cufile_stream_deregister(::CUstream __stream)
{
  __call_cufile_api<__throw_on_error>(::cuFileStreamDeregister, "Failed to deregister a CUfile stream", __stream);
}

} // namespace cuda::experimental::detail

#endif // _CUDAX__FILE_CUFILE_API
