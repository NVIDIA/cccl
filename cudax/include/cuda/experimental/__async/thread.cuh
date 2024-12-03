//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_THREAD
#define __CUDAX_ASYNC_DETAIL_THREAD

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__detail/config.cuh>

#include <thread>

#if defined(__CUDACC__)
#  include <nv/target>
#  define _CUDAX_FOR_HOST_OR_DEVICE(_FOR_HOST, _FOR_DEVICE) NV_IF_TARGET(NV_IS_HOST, _FOR_HOST, _FOR_DEVICE)
#else
#  define _CUDAX_FOR_HOST_OR_DEVICE(_FOR_HOST, _FOR_DEVICE) {_NV_EVAL _FOR_HOST}
#endif

namespace cuda::experimental::__async
{
#if defined(__CUDA_ARCH__)
using __thread_id = int;
#elif _CCCL_COMPILER(NVHPC)
struct __thread_id
{
  union
  {
    ::std::thread::id __host_;
    int __device_;
  };

  _CUDAX_API __thread_id() noexcept
      : __host_()
  {}
  _CUDAX_API __thread_id(::std::thread::id __host) noexcept
      : __host_(__host)
  {}
  _CUDAX_API __thread_id(int __device) noexcept
      : __device_(__device)
  {}

  _CUDAX_API friend bool operator==(const __thread_id& __self, const __thread_id& __other) noexcept
  {
    _CUDAX_FOR_HOST_OR_DEVICE((return __self.__host_ == __other.__host_;),
                              (return __self.__device_ == __other.__device_;))
  }

  _CUDAX_API friend bool operator!=(const __thread_id& __self, const __thread_id& __other) noexcept
  {
    return !(__self == __other);
  }
};
#else
using __thread_id = ::std::thread::id;
#endif

inline _CUDAX_API __thread_id __this_thread_id() noexcept
{
  _CUDAX_FOR_HOST_OR_DEVICE((return ::std::this_thread::get_id();),
                            (return static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);))
}

inline _CUDAX_API void __this_thread_yield() noexcept
{
  _CUDAX_FOR_HOST_OR_DEVICE((::std::this_thread::yield();), (void();))
}
} // namespace cuda::experimental::__async

#endif
