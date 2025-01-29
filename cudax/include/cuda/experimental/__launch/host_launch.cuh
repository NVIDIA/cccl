//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__LAUNCH_HOST_LAUNCH
#define _CUDAX__LAUNCH_HOST_LAUNCH
#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__functional/reference_wrapper.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/tuple>
#include <cuda/stream_ref>

namespace cuda::experimental
{

template <typename _CallablePtr>
void __stream_callback_caller(cudaStream_t, cudaError_t __status, void* __callable_ptr)
{
  auto __casted_callable = static_cast<_CallablePtr>(__callable_ptr);
  if (__status == cudaSuccess)
  {
    (*__casted_callable)();
  }
  delete __casted_callable;
}

//! @brief Launches a host callable to be executed in stream order on the provided stream
//!
//! Callable and arguments are copied into an internal dynamic allocation to preserve them
//! until the asynchronous call happens. Lambda capture or reference_wrapper can be used if
//! there is a need to pass something by reference.
//!
//! @param __stream Stream to launch the host function on
//! @param __callable Host function or callable object to call in stream order
//! @param __args Arguments to call the supplied callable with
template <typename _Callable, typename... _Args>
void host_launch(stream_ref __stream, _Callable __callable, _Args... __args)
{
  static_assert(_CUDA_VSTD::is_invocable_v<_Callable, _Args...>,
                "Callable can't be called with the supplied arguments");
  auto __lambda_ptr = new auto([__callable   = _CUDA_VSTD::move(__callable),
                                __args_tuple = _CUDA_VSTD::make_tuple(_CUDA_VSTD::move(__args)...)]() mutable {
    _CUDA_VSTD::apply(__callable, __args_tuple);
  });

  _CCCL_TRY_CUDA_API(
    cudaStreamAddCallback,
    "Failed to launch host function",
    __stream.get(),
    __stream_callback_caller<decltype(__lambda_ptr)>,
    static_cast<void*>(__lambda_ptr),
    0);
}

template <typename _CallablePtr>
void __host_func_launcher(void* __callable_ptr)
{
  auto __casted_callable = static_cast<_CallablePtr>(__callable_ptr);
  (*__casted_callable)();
}

//! @brief Launches a host callable to be executed in stream order on the provided stream
//!
//! Callable will be called using the supplied reference. If the callable was destroyed
//! or moved by the time it is asynchronously called the behavior is undefined.
//!
//! Callable can't take any arguments, if some additional state is required a lambda can be used
//! to capture it.
//!
//! @param __stream Stream to launch the host function on
//! @param __callable A reference to a host function or callable object to call in stream order
template <typename _Callable, typename... _Args>
void host_launch(stream_ref __stream, ::cuda::std::reference_wrapper<_Callable> __callable)
{
  static_assert(_CUDA_VSTD::is_invocable_v<_Callable>, "Callable in reference_wrapper can't take any arguments");
  _CCCL_TRY_CUDA_API(
    cudaLaunchHostFunc,
    "Failed to launch host function",
    __stream.get(),
    __host_func_launcher<_Callable*>,
    &__callable.get());
}

} // namespace cuda::experimental

#endif // !_CUDAX__LAUNCH_HOST_LAUNCH
