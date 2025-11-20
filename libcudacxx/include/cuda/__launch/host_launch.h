//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___LAUNCH_HOST_LAUNCH_H
#define _CUDA___LAUNCH_HOST_LAUNCH_H

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#  include <cuda/__driver/driver_api.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__functional/reference_wrapper.h>
#  include <cuda/std/__type_traits/decay.h>
#  include <cuda/std/__type_traits/is_move_constructible.h>
#  include <cuda/std/__utility/forward.h>
#  include <cuda/std/tuple>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <class _Callable, class... _Args>
struct __stream_callback_data
{
  _Callable __callable_;
  ::cuda::std::tuple<_Args...> __args_;
};

template <class _CallbackData>
_CCCL_HOST_API inline void CUDA_CB __stream_callback_launcher(::CUstream, ::CUresult __status, void* __data_ptr)
{
  auto* __casted_data_ptr = static_cast<_CallbackData*>(__data_ptr);
  if (__status == ::CUDA_SUCCESS)
  {
    (void) ::cuda::std::apply(__casted_data_ptr->__callable_, ::cuda::std::move(__casted_data_ptr->__args_));
  }
  delete __casted_data_ptr;
}

//! @brief Launches a host callable to be executed in stream order on the provided stream
//!
//! Callable and arguments are copied into an internal dynamic allocation to preserve them
//! until the asynchronous call happens. Lambda capture or reference_wrapper can be used if
//! there is a need to pass something by reference.
//!
//! Callable must not call any APIs from cuda, thrust or cub namespaces.
//! It must not call into CUDA Runtime or Driver APIs. It also can't depend on another
//! thread that might block on any asynchronous CUDA work.
//!
//! @param __stream Stream to launch the host function on
//! @param __callable Host function or callable object to call in stream order
//! @param __args Arguments to call the supplied callable with
template <class _Callable, class... _Args>
_CCCL_HOST_API void host_launch(stream_ref __stream, _Callable __callable, _Args... __args)
{
  static_assert(::cuda::std::is_invocable_v<_Callable, _Args...>,
                "Callable can't be called with the supplied arguments");
  static_assert(::cuda::std::is_move_constructible_v<_Callable>, "The callable must be move constructible");
  static_assert((::cuda::std::is_move_constructible_v<_Args> && ...),
                "All callback arguments must be move constructible");

  using _CallbackData                = __stream_callback_data<_Callable, _Args...>;
  _CallbackData* __callback_data_ptr = new _CallbackData{::cuda::std::move(__callable), {::cuda::std::move(__args)...}};

  // We use the callback here to have it execute even on stream error, because it needs to free the above allocation
  ::cuda::__driver::__streamAddCallback(__stream.get(), __stream_callback_launcher<_CallbackData>, __callback_data_ptr);
}

template <class _Callable>
_CCCL_HOST_API inline void CUDA_CB __host_func_launcher(void* __callable_ptr)
{
  (*static_cast<_Callable*>(__callable_ptr))();
}

//! @brief Launches a host callable to be executed in stream order on the provided stream
//!
//! Callable will be called using the supplied reference. If the callable was destroyed
//! or moved by the time it is asynchronously called the behavior is undefined.
//!
//! Callable must not call any APIs from cuda, thrust or cub namespaces.
//! It must not call into CUDA Runtime or Driver APIs. It also can't depend on another
//! thread that might block on any asynchronous CUDA work.
//!
//! @param __stream Stream to launch the host function on
//! @param __callable A reference to a host function or callable object to call in stream order
template <class _Callable>
_CCCL_HOST_API void host_launch(stream_ref __stream, ::cuda::std::reference_wrapper<_Callable> __callable)
{
  static_assert(::cuda::std::is_invocable_v<_Callable>, "Callable in reference_wrapper can't take any arguments");
  ::cuda::__driver::__launchHostFunc(
    __stream.get(), __host_func_launcher<_Callable>, ::cuda::std::addressof(__callable.get()));
}
_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif // !_CUDA___LAUNCH_HOST_LAUNCH_H
