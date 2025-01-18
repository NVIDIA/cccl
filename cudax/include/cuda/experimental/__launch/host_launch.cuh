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

#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/tuple>
#include <cuda/stream_ref>

namespace cuda::experimental
{

template <typename _Fn>
void __stream_callback_caller(cudaStream_t, cudaError_t __status, void* __fn_ptr)
{
  auto __casted_fn = static_cast<_Fn>(__fn_ptr);
  if (__status == cudaSuccess)
  {
    (*__casted_fn)();
  }
  delete __casted_fn;
}

template <typename _Fn, typename... _Args>
void host_launch(stream_ref __stream, _Fn __fn, _Args... __args)
{
  auto __fn_lambda =
    new auto([__fn = _CUDA_VSTD::move(__fn), __args_tuple = _CUDA_VSTD::make_tuple(_CUDA_VSTD::move(__args)...)]() {
      _CUDA_VSTD::apply(__fn, __args_tuple);
    });

  _CCCL_TRY_CUDA_API(
    cudaStreamAddCallback,
    "Failed to launch host function",
    __stream.get(),
    __stream_callback_caller<decltype(__fn_lambda)>,
    static_cast<void*>(__fn_lambda),
    0);
}

template <typename _Fn>
void __host_func_launcher(void* __fn_ptr)
{
  auto __casted_fn = static_cast<_Fn>(__fn_ptr);
  (*__casted_fn)();
}

template <typename _Fn, typename... _Args>
void host_launch_by_reference(stream_ref __stream, _Fn& __fn)
{
  _CCCL_TRY_CUDA_API(
    cudaLaunchHostFunc, "Failed to launch host function", __stream.get(), __host_func_launcher<decltype(&__fn)>, &__fn);
}

} // namespace cuda::experimental

#endif // !_CUDAX__LAUNCH_HOST_LAUNCH
