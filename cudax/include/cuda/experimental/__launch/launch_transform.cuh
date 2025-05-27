//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__LAUNCH_LAUNCH_TRANSFORM
#define _CUDAX__LAUNCH_LAUNCH_TRANSFORM
#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/utility>
#include <cuda/stream_ref>

#include <cuda/experimental/__detail/utility.cuh>

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_STD_VER >= 2017
namespace cuda::experimental
{
namespace __transforms
{
// Launch transform:
//
// The launch transform is a mechanism to transform arguments passed to the
// cudax::launch API prior to actually launching a kernel. This is useful for
// example, to automatically convert contiguous ranges into spans. It is also
// useful for executing per-argument actions before and after the kernel launch.
// A host_vector might want a pre-launch action to copy data from host to device
// and a post-launch action to copy data back from device to host.
//
// The launch transform happens in two steps. First, `cudax::launch` calls
// __launch_transform on each argument. If the argument has hooked the
// __launch_transform customization point, this returns a temporary object that
// has the pre-launch action in its constructor and the post-launch action in
// its destructor. The temporaries are all constructed before launching the
// kernel, and they are all destroyed immediately after, at the end of the full
// expression that performs the launch. If the `cudax::launch` argument has not
// hooked the __launch_transform customization point, then the argument is
// passed through.
//
// The result of __launch_transform is not necessarily what is passed to the
// kernel though. If __launch_transform returns an object with a
// `.kernel_transform()` member function, then `cudax::launch` will call that
// function. Its result is what gets passed as an argument to the kernel. If the
// __launch_transform result does not have a `.kernel_transform()` member
// function, then the __launch_transform result itself is passed to the kernel.

void __cudax_launch_transform();

// Types that want to customize `__launch_transform` should define overloads of
// __cudax_launch_transform that are find-able by ADL.
template <typename _Arg>
using __launch_transform_direct_result_t =
  decltype(__cudax_launch_transform(::cuda::stream_ref{}, _CUDA_VSTD::declval<_Arg>()));

struct __launch_fn
{
  template <typename _Arg>
  [[nodiscard]] decltype(auto) operator()([[maybe_unused]] ::cuda::stream_ref __stream, _Arg&& __arg) const
  {
    if constexpr (_CUDA_VSTD::_IsValidExpansion<__launch_transform_direct_result_t, _Arg>::value)
    {
      // This call is unqualified to allow ADL
      return __cudax_launch_transform(__stream, _CUDA_VSTD::forward<_Arg>(__arg));
    }
    else
    {
      return _CUDA_VSTD::forward<_Arg>(__arg);
    }
  }
};

template <typename _Arg>
using __launch_transform_result_t = decltype(__launch_fn{}(::cuda::stream_ref{}, _CUDA_VSTD::declval<_Arg>()));

template <typename _Arg>
using __kernel_transform_direct_result_t = decltype(_CUDA_VSTD::declval<_Arg>().kernel_transform());

struct __kernel_fn
{
  template <typename _Arg>
  [[nodiscard]] decltype(auto) operator()(_Arg&& __arg) const
  {
    if constexpr (_CUDA_VSTD::_IsValidExpansion<__kernel_transform_direct_result_t, _Arg>::value)
    {
      return _CUDA_VSTD::forward<_Arg>(__arg).kernel_transform();
    }
    else
    {
      return _CUDA_VSTD::forward<_Arg>(__arg);
    }
  }
};

template <typename _Arg>
using __kernel_transform_result_t = decltype(__kernel_fn{}(_CUDA_VSTD::declval<_Arg>()));

} // namespace __transforms

using __transforms::__kernel_transform_result_t;
using __transforms::__launch_transform_result_t;

_CCCL_GLOBAL_CONSTANT __transforms::__launch_fn __launch_transform{};
_CCCL_GLOBAL_CONSTANT __transforms::__kernel_fn __kernel_transform{};

template <typename _Arg>
using kernel_arg_t = _CUDA_VSTD::decay_t<__kernel_transform_result_t<__launch_transform_result_t<_Arg>>>;

} // namespace cuda::experimental

#endif // _CCCL_STD_VER >= 2017

#include <cuda/std/__cccl/epilogue.h>

#endif // !_CUDAX__LAUNCH_LAUNCH_TRANSFORM
