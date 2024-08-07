//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

#if _CCCL_STD_VER >= 2017
namespace cuda::experimental::detail
{
// Types should define overloads of __cudax_launch_transform that are find-able
// by ADL in order to customize how cudax::launch handles that type. The
// overload below, which simply returns the argument unmodified, is the overload
// that gets chosen if no other overload matches. It takes __ignore as the first
// argument to make this overload less preferred than other overloads that take
// a stream_ref as the first argument.
template <typename _Arg>
_CCCL_NODISCARD constexpr _Arg&& __cudax_launch_transform(__ignore, _Arg&& __arg) noexcept
{
  return _CUDA_VSTD::forward<_Arg>(__arg);
}

template <typename _Arg>
using __launch_transform_direct_result_t =
  decltype(__cudax_launch_transform(::cuda::stream_ref{}, _CUDA_VSTD::declval<_Arg>()));

struct __fn
{
  template <typename _Arg>
  _CCCL_NODISCARD __launch_transform_direct_result_t<_Arg> operator()(::cuda::stream_ref __stream, _Arg&& __arg) const
  {
    // This call is unqualified to allow ADL
    return __cudax_launch_transform(__stream, _CUDA_VSTD::forward<_Arg>(__arg));
  }
};

template <typename _Arg, typename _Enable = void>
struct __launch_transform_result
{
  using type = _CUDA_VSTD::decay_t<__launch_transform_direct_result_t<_Arg>>;
};

template <typename _Arg>
struct __launch_transform_result<
  _Arg,
  _CUDA_VSTD::void_t<typename _CUDA_VSTD::decay_t<__launch_transform_direct_result_t<_Arg>>::__launch_transform_result>>
{
  using type = typename _CUDA_VSTD::decay_t<__launch_transform_direct_result_t<_Arg>>::__launch_transform_result;
};

template <typename _Arg>
using __launch_transform_result_t = typename __launch_transform_result<_Arg>::type;

_CCCL_GLOBAL_CONSTANT __fn __launch_transform{};
} // namespace cuda::experimental::detail

#endif // _CCCL_STD_VER >= 2017
#endif // !_CUDAX__LAUNCH_LAUNCH_TRANSFORM
