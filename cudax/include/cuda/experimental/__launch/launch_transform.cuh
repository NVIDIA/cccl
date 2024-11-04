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
namespace cuda::experimental
{
namespace detail
{
// Types should define overloads of __cudax_launch_transform that are find-able
// by ADL in order to customize how cudax::launch handles that type.
template <typename _Arg>
using __launch_transform_direct_result_t =
  decltype(__cudax_launch_transform(::cuda::stream_ref{}, _CUDA_VSTD::declval<_Arg>()));

struct __fn
{
  template <typename _Arg>
  _CCCL_NODISCARD decltype(auto) operator()(::cuda::stream_ref __stream, _Arg&& __arg) const
  {
    if constexpr (::cuda::std::_IsValidExpansion<__launch_transform_direct_result_t, _Arg>::value)
    {
      // This call is unqualified to allow ADL
      return __cudax_launch_transform(__stream, _CUDA_VSTD::forward<_Arg>(__arg));
    }
    else
    {
      (void) __stream;
      return _CUDA_VSTD::forward<_Arg>(__arg);
    }
  }
};

template <typename _Arg>
using __launch_transform_result_t = decltype(__fn{}(::cuda::stream_ref{}, _CUDA_VSTD::declval<_Arg>()));

template <typename _Arg, typename _Enable = void>
struct __as_copy_arg
{
  using type = __launch_transform_result_t<_Arg>;
};

// Copy needs to know if original value is a reference
template <typename _Arg>
struct __as_copy_arg<_Arg,
                     _CUDA_VSTD::void_t<typename _CUDA_VSTD::decay_t<__launch_transform_result_t<_Arg>>::__as_kernel_arg>>
{
  using type = typename _CUDA_VSTD::decay_t<__launch_transform_result_t<_Arg>>::__as_kernel_arg;
};

template <typename _Arg>
using __as_copy_arg_t = typename detail::__as_copy_arg<_Arg>::type;

// While kernel argument can't be a reference
template <typename _Arg>
struct __as_kernel_arg
{
  using type = _CUDA_VSTD::decay_t<typename __as_copy_arg<_Arg>::type>;
};

_CCCL_GLOBAL_CONSTANT __fn __launch_transform{};
} // namespace detail

template <typename _Arg>
using as_kernel_arg_t = typename detail::__as_kernel_arg<_Arg>::type;

} // namespace cuda::experimental

#endif // _CCCL_STD_VER >= 2017
#endif // !_CUDAX__LAUNCH_LAUNCH_TRANSFORM
