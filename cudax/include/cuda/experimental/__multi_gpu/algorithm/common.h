// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_COMMON_H
#define _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_COMMON_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__container/buffer.h>
#include <cuda/__device/device_ref.h>
#include <cuda/__functional/lazy_call_or.h>
#include <cuda/__memory_pool/device_memory_pool.h>
#include <cuda/__memory_resource/get_memory_resource.h>
#include <cuda/__runtime/api_wrapper.h>
#include <cuda/__runtime/ensure_current_context.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/__utility/no_init.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/readable_traits.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__type_traits/is_trivially_constructible.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental::__detail
{
#define __CUDAX_MULTI_GPU_DISPATCH(__logical_device, __call, ...)                              \
  do                                                                                           \
  {                                                                                            \
    const auto __cur_context = ::cuda::__ensure_current_context{(__logical_device).context()}; \
    _CCCL_TRY_CUDA_API(__call, "performing" #__call "(" #__VA_ARGS__ ")", __VA_ARGS__);        \
  } while (0)

template <class _Env>
[[nodiscard]]
_CCCL_HOST_API constexpr decltype(auto) __resource_from_env(const _Env& __env, ::cuda::device_ref __device)
{
  return ::cuda::__lazy_call_or(
    ::cuda::mr::get_memory_resource,
    [&] {
      return ::cuda::device_default_memory_pool(__device);
    },
    __env);
}

template <class _Env>
using __resource_type_for = ::cuda::std::remove_cvref_t<decltype(::cuda::experimental::__detail::__resource_from_env(
  ::cuda::std::declval<const _Env&>(), ::cuda::std::declval<::cuda::device_ref>()))>;

template <typename _Env>
[[nodiscard]] _CCCL_HOST_API constexpr decltype(auto) __sanitize_buffer_env(const _Env& __env)
{
  if constexpr (::cuda::__buffer_compatible_env<_Env>)
  {
    return __env;
  }
  else
  {
    return ::cuda::std::execution::env<>{};
  }
}

template <class _Tp, class _Resource, class _Env>
[[nodiscard]] _CCCL_HOST_API constexpr auto __make_safe_uninitialized_buffer(
  ::cuda::stream_ref __stream, _Resource __resource, ::cuda::std::size_t __size, const _Env& __env)
{
  if constexpr (::cuda::std::is_trivially_constructible_v<_Tp>)
  {
    return ::cuda::make_buffer<_Tp>(
      __stream, ::cuda::std::move(__resource), __size, ::cuda::no_init, __sanitize_buffer_env(__env));
  }
  else
  {
    return ::cuda::make_buffer<_Tp>(
      __stream, ::cuda::std::move(__resource), __size, _Tp{}, __sanitize_buffer_env(__env));
  }
}

template <class _InputRangeOfRanges, class _RangeOfOutputIt, class _EnvRange>
struct __in_range_out_it_properties
{
  using __input_type  = ::cuda::std::ranges::range_value_t<::cuda::std::ranges::range_reference_t<_InputRangeOfRanges>>;
  using __output_type = ::cuda::std::iter_value_t<::cuda::std::ranges::range_reference_t<_RangeOfOutputIt>>;

  using __env_type = ::cuda::std::ranges::range_value_t<_EnvRange>;

  using __resource_type = ::cuda::experimental::__detail::__resource_type_for<__env_type>;

  using __buffer_type = ::cuda::__buffer_type_for_props<__output_type, typename __resource_type::default_queries>;
};

template <class _RangeOfRanges>
_CCCL_CONCEPT __range_of_sized_random_access_ranges = _CCCL_REQUIRES_EXPR((_RangeOfRanges), )(
  requires(::cuda::std::ranges::forward_range<_RangeOfRanges>),
  requires(::cuda::std::ranges::sized_range<_RangeOfRanges>),
  requires(::cuda::std::ranges::random_access_range<::cuda::std::ranges::range_reference_t<_RangeOfRanges>>));

template <class _RangeOfIters, class _Tp>
_CCCL_CONCEPT __range_of_output_iters = _CCCL_REQUIRES_EXPR((_RangeOfIters, _Tp), )(
  requires(::cuda::std::ranges::forward_range<_RangeOfIters>),
  requires(
    ::cuda::std::output_iterator<::cuda::std::remove_cvref_t<::cuda::std::ranges::range_reference_t<_RangeOfIters>>,
                                 _Tp>));
} // namespace cuda::experimental::__detail

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_COMMON_H
