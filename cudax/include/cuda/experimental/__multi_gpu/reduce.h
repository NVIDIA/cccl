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

#ifndef _CUDA_EXPERIMENTAL___MULTI_GPU_REDUCE_H
#define _CUDA_EXPERIMENTAL___MULTI_GPU_REDUCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/device_reduce.cuh>
#include <cub/device/dispatch/kernels/kernel_reduce.cuh>

#include <cuda/std/__execution/env.h>
#include <cuda/std/__iterator/readable_traits.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/zip_view.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/span>

#include <cuda/experimental/__multi_gpu/algorithm_utils.h>
#include <cuda/experimental/__multi_gpu/communicator.h>
#include <cuda/experimental/__multi_gpu/concepts.h>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental
{
_CCCL_TEMPLATE(
  class _CommRange,
  class _EnvRange,
  class _InputRange,
  class _OutputItRange,
  class _Tp =
    ::cuda::std::iter_value_t<::cuda::std::remove_cvref_t<::cuda::std::ranges::range_reference_t<_OutputItRange>>>,
  class _BinaryOp = ::cuda::std::plus<>)
_CCCL_REQUIRES(__range_of_communicators<_CommRange> _CCCL_AND //
               ::cuda::std::ranges::input_range<_EnvRange> _CCCL_AND //
                 __range_of_sized_ra_ranges<_InputRange> _CCCL_AND //
                   __range_of_output_iters<_OutputItRange, _Tp>)
void reduce(_CommRange&& __comms,
            _EnvRange&& __envs,
            _InputRange&& __range_of_input_ranges,
            _OutputItRange&& __outputs,
            _Tp __init     = {},
            _BinaryOp __op = {})
{
  __validate_input_range<_InputRange>();

  using __input_type = ::cuda::std::ranges::range_value_t<::cuda::std::ranges::range_reference_t<_InputRange>>;

  using __result_type =
    ::cuda::std::iter_value_t<::cuda::std::remove_cvref_t<::cuda::std::ranges::range_reference_t<_OutputItRange>>>;

#if 0
  if constexpr (__nccl::__can_specialize_nccl_call<__input_type, __result_type, _BinaryOp>)
  {
    // NCCL has no concept of initializers, so we can only do this if the initializer is identity
    if (__init == ::cuda::identity_element<_BinaryOp, _Tp>())
    {
      for (const auto& [__comm, __env, __input, __output] :
           ::cuda::std::ranges::views::zip(__comms, __envs, __range_of_input_ranges, __outputs))
      {
        const __input_type* const __input_ptr = &*::cuda::std::ranges::begin(__input);
        __result_type* const __output_ptr     = &*__output;
        const ::cuda::stream_ref __stream     = ::cuda::__lazy_call_or(
          ::cuda::get_stream,
          [] {
            return ::cuda::stream_ref{::CUstream{}};
          },
          __env);

        __nccl::__ncclAllReduce(
          __input_ptr,
          __output_ptr,
          ::cuda::std::ranges::size(__input),
          __nccl::__nccl_type_of_v<__input_type>,
          __nccl::__nccl_redop_of_v<_BinaryOp>,
          __comm.comm(),
          __stream);
      }
      return;
    }
  }
#endif

  const auto __partials = __detail::__partial_reduction<__result_type>(
    __comms,
    ::cuda::std::forward<_EnvRange>(__envs),
    ::cuda::std::forward<_InputRange>(__range_of_input_ranges),
    __init,
    __op);

  // TODO(jfaibussowit): Implement specialized reduction path where we call ncclAllReduce()
  // directly on the partials (or directly on the inputs). Calling on the partials requires:
  //
  // 1. The op maps directly to a nccl op.
  // 2. The value type maps directly to a nccl value type.
  //
  // Calling directly on the inputs further requires:
  //
  // 1. All input ranges are contiguous ranges.
  // 2. The initializer is exactly the identity value for the chosen op. So 0 for sum or prod
  //    and MIN/MAX for max/min respectively.
  for (auto&& [__comm, __part, __out] : ::cuda::std::ranges::views::zip(__comms, __partials, __outputs))
  {
    auto&& [__buffer, __env, _] = __part;
    const auto __num_items      = __buffer.size();

    __CUDAX_MULTI_GPU_DISPATCH(
      __comm.device(),
      __num_items,
      ::cub::DeviceReduce::Reduce,
      (__buffer.begin(),
       __out,
       __num_items_fixed,
       __op,
       ::cub::detail::reduce::empty_problem_init_t<_Tp>{__init},
       __env));
  }
}

_CCCL_TEMPLATE(class _Env,
               class _InputRange,
               class _OutputIt,
               class _Tp       = ::cuda::std::ranges::range_value_t<_InputRange>,
               class _BinaryOp = ::cuda::std::plus<>)
_CCCL_REQUIRES(::cuda::std::ranges::input_range<_InputRange>)
void reduce(const communicator& __comm,
            const _Env& __env,
            _InputRange&& __input_range,
            _OutputIt __output,
            _Tp __init     = {},
            _BinaryOp __op = {})
{
  reduce(::cuda::std::span<const communicator, 1>{&__comm, /*__count=*/1},
         ::cuda::std::span<const _Env, 1>{::cuda::std::addressof(__env), /*__count=*/1},
         ::cuda::std::span<const _InputRange, 1>{::cuda::std::addressof(__input_range), /*__count=*/1},
         ::cuda::std::span<const _OutputIt, 1>{::cuda::std::addressof(__output), /*__count=*/1},
         ::cuda::std::move(__init),
         ::cuda::std::move(__op));
}

_CCCL_TEMPLATE(class _InputRange,
               class _OutputIt,
               class _Tp       = ::cuda::std::ranges::range_value_t<_InputRange>,
               class _BinaryOp = ::cuda::std::plus<>)
_CCCL_REQUIRES(::cuda::std::ranges::input_range<_InputRange>)
void reduce(
  const communicator& __comm, _InputRange&& __input_range, _OutputIt __output, _Tp __init = {}, _BinaryOp __op = {})
{
  reduce(__comm,
         ::cuda::std::execution::env<>{},
         ::cuda::std::forward<_InputRange>(__input_range),
         ::cuda::std::move(__output),
         ::cuda::std::move(__init),
         ::cuda::std::move(__op));
}
} // namespace cuda::experimental

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_REDUCE_H
