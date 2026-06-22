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

#ifndef _CUDA_EXPERIMENTAL___MULTI_GPU_SCAN_H
#define _CUDA_EXPERIMENTAL___MULTI_GPU_SCAN_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/dispatch/kernels/kernel_reduce.cuh>

#include <thrust/system/cuda/detail/dispatch.h>
#include <thrust/system/cuda/detail/util.h>

#include <cuda/__container/buffer.h>
#include <cuda/__device/device_ref.h>
#include <cuda/__functional/lazy_call_or.h>
#include <cuda/__memory_pool/device_memory_pool.h>
#include <cuda/__memory_resource/get_memory_resource.h>
#include <cuda/__runtime/ensure_current_context.h>
#include <cuda/__stream/get_stream.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/__utility/no_init.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__iterator/readable_traits.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__numeric/accumulate.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/repeat_view.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__ranges/zip_view.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/span>

#include <cuda/experimental/__device/logical_device.cuh>
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
void exclusive_scan(
  _CommRange&& __comms,
  _EnvRange&& __envs,
  _InputRange&& __range_of_input_ranges,
  _OutputItRange&& __outputs,
  _Tp __init,
  _Tp __identity,
  _BinaryOp __op = {})
{
  __validate_input_range<_InputRange>();

  using __result_type =
    ::cuda::std::iter_value_t<::cuda::std::remove_cvref_t<::cuda::std::ranges::range_reference_t<_OutputItRange>>>;

  const auto __partials = __detail::__partial_reduction<__result_type>(
    __comms, ::cuda::std::forward<_EnvRange>(__envs), __range_of_input_ranges, __identity, __op);

  using __buffer_type = typename ::cuda::std::remove_cvref_t<decltype(__partials)>::value_type::__buffer_type;

  ::std::vector<__buffer_type> __prefixes;

  __prefixes.reserve(__partials.size());
  for (auto&& [__comm, __part] : ::cuda::std::ranges::views::zip(__comms, __partials))
  {
    auto&& [__buffer, __env, _] = __part;
    // Compute only the reduction of previous rank's reductions. We later use this as the
    // initializer value in the ExclusiveScan call. Note that for rank 0, this loop body
    // effectively becomes
    //
    // 1. buffer = __prefixes.emplace_back(1);
    // 2. memcpy(buffer.begin(), init);
    const auto __num_items = __comm.rank();

    auto& __prefix_tmp =
      __prefixes.emplace_back(__buffer.stream(), __buffer.memory_resource(), /*__size=*/1, ::cuda::no_init, __env);

    // Note the use of __init here. We need to apply the initializer exactly here so that it's
    // applied once per device over the local reductions.
    __CUDAX_MULTI_GPU_DISPATCH(
      __comm.device(),
      __num_items,
      ::cub::DeviceReduce::Reduce,
      (__buffer.begin(), __prefix_tmp.begin(), __num_items_fixed, __op, __init, __env));
  }

  for (auto&& [__comm, __input, __part, __prefix, __out] :
       ::cuda::std::ranges::views::zip(__comms, __range_of_input_ranges, __partials, __prefixes, __outputs))
  {
    const auto& [_0, __env, _1] = __part;
    const auto __num_items      = ::cuda::std::ranges::size(__input);

    using __future_type = ::cub::FutureValue<typename __buffer_type::value_type, typename __buffer_type::iterator>;

    __CUDAX_MULTI_GPU_DISPATCH(
      __comm.device(),
      __num_items,
      ::cub::DeviceScan::ExclusiveScan,
      (::cuda::std::ranges::begin(__input),
       __out,
       __op,
       /*__init=*/__future_type{__prefix.begin()},
       __num_items_fixed,
       __env));
  }
}

_CCCL_TEMPLATE(
  class _CommRange,
  class _EnvRange,
  class _InputRange,
  class _OutputItRange,
  class _Tp =
    ::cuda::std::iter_value_t<::cuda::std::remove_cvref_t<::cuda::std::ranges::range_reference_t<_OutputItRange>>>)
_CCCL_REQUIRES(__range_of_communicators<_CommRange> _CCCL_AND //
               ::cuda::std::ranges::input_range<_EnvRange> _CCCL_AND //
                 __range_of_sized_ra_ranges<_InputRange> _CCCL_AND //
                   __range_of_output_iters<_OutputItRange, _Tp>)
void exclusive_scan(_CommRange&& __comms,
                    _EnvRange&& __envs,
                    _InputRange&& __range_of_input_ranges,
                    _OutputItRange&& __outputs,
                    _Tp __init = {})
{
  exclusive_scan(
    ::cuda::std::forward<_CommRange>(__comms),
    ::cuda::std::forward<_EnvRange>(__envs),
    ::cuda::std::forward<_InputRange>(__range_of_input_ranges),
    ::cuda::std::forward<_OutputItRange>(__outputs),
    ::cuda::std::move(__init),
    _Tp{},
    ::cuda::std::plus<>{});
}

_CCCL_TEMPLATE(
  class _CommRange,
  class _InputRange,
  class _OutputItRange,
  class _Tp =
    ::cuda::std::iter_value_t<::cuda::std::remove_cvref_t<::cuda::std::ranges::range_reference_t<_OutputItRange>>>,
  class _BinaryOp = ::cuda::std::plus<>)
_CCCL_REQUIRES(__range_of_communicators<_CommRange> _CCCL_AND //
                 __range_of_sized_ra_ranges<_InputRange> _CCCL_AND //
                   __range_of_output_iters<_OutputItRange, _Tp>)
void exclusive_scan(
  _CommRange&& __comms,
  _InputRange&& __range_of_input_ranges,
  _OutputItRange&& __outputs,
  _Tp __init,
  _Tp __identity,
  _BinaryOp __op = {})
{
  exclusive_scan(
    ::cuda::std::forward<_CommRange>(__comms),
    ::cuda::std::ranges::views::repeat(::cuda::std::execution::env<>{}),
    ::cuda::std::forward<_InputRange>(__range_of_input_ranges),
    ::cuda::std::forward<_OutputItRange>(__outputs),
    ::cuda::std::move(__init),
    ::cuda::std::move(__identity),
    ::cuda::std::move(__op));
}

_CCCL_TEMPLATE(class _InputRange,
               class _OutputIt,
               class _Tp       = ::cuda::std::iter_value_t<::cuda::std::remove_cvref_t<_OutputIt>>,
               class _BinaryOp = ::cuda::std::plus<>)
_CCCL_REQUIRES(::cuda::std::ranges::random_access_range<_InputRange> _CCCL_AND //
               ::cuda::std::output_iterator<_OutputIt, _Tp>)
void exclusive_scan(
  const communicator& __comm,
  _InputRange&& __input_range,
  _OutputIt __output,
  _Tp __init,
  _Tp __identity,
  _BinaryOp __op = {})
{
  exclusive_scan(
    ::cuda::std::span<const communicator, 1>{&__comm, /*__count=*/1},
    ::cuda::std::execution::env<>{},
    ::cuda::std::span<const _InputRange, 1>{::cuda::std::addressof(__input_range), /*__count=*/1},
    ::cuda::std::span<const _OutputIt, 1>{::cuda::std::addressof(__output), /*__count=*/1},
    ::cuda::std::move(__init),
    ::cuda::std::move(__identity),
    ::cuda::std::move(__op));
}
} // namespace cuda::experimental

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_SCAN_H
