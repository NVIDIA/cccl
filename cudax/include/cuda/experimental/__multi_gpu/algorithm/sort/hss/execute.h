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

#ifndef _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_EXECUTE_H
#define _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_EXECUTE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/device_merge_sort.cuh>

#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__ranges/zip_view.h>
#include <cuda/std/__type_traits/is_callable.h>

#include <cuda/experimental/__multi_gpu/algorithm/common.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/data_exchange.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/histogramming.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/local_setup.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/rebalance.h>
#include <cuda/experimental/__utility/result_policy.cuh>

#include <vector>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental::__detail::__hss_sort
{
template <class _Traits, class _Policy, class _CommRange, class _EnvRange, class _InputRange, class _BinaryOp>
_CCCL_HOST_API void __execute(
  const __result_policy_base<_Policy>&,
  _CommRange&& __comms,
  _EnvRange&& __envs,
  _InputRange&& __local_inputs,
  _BinaryOp __cmp)
{
  static_assert(::cuda::std::ranges::sized_range<_CommRange>);

  // Could use ::cuda::std::invocable here, but it is overkill (compile-time wise). We know
  // that get_stream_t is a normal CPO and normally callable.
  static_assert(::cuda::std::__is_callable_v<::cuda::get_stream_t, ::cuda::std::ranges::range_value_t<_EnvRange>>,
                "Environment must contain a stream");

  static_assert(::cuda::std::same_as<_Policy, distributed_t>,
                "Only distributed results are currently supported. Please open an issue at "
                "github.com/NVIDIA/cccl/issue requesting support for your specified policy.");

  if (::cuda::std::ranges::size(__comms) == 0)
  {
    // We have no inputs, so... nothing to do
    return;
  }

  // First and foremost, kick off the local sorts
  for (auto&& [__comm, __env, __input] : ::cuda::std::ranges::views::zip(__comms, __envs, __local_inputs))
  {
    __CUDAX_MULTI_GPU_DISPATCH(
      __comm.logical_device(),
      CUB_NS_QUALIFIER::DeviceMergeSort::SortKeys,
      ::cuda::std::ranges::begin(__input),
      ::cuda::std::ranges::size(__input),
      __cmp,
      __env);
  }

  const auto __comm_size = ::cuda::std::ranges::begin(__comms)->size();

  if (__comm_size == 1)
  {
    // Single communicator, nothing to do we have already sorted
    return;
  }

  const auto __setup =
    ::cuda::experimental::__detail::__hss_sort::__local_setup<_Traits>(__comms, __envs, __local_inputs, __comm_size);

  if (__setup.__N == 0)
  {
    return;
  }

  {
    const ::std::vector<typename _Traits::__per_comm_splitters_type> __local_splitters =
      ::cuda::experimental::__detail::__hss_sort::__histogramming_phase<_Traits>(
        __setup, __comms, __envs, __local_inputs, __cmp);

    ::cuda::experimental::__detail::__hss_sort::__data_exchange<_Traits>(
      __setup, __comms, __envs, __local_inputs, __cmp, __local_splitters);
  }

  ::cuda::experimental::__detail::__hss_sort::__rebalance_to_original_counts<_Traits>(
    __setup, __comms, __envs, __local_inputs);
}
} // namespace cuda::experimental::__detail::__hss_sort

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_EXECUTE_H
