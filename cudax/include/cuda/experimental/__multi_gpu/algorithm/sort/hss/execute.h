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

#include <cuda/experimental/__multi_gpu/algorithm/common.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/data_exchange.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/histogramming.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/local_setup.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/rebalance.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/sorter.h>
#include <cuda/experimental/__utility/result_policy.cuh>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental::__detail::__hss_sort
{
//! @brief Drive the complete multi-GPU HSS sort across all communicators.
//!
//! Top-level entry point that sequences the histogram-sort-with-sampling phases. Implements
//! the algorithm defined in "Histogram Sort with Sampling" by Harsh
//! et. al. (arxiv.org/abs/1803.01237, alternatively
//! dl.acm.org/doi/10.1145/3323165.3323184)
//!
//! @tparam _Traits The `__hss_traits` instantiation carrying the value, environment, and
//!                 comparator types.
//!
//! @param[in] __result_policy_base The result-policy tag (must be `distributed_t`); its value is
//!            unused.
//! @param[in] __comms The range of per-rank communicators.
//! @param[in] __envs The range of per-rank execution environments (one stream each).
//! @param[in,out] __local_inputs The range of per-rank local key ranges, sorted in place.
//! @param[in] __cmp The comparator defining the sorted order.
template <class _Tp, class _Env, class _BinaryOp>
template <class _Policy, class _CommRange, class _EnvRange, class _InputRange>
_CCCL_HOST_API void _HSSSorter<_Tp, _Env, _BinaryOp>::__execute(
  const __result_policy_base<_Policy>&,
  _CommRange&& __comms,
  _EnvRange&& __envs,
  _InputRange&& __local_inputs,
  _BinaryOp __cmp)
{
  static_assert(::cuda::std::same_as<_Policy, distributed_t>,
                "Only distributed results are currently supported. Please open an issue at "
                "github.com/NVIDIA/cccl/issue requesting support for your specified policy.");
  static_assert(::cuda::std::ranges::sized_range<_CommRange>);

  // TODO(jfaibussowit):
  //
  // We should consider supporting random-access ranges too, but then we need temp buffers for
  // the various comms calls.
  //
  // We cannot assert the following because thrust iterators don't play nice with it.
#if 0
  using __base_iter =
    ::cuda::std::ranges::iterator_t<::cuda::std::remove_cvref_t<::cuda::std::ranges::range_reference_t<_InputRange>>>;

  static_assert(::cuda::std::__has_contiguous_traversal<__base_iter>);
#endif

  // First and foremost, kick off the local sorts...
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

  // ...so we can exit as early as possible in the single GPU case
  if (__comm_size == 1)
  {
    return;
  }

  const auto __setup = __local_setup(__comms, __envs, __local_inputs, __comm_size);

  if (__setup.__N == 0)
  {
    return;
  }

  {
    const auto __local_splitters = __histogramming_phase(__setup, __comms, __envs, __local_inputs, __cmp);

    __data_exchange(__setup, __comms, __envs, __local_inputs, __cmp, __local_splitters);
  }

  __rebalance_to_original_counts(__setup, __comms, __envs, __local_inputs);
}
} // namespace cuda::experimental::__detail::__hss_sort

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_EXECUTE_H
