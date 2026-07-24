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

#ifndef _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_SAMPLE_PROBES_H
#define _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_SAMPLE_PROBES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__launch/launch.h>
#include <cuda/std/__algorithm/lower_bound.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__algorithm/sample.h>
#include <cuda/std/__cmath/rounding_functions.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__optional/optional.h>
#include <cuda/std/__random/philox_engine.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/cstdint>
#include <cuda/std/span>

#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/traits.h>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental::__detail::__hss_sort
{
//! @brief Single-thread kernel that draws sample keys from the union of splitter intervals.
//!
//! Runs on exactly one grid thread (all others return immediately). It walks the per-splitter
//! sampling intervals `__I_j`, and for each interval `[lo, hi)` locates the corresponding sorted
//! sub-range of the local keys. Implements the per-round sampling of the paper's
//! histogram-sort-with-sampling loop.
//!
//!
//! @param[in] __config The launch configuration used to compute the thread's grid rank.
//! @param[in] __gen The Philox random engine used for sampling.
//! @param[in] __prob The per-key sampling probability.
//! @param[in] __begin Pointer to the first local key; advanced internally across intervals.
//! @param[in] __end Pointer one past the last local key.
//! @param[in] __I_j The span of per-splitter `[lo, hi]` sampling intervals.
//! @param[in] __cmp The comparator defining the sorted order.
//! @param[out] __samples The span the drawn sample keys are written into.
//! @param[out] __samples_size Receives the number of keys actually drawn.
//
// TODO(jfaibussowit):
//
// Parallelize with multiple threads (but not too many!). __I_j is O(p-1), so in
// practice at the absolute max a few thousand if you are running on the worlds
// largest supercomputers.
template <class _Config, class _Tp, class _BinaryOp>
_CCCL_KERNEL_ATTRIBUTES void __sample_probes_kernel(
  _Config __config,
  ::cuda::std::philox4x64 __gen,
  const double __prob,
  const _Tp* __begin,
  const _Tp* const __end,
  const ::cuda::std::span<const ::cuda::std::pair<::cuda::std::optional<_Tp>, ::cuda::std::optional<_Tp>>> __I_j,
  _BinaryOp __cmp,
  const ::cuda::std::span<_Tp> __samples,
  ::cuda::std::size_t* const __samples_size)
{
  if (cuda::gpu_thread.rank(cuda::grid, __config) != 0)
  {
    // Just in case
    return;
  }

  auto __samples_it = __samples.begin();

  // By value so that load from global memory happens only once
  for (const auto [__lo, __hi] : __I_j)
  {
    // Sample from the union of splitter intervals. Splitter intervals are disjoint or
    // identical; lo_it skips an identical interval already covered by an earlier splitter.
    const auto __last  = __hi.has_value() ? ::cuda::std::lower_bound(__begin, __end, *__hi, __cmp) : __end;
    const auto __first = __lo.has_value() ? ::cuda::std::lower_bound(__begin, __last, *__lo, __cmp) : __begin;

    _CCCL_ASSERT(__first <= __last, "Inputs are not sorted for binary search");

    const auto __num_samples       = ::cuda::std::ceil(static_cast<double>(__last - __first) * __prob);
    const auto __remaining_samples = __samples.end() - __samples_it;
    const auto __n                 = ::cuda::std::min(
      static_cast<::cuda::std::uint64_t>(__num_samples), static_cast<::cuda::std::uint64_t>(__remaining_samples));

    __samples_it = ::cuda::std::sample(__first, __last, __samples_it, __n, __gen);
    __begin      = __last;
  }

  *__samples_size = static_cast<::cuda::std::size_t>(__samples_it - __samples.begin());
}

//! @brief Launch the sampling kernel to draw sample keys from a rank's local input.
//!
//! @tparam _Traits The `__hss_traits` instantiation carrying the value and buffer types.
//!
//! @param[in] __input The rank's sorted local key range.
//! @param[in] __I_j The per-splitter `[lo, hi]` sampling intervals.
//! @param[in] __sampling_probability The per-key sampling probability.
//! @param[in] __cmp The comparator defining the sorted order.
//! @param[out] __samples The buffer the drawn sample keys are written into.
//! @param[out] __sample_size The buffer receiving the number of keys drawn.
template <class _Traits, class _Tp, class _BinaryOp, class _InputRange>
_CCCL_HOST_API void __sample_probes(
  _InputRange&& __input,
  const __buffer_of<_Traits, ::cuda::std::pair<::cuda::std::optional<_Tp>, ::cuda::std::optional<_Tp>>>& __I_j,
  double __sampling_probability,
  _BinaryOp __cmp,
  __buffer_of<_Traits, _Tp>* __samples,
  __buffer_of<_Traits, ::cuda::std::size_t>* __sample_size)
{
  constexpr auto __config =
    ::cuda::make_config(::cuda::make_hierarchy(::cuda::block_dims<1>(), ::cuda::grid_dims<1>()));

  _CCCL_VERIFY(__sampling_probability > 0, "Cannot have 0 probably of picking elements");

  ::cuda::launch(
    // All inputs should be on the same stream here
    __I_j.__get().stream(),
    __config,
    __sample_probes_kernel<::cuda::std::remove_cvref_t<decltype(__config)>, _Tp, _BinaryOp>,
    ::cuda::std::philox4x64{static_cast<::cuda::std::uint32_t>(__sampling_probability)},
    __sampling_probability,
    ::cuda::std::to_address(::cuda::std::ranges::begin(__input)),
    ::cuda::std::to_address(::cuda::std::ranges::end(__input)),
    __I_j.__get(),
    ::cuda::std::move(__cmp),
    ::cuda::std::span<_Tp>{*__samples},
    __sample_size->__get().data());
}
} // namespace cuda::experimental::__detail::__hss_sort

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_SAMPLE_PROBES_H
