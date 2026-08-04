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

#ifndef _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_SORT_H
#define _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_SORT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__nvtx/nvtx.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/span>

#include <cuda/experimental/__multi_gpu/algorithm/common.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/execute.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/sorter.h>
#include <cuda/experimental/__multi_gpu/concepts.h>
#include <cuda/experimental/__utility/result_policy.cuh>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental
{
_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

//! @brief Sort input ranges distributed over a communicator in place.
//!
//! Treats the input ranges from all communicator ranks as one logical sequence and sorts it
//! with respect to `__cmp`. Each rank's input range is overwritten with the slice of the
//! globally sorted sequence belonging to that rank, with the slices ordered by ascending
//! communicator rank. The number of elements each rank holds is unchanged, so a rank's input
//! range keeps its original size even though the values it receives may have originated on
//! other ranks. Sorting is not stable, and elements that compare equivalent may end up on any
//! rank that the equivalent run spans.
//!
//! The communicators, environments, and input ranges are iterated in lockstep. Each tuple
//! describes one local communicator rank. This overload is intended for a thread or process
//! that owns multiple local GPUs. For example, if each process owns two GPUs, each process can
//! pass both local ranks in one call, as shown in the test below.
//!
//! @snippet sort/range_basic.cu sort
//!
//! All three outer ranges must have the same length. The algorithm caps lockstep iteration at
//! the shortest range, but this must not be relied upon and may change at any time. Each input
//! range must refer to writable device-accessible storage, and its iterators must be
//! contiguous.
//!
//! Every communicator rank must participate in the collective call, including ranks whose input
//! range is empty. `__cmp` must describe the same strict weak ordering on every rank.
//!
//! Each environment supplies the *required* stream and optional memory resource for its local
//! rank, and is also forwarded to the underlying CUB algorithms, so it may carry any
//! parameters CUB recognizes.
//!
//! @tparam _Policy The result policy. Currently only `distributed_t` is supported.
//! @tparam _CommRange The range of communicators. Each element must model the communicator
//!         concept.
//! @tparam _EnvRange The range of execution environments.
//! @tparam _InputRange The range whose elements are the per-communicator input ranges. Each
//!         element must be a sized random-access range.
//! @tparam _BinaryOp The comparator type. Defaults to `::cuda::std::less<>`.
//!
//! @param[in] __policy The result policy object. Only `cudax::distributed` is currently supported.
//! @param[in] __comms The range of communicators.
//! @param[in] __envs The range of execution environments. Each environment must contain a
//!                   stream.
//! @param[in,out] __range_of_input_ranges The range of per-communicator key ranges, sorted in
//!                place.
//! @param[in] __cmp The comparator defining the sorted order.
_CCCL_TEMPLATE(
  class _Policy, class _CommRange, class _EnvRange, class _InputRange, class _BinaryOp = ::cuda::std::less<>)
_CCCL_REQUIRES(__range_of_communicators<_CommRange> _CCCL_AND ::cuda::std::ranges::forward_range<_EnvRange>
                 _CCCL_AND ::cuda::experimental::__detail::__range_of_sized_random_access_ranges<_InputRange>)
void sort(const __result_policy_base<_Policy>& __policy,
          _CommRange&& __comms,
          _EnvRange&& __envs,
          _InputRange&& __range_of_input_ranges,
          _BinaryOp __cmp = {})
{
  using _Env = ::cuda::std::ranges::range_value_t<_EnvRange>;

  // Could use ::cuda::std::invocable here, but it is overkill (compile-time wise). We know
  // that get_stream_t is a normal CPO and normally callable.
  static_assert(::cuda::std::__is_callable_v<::cuda::get_stream_t, _Env>, "Environment must contain a stream");

  if (::cuda::std::ranges::size(__comms) == 0)
  {
    // We have no inputs, so... nothing to do
    return;
  }

  _CCCL_NVTX_RANGE_SCOPE("cuda::experimental::sort");

  using _Tp =
    ::cuda::std::ranges::range_value_t<::cuda::std::remove_cvref_t<::cuda::std::ranges::range_reference_t<_InputRange>>>;

  ::cuda::experimental::__detail::__hss_sort::_HSSSorter<_Tp, _Env, ::cuda::std::remove_cvref_t<_BinaryOp>>::__execute(
    __policy,
    ::cuda::std::forward<_CommRange>(__comms),
    ::cuda::std::forward<_EnvRange>(__envs),
    ::cuda::std::forward<_InputRange>(__range_of_input_ranges),
    ::cuda::std::move(__cmp));
}

//! @brief Sort a single input range over one communicator in place.
//!
//! Treats the input ranges from all communicator ranks as one logical sequence and sorts it
//! with respect to `__cmp`. `__input` is overwritten with the slice of the globally sorted
//! sequence belonging to `__comm`, with the slices ordered by ascending communicator rank.
//! `__input` keeps its original size, even though the values it receives may have originated on
//! other ranks. Sorting is not stable.
//!
//! This convenience overload forwards one communicator, environment, and input range to the
//! range-based overload. It is intended for a thread or process that owns one local GPU. See
//! the range overload for a description of the algorithm.
//!
//! @snippet sort/single_comm_basic.cu sort_single_range
//!
//! Every communicator rank must participate in the collective call, including ranks whose input
//! range is empty, and `__cmp` must describe the same strict weak ordering on every rank.
//! Because each call takes part in a collective on a single communicator, a caller that owns
//! several local ranks must issue those calls concurrently, for example one thread per rank;
//! issuing them serially on one thread deadlocks. Prefer the range overload in that case.
//!
//! `__input` must refer to writable device-accessible storage and its iterators must be
//! contiguous, since the range is handed to the communicator collectives directly.
//!
//! The environment supplies the stream and optional memory resource for the local rank, and is
//! also forwarded to the underlying CUB algorithms.
//!
//! @tparam _Policy The result policy. Currently only `distributed_t` is supported.
//! @tparam _Comm The communicator type. Must model the communicator concept.
//! @tparam _Env The execution environment type. Supplies the stream and optional memory
//!              resource.
//! @tparam _InputRange The input range type. Must be a sized random-access range.
//! @tparam _BinaryOp The comparator type. Defaults to `::cuda::std::less<>`.
//!
//! @param[in] __policy The result policy object. Only `cudax::distributed` is currently supported.
//! @param[in] __comm The communicator.
//! @param[in] __env The execution environment. Must contain a stream.
//! @param[in,out] __input The local key range, sorted in place.
//! @param[in] __cmp The comparator defining the sorted order.
_CCCL_TEMPLATE(class _Policy, class _Comm, class _Env, class _InputRange, class _BinaryOp = ::cuda::std::less<>)
_CCCL_REQUIRES(__communicator<_Comm> _CCCL_AND ::cuda::std::ranges::random_access_range<_InputRange>)
void sort(const __result_policy_base<_Policy>& __policy,
          _Comm&& __comm,
          _Env&& __env,
          _InputRange&& __input,
          _BinaryOp __cmp = {})
{
  ::cuda::experimental::sort(
    __policy,
    ::cuda::std::span<::cuda::std::remove_reference_t<_Comm>, 1>{::cuda::std::addressof(__comm), 1},
    ::cuda::std::span<::cuda::std::remove_reference_t<_Env>, 1>{::cuda::std::addressof(__env), 1},
    ::cuda::std::span<::cuda::std::remove_reference_t<_InputRange>, 1>{::cuda::std::addressof(__input), 1},
    ::cuda::std::move(__cmp));
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT
} // namespace cuda::experimental

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_SORT_H
