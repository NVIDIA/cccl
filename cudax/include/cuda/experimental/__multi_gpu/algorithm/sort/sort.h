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
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/readable_traits.h>
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

//! @brief Sort inputs distributed over a communicator in place.
//!
//! Treats the inputs from all communicator ranks as one logical sequence and sorts it with
//! respect to `__cmp`. Each rank's input is overwritten with the slice of the globally sorted
//! sequence belonging to that rank, with the slices ordered by ascending communicator rank. The
//! number of elements each rank holds is unchanged, so a rank keeps its original size even
//! though the values it receives may have originated on other ranks. Sorting is not stable, and
//! elements that compare equivalent may end up on any rank that the equivalent run spans.
//!
//! The communicators, environments, input iterators, and input sizes are iterated in lockstep.
//! Each tuple describes one local communicator rank. This overload is intended for a thread or
//! process that owns multiple local GPUs. For example, if each process owns two GPUs, each
//! process can pass both local ranks in one call, as shown in the test below.
//!
//! @snippet sort/range_basic.cu sort
//!
//! All four outer ranges must have the same length. The algorithm caps lockstep iteration at
//! the shortest range, but this must not be relied upon and may change at any time. Each input
//! iterator must refer to writable device-accessible storage for at least as many values as the
//! corresponding input size gives, and must be contiguous.
//!
//! Every communicator rank must participate in the collective call, including ranks whose input
//! is empty. `__cmp` must describe the same strict weak ordering on every rank.
//!
//! Each environment supplies the *required* stream and optional memory resource for its local
//! rank, and is also forwarded to the underlying CUB algorithms, so it may carry any
//! parameters CUB recognizes.
//!
//! @param[in] __policy The result policy object. Currently must be `cudax::distributed`.
//! @param[in] __comms The range of communicators.
//! @param[in] __envs The range of execution environments. Each environment must contain a
//!                   stream.
//! @param[in,out] __input_iters The range of per-communicator key iterators, sorted in place.
//! @param[in] __num_items_range A range of sizes per input iterator to sort.
//! @param[in] __cmp The comparator defining the sorted order.
_CCCL_TEMPLATE(class _Policy,
               class _CommRange,
               class _EnvRange,
               class _InputIterRange,
               class _SizeTRange,
               class _BinaryOp = ::cuda::std::less<>)
_CCCL_REQUIRES(__range_of_communicators<_CommRange> _CCCL_AND ::cuda::std::ranges::forward_range<_EnvRange>
                 _CCCL_AND ::cuda::experimental::__detail::__range_of_random_access_iterators<_InputIterRange>
                   _CCCL_AND ::cuda::std::ranges::forward_range<_SizeTRange>)
void sort(const __result_policy_base<_Policy>& __policy,
          _CommRange&& __comms,
          _EnvRange&& __envs,
          _InputIterRange&& __input_iters,
          _SizeTRange&& __num_items_range,
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
    ::cuda::std::iter_value_t<::cuda::std::remove_cvref_t<::cuda::std::ranges::range_reference_t<_InputIterRange>>>;

  ::cuda::experimental::__detail::__hss_sort::_HSSSorter<_Tp, _Env, ::cuda::std::remove_cvref_t<_BinaryOp>>::__execute(
    __policy,
    ::cuda::std::forward<_CommRange>(__comms),
    ::cuda::std::forward<_EnvRange>(__envs),
    ::cuda::std::forward<_InputIterRange>(__input_iters),
    ::cuda::std::forward<_SizeTRange>(__num_items_range),
    ::cuda::std::move(__cmp));
}

//! @brief Sort a single input over one communicator in place.
//!
//! Treats the inputs from all communicator ranks as one logical sequence and sorts it with
//! respect to `__cmp`. `__input_iter` is overwritten with the slice of the globally sorted
//! sequence belonging to `__comm`, with the slices ordered by ascending communicator rank. The
//! local input keeps its original size, even though the values it receives may have originated
//! on other ranks. Sorting is not stable.
//!
//! This convenience overload forwards one communicator, environment, input iterator, and input
//! size to the range-based overload. It is intended for a thread or process that owns one local
//! GPU. See the range overload for a description of the algorithm.
//!
//! @snippet sort/single_comm_basic.cu sort_single_range
//!
//! Every communicator rank must participate in the collective call, including ranks whose input
//! is empty, and `__cmp` must describe the same strict weak ordering on every rank. Because each
//! call takes part in a collective on a single communicator, a caller that owns several local
//! ranks must issue those calls concurrently, for example one thread per rank; issuing them
//! serially on one thread deadlocks. Prefer the range overload in that case.
//!
//! `__input_iter` must refer to writable device-accessible storage for at least `__num_items`
//! values and must be contiguous, since it is handed to the communicator collectives directly.
//!
//! The environment supplies the stream and optional memory resource for the local rank, and is
//! also forwarded to the underlying CUB algorithms.
//!
//! @param[in] __policy The result policy object. Currently must be `cudax::distributed`.
//! @param[in] __comm The communicator.
//! @param[in] __env The execution environment. Must contain a stream.
//! @param[in,out] __input_iter The local key iterator, sorted in place.
//! @param[in] __num_items The number of items in `__input_iter` to sort.
//! @param[in] __cmp The comparator defining the sorted order.
_CCCL_TEMPLATE(
  class _Policy, class _Comm, class _Env, class _InputIt, class _SizeT, class _BinaryOp = ::cuda::std::less<>)
_CCCL_REQUIRES(__communicator<_Comm> _CCCL_AND ::cuda::std::random_access_iterator<_InputIt>)
void sort(const __result_policy_base<_Policy>& __policy,
          _Comm&& __comm,
          _Env&& __env,
          _InputIt __input_iter,
          _SizeT __num_items,
          _BinaryOp __cmp = {})
{
  ::cuda::experimental::sort(
    __policy,
    ::cuda::std::span<::cuda::std::remove_reference_t<_Comm>, 1>{::cuda::std::addressof(__comm), 1},
    ::cuda::std::span<::cuda::std::remove_reference_t<_Env>, 1>{::cuda::std::addressof(__env), 1},
    ::cuda::std::span<_InputIt, 1>{::cuda::std::addressof(__input_iter), 1},
    ::cuda::std::span<_SizeT, 1>{::cuda::std::addressof(__num_items), 1},
    ::cuda::std::move(__cmp));
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT
} // namespace cuda::experimental

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_SORT_H
