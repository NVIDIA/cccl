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
#include <cub/util_type.cuh>

#include <cuda/__functional/operator_properties.h>
#include <cuda/__nvtx/nvtx.h>
#include <cuda/__stream/get_stream.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/readable_traits.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__ranges/zip_view.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/span>

#include <cuda/experimental/__multi_gpu/algorithm/common.h>
#include <cuda/experimental/__multi_gpu/algorithm/reduce/reduce.h>
#include <cuda/experimental/__multi_gpu/concepts.h>

#include <vector>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental
{
namespace __detail::__scan
{
enum class __kind : ::cuda::std::uint8_t
{
  __exclusive,
  __inclusive
};

template <__kind _Kind,
          class _CommRange,
          class _EnvRange,
          class _InputRangeRange,
          class _OutputItRange,
          class _Tp,
          class _BinaryOp>
_CCCL_HOST_API void __scan(
  _CommRange&& __comms,
  _EnvRange&& __envs,
  _InputRangeRange&& __range_of_inputs,
  _OutputItRange&& __outputs,
  _Tp __init,
  _BinaryOp __op,
  _Tp __ident)
{
  static_assert(::cuda::std::ranges::sized_range<_CommRange>);

  using __properties =
    ::cuda::experimental::__detail::__in_range_out_it_properties<_InputRangeRange, _OutputItRange, _EnvRange>;

  static_assert(::cuda::experimental::__indirectly_binary_reducible<
                _BinaryOp,
                _Tp,
                ::cuda::std::ranges::iterator_t<::cuda::std::ranges::range_reference_t<_InputRangeRange>>>);

  // Could use ::cuda::std::invocable here, but it is overkill (compile-time wise). We know
  // that get_stream_t is a normal CPO and normally callable.
  static_assert(::cuda::std::__is_callable_v<::cuda::get_stream_t, typename __properties::__env_type>,
                "Environment must contain a stream");

  const auto __num_local = ::cuda::std::ranges::size(__comms);

  if (!__num_local)
  {
    return;
  }

  _CCCL_NVTX_RANGE_SCOPE(
    _Kind == __kind::__exclusive ? "cuda::experimental::exclusive_scan" : "cuda::experimental::inclusive_scan");

  using __properties =
    ::cuda::experimental::__detail::__in_range_out_it_properties<_InputRangeRange, _OutputItRange, _EnvRange>;

  constexpr auto __ROOT_RANK = 0;
  auto __partials            = ::std::vector<typename __properties::__buffer_type>{};

  __partials.reserve(__num_local);
  // TODO(jfaibussowit): can just be ranges::zip | ranges::transform | ranges::to() (and then
  // we don't need to do the env, and buffer type deduction upfront)
  for (auto&& [__comm, __env, __inputs] : ::cuda::std::ranges::views::zip(__comms, __envs, __range_of_inputs))
  {
    __partials.emplace_back(
      ::cuda::experimental::__detail::__reduce::__local_reduction<typename __properties::__buffer_type>(
        __ROOT_RANK, __comm, __env, __inputs, __init, __op, __ident));
  }

  {
    auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();

    for (auto&& [__comm, __part] : ::cuda::std::ranges::views::zip(__comms, __partials))
    {
      auto* const __ptr = __part.data();

      // in-place, sendbuff == recvbuff + rank
      __comm.all_gather(
        __guard,
        __ptr + __comm.rank(),
        __ptr,
        /*__sendcount*/ 1,
        __part.stream());
    }
  }

  // At this point (assuming init of 0 and plus) __partials contains a copy of everyones local
  // reductions, so given:
  //
  // rank 0: [1, 2, 3]
  // rank 1: [4, 5, 6]
  // rank 2: [7, 8, 9]
  //
  // then __partials should be
  //
  // rank 0: [6, 15, 24]
  // rank 1: [6, 15, 24]
  // rank 2: [6, 15, 24]
  //
  // we now want to compute the prefix sum up to our current rank, by summing the local
  // reductions progressively in __prefix
  //
  // rank 0: [0]
  // rank 1: [6]
  // rank 2: [21]
  //
  // Then later we can do the proper prefix scan using __prefix as the initializer for it.
  for (auto&& [__comm, __env, __part, __inputs, __out] :
       ::cuda::std::ranges::views::zip(__comms, __envs, __partials, __range_of_inputs, __outputs))
  {
    auto __prefix = ::cuda::experimental::__detail::__make_safe_uninitialized_buffer<_Tp>(
      __part.stream(), __part.memory_resource(), /*__size=*/1, __env);

    {
      const auto __num_items = __comm.rank();
      // Root rank has no preceding partials and therefore starts directly with init. Other ranks
      // include root rank partial, which already contains init, so their prefix reduction starts
      // with ident.
      const auto& __prefix_init = __comm.rank() == __ROOT_RANK ? __init : __ident;

      // TODO(jfaibussowit):
      //
      // We could potentially just fold this into the scan kernel directly with fancy iterators.
      // Basically, every rank needs to compute its local prefix sum on the init dereference. If
      // that only happens once, we can probably reasonably fuse these kernels. If it happens
      // multiple times, we need to cache the result.
      __CUDAX_MULTI_GPU_DISPATCH(
        __comm.logical_device(),
        __num_items,
        CUB_NS_QUALIFIER::DeviceReduce::Reduce,
        (__part.begin(), __prefix.begin(), __num_items_fixed, __op, __prefix_init, __env));
    }

    if constexpr (_Kind == __kind::__exclusive)
    {
      const auto __num_items = ::cuda::std::ranges::size(__inputs);

      // Don't rely on CTAD for FutureValue, we want to be absolutely sure the iterator does not
      // convert to something we dont expect
      using __future_type = CUB_NS_QUALIFIER::FutureValue<typename __properties::__buffer_type::value_type,
                                                          typename __properties::__buffer_type::iterator>;

      __CUDAX_MULTI_GPU_DISPATCH(
        __comm.logical_device(),
        __num_items,
        CUB_NS_QUALIFIER::DeviceScan::ExclusiveScan,
        (::cuda::std::ranges::begin(__inputs),
         __out,
         __op,
         /*__init=*/__future_type{__prefix.begin()},
         __num_items_fixed,
         __env));
    }
    else
    {
      static_assert(_Kind == __kind::__exclusive, "CUB does not support inclusive scans with a future initial value");
    }
  }
}
} // namespace __detail::__scan

//! @brief Compute an exclusive scan across input ranges distributed over a communicator.
//!
//! Treats the input ranges from all communicator ranks as one logical sequence, with the
//! ranges concatenated in ascending communicator rank order. For each local rank, writes the
//! portion of the exclusive scan corresponding to that rank's input range. The first output
//! element in the logical sequence is `__init`. Every other output element is the reduction of
//! `__init` with all preceding elements in the logical sequence. Empty input ranges contribute
//! no elements.
//!
//! The communicators, environments, input ranges, and output iterators are iterated in
//! lockstep. Each tuple describes one local communicator rank. This overload is intended for
//! a thread or process that owns multiple local GPUs. For example, if each process owns two
//! GPUs, each process can pass both local ranks in one call, as shown in the test below.
//!
//! @snippet exclusive_scan/range_basic.cu exclusive_scan
//!
//! All four outer ranges must have the same length. The algorithm caps lockstep iteration at
//! the shortest range, but this must not be relied upon and may change at any time. Each
//! output iterator must refer to writable device-accessible storage for at least as many
//! values as the corresponding input range contains.
//!
//! Every communicator rank must participate in the collective call. `__init`, `__op`, and
//! `__ident` must describe the same operation on every rank. `__op` must be associative, and
//! `__ident` must be an identity element for `__op`. For example, the identity element for
//! addition is zero.
//!
//! Each environment supplies the stream and optional memory resource for its local rank. The
//! results are ready after the work enqueued on the corresponding streams completes.
//!
//! @tparam _CommRange The range of communicators. Each element must model the communicator
//!         concept.
//! @tparam _EnvRange The range of execution environments.
//! @tparam _InputRangeRange The range whose elements are the per-communicator input ranges.
//!         Each element must be a sized random-access range.
//! @tparam _OutputItRange The range of output iterators, one per communicator.
//! @tparam _Tp The scan value type. Deduced by default from the input element type.
//! @tparam _BinaryOp The binary scan operator type. Defaults to `::cuda::std::plus<>`.
//!
//! @param[in] __comms The range of communicators.
//! @param[in] __envs The range of execution environments. Each environment must contain a
//!                   stream.
//! @param[in] __range_of_inputs The range of per-communicator input ranges to scan.
//! @param[out] __outputs The range of output iterators receiving the per-communicator results.
//! @param[in] __init The initial value for the scan.
//! @param[in] __op The associative binary scan operator.
//! @param[in] __ident The identity element for `__op`.
_CCCL_TEMPLATE(class _CommRange,
               class _EnvRange,
               class _InputRangeRange,
               class _OutputItRange,
               class _Tp = ::cuda::std::ranges::range_value_t<::cuda::std::ranges::range_reference_t<_InputRangeRange>>,
               class _BinaryOp = ::cuda::std::plus<>)
_CCCL_REQUIRES(__range_of_communicators<_CommRange> _CCCL_AND ::cuda::std::ranges::forward_range<_EnvRange> _CCCL_AND
                 __detail::__range_of_sized_random_access_ranges<_InputRangeRange> _CCCL_AND
                   __detail::__range_of_output_iters<_OutputItRange, _Tp>)
_CCCL_HOST_API void exclusive_scan(
  _CommRange&& __comms,
  _EnvRange&& __envs,
  _InputRangeRange&& __range_of_inputs,
  _OutputItRange&& __outputs,
  _Tp __init     = {},
  _BinaryOp __op = {},
  _Tp __ident    = ::cuda::identity_element<_BinaryOp, _Tp>())
{
  ::cuda::experimental::__detail::__scan::__scan<::cuda::experimental::__detail::__scan::__kind::__exclusive>(
    ::cuda::std::forward<_CommRange>(__comms),
    ::cuda::std::forward<_EnvRange>(__envs),
    ::cuda::std::forward<_InputRangeRange>(__range_of_inputs),
    ::cuda::std::forward<_OutputItRange>(__outputs),
    ::cuda::std::move(__init),
    ::cuda::std::move(__op),
    ::cuda::std::move(__ident));
}

//! @brief Compute an exclusive scan over a single input range on one communicator.
//!
//! Treats the input ranges from all communicator ranks as one logical sequence, with the
//! ranges concatenated in ascending communicator rank order. Writes the portion of the
//! exclusive scan corresponding to `__comm` to `__output`. The first output element in the
//! logical sequence is `__init`. Every other output element is the reduction of `__init` with
//! all preceding elements in the logical sequence.
//!
//! This convenience overload forwards one communicator, environment, input range, and output
//! iterator to the range-based overload. It is intended for a thread or process that owns one
//! local GPU.
//!
//! @snippet exclusive_scan/single_comm_basic.cu exclusive_scan_single_range
//!
//! Every communicator rank must participate in the collective call. `__init`, `__op`, and
//! `__ident` must describe the same operation on every rank. `__op` must be associative, and
//! `__ident` must be an identity element for `__op`. `__output` must refer to writable
//! device-accessible storage for at least as many values as `__input` contains.
//!
//! The environment supplies the stream and optional memory resource for the local rank. The
//! results are ready after the work enqueued on that stream completes.
//!
//! @tparam _Comm The communicator type. Must model the communicator concept.
//! @tparam _Env The execution environment type. Supplies the stream and optional memory
//!              resource.
//! @tparam _InputRange The input range type. Must be a sized random-access range.
//! @tparam _OutputIt The output iterator type.
//! @tparam _Tp The scan value type. Deduced by default from the input element type.
//! @tparam _BinaryOp The binary scan operator type. Defaults to `::cuda::std::plus<>`.
//!
//! @param[in] __comm The communicator.
//! @param[in] __env The execution environment. Must contain a stream.
//! @param[in] __input The input range to scan.
//! @param[out] __output The output iterator receiving the local scan results.
//! @param[in] __init The initial value for the scan.
//! @param[in] __op The associative binary scan operator.
//! @param[in] __ident The identity element for `__op`.
_CCCL_TEMPLATE(class _Comm,
               class _Env,
               class _InputRange,
               class _OutputIt,
               class _Tp       = ::cuda::std::ranges::range_value_t<_InputRange>,
               class _BinaryOp = ::cuda::std::plus<>)
_CCCL_REQUIRES(__communicator<_Comm> _CCCL_AND ::cuda::std::ranges::random_access_range<_InputRange>
                 _CCCL_AND ::cuda::std::output_iterator<_OutputIt, _Tp>)
_CCCL_HOST_API void exclusive_scan(
  _Comm&& __comm,
  _Env&& __env,
  _InputRange&& __input,
  _OutputIt __output,
  _Tp __init     = {},
  _BinaryOp __op = {},
  _Tp __ident    = ::cuda::identity_element<_BinaryOp, _Tp>())
{
  exclusive_scan(
    ::cuda::std::span<::cuda::std::remove_reference_t<_Comm>, 1>{::cuda::std::addressof(__comm), 1},
    ::cuda::std::span<::cuda::std::remove_reference_t<_Env>, 1>{::cuda::std::addressof(__env), 1},
    ::cuda::std::span<::cuda::std::remove_reference_t<_InputRange>, 1>{::cuda::std::addressof(__input), 1},
    ::cuda::std::span<_OutputIt, 1>{::cuda::std::addressof(__output), 1},
    ::cuda::std::move(__init),
    ::cuda::std::move(__op),
    ::cuda::std::move(__ident));
}
} // namespace cuda::experimental

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_SCAN_H
