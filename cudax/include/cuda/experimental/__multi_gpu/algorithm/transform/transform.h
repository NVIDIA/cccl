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

#ifndef _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_TRANSFORM_TRANSFORM_H
#define _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_TRANSFORM_TRANSFORM_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/device_transform.cuh>

#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/zip_view.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/span>

#include <cuda/experimental/__multi_gpu/algorithm/common.h>
#include <cuda/experimental/__multi_gpu/concepts.h>
#include <cuda/experimental/__utility/result_policy.cuh>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental
{
//! @brief Apply a transform operator to inputs distributed over a communicator.
//!
//! Applies `__op` to every element of each rank's input and writes the result to the
//! corresponding position of that rank's output. The operation is rank-local: a rank reads only
//! the elements it owns, and no value moves between ranks. Each output therefore holds exactly as
//! many values as the matching input size gives, in the same order as the input.
//!
//! The communicators, environments, input iterators, input sizes, and output iterators are
//! iterated in lockstep. Each tuple describes one local communicator rank. This overload is
//! intended for a thread or process that owns multiple local GPUs. For example, if each process
//! owns two GPUs, each process can pass both local ranks in one call, as shown in the test below.
//!
//! @snippet transform/range_basic.cu transform
//!
//! All five outer ranges must have the same length. The algorithm caps lockstep iteration at the
//! shortest range, but this must not be relied upon and may change at any time. Each input
//! iterator must refer to readable device-accessible storage for at least as many values as the
//! corresponding input size gives, and each output iterator must refer to writable
//! device-accessible storage for the same number of values.
//!
//! An output iterator may alias its input iterator, which transforms that rank in place. Any
//! other overlap between an input and an output gives undefined behavior.
//!
//! Each environment supplies the *required* stream and optional memory resource for its local
//! rank, and is also forwarded to the underlying CUB algorithms, so it may carry any parameters
//! CUB recognizes.
//!
//! @param[in] __policy The result policy object. Currently must be `cudax::distributed`.
//! @param[in] __comms The range of communicators.
//! @param[in] __envs The range of execution environments. Each environment must contain a
//!                   stream.
//! @param[in] __input_iters The range of per-communicator input iterators.
//! @param[in] __num_items_range A range of sizes per input iterator to transform.
//! @param[out] __output_iters The range of per-communicator output iterators.
//! @param[in] __op The operator applied to every input element.
_CCCL_TEMPLATE(class _Policy,
               class _CommRange,
               class _EnvRange,
               class _InputIterRange,
               class _SizeTRange,
               class _OutputIterRange,
               class _TransformOp)
_CCCL_REQUIRES(__range_of_communicators<_CommRange> _CCCL_AND ::cuda::std::ranges::forward_range<_EnvRange> _CCCL_AND
                 __detail::__range_of_random_access_iterators<_InputIterRange>
                   _CCCL_AND ::cuda::std::ranges::forward_range<_SizeTRange> _CCCL_AND
                     __detail::__range_of_random_access_iterators<_OutputIterRange>)
_CCCL_HOST_API void transform(
  [[maybe_unused]] const __result_policy_base<_Policy>& __policy,
  _CommRange&& __comms,
  _EnvRange&& __envs,
  _InputIterRange&& __input_iters,
  _SizeTRange&& __num_items_range,
  _OutputIterRange&& __output_iters,
  _TransformOp __op)
{
  static_assert(::cuda::std::ranges::sized_range<_CommRange>);
  static_assert(::cuda::std::same_as<_Policy, distributed_t>,
                "Only distributed results are currently supported. Please open an issue at "
                "github.com/NVIDIA/cccl/issue requesting support for your specified policy.");

  using __properties =
    ::cuda::experimental::__detail::__in_range_out_it_properties<_InputIterRange, _OutputIterRange, _EnvRange>;

  // Could use ::cuda::std::invocable here, but it is overkill (compile-time wise). We know
  // that get_stream_t is a normal CPO and normally callable.
  static_assert(::cuda::std::__is_callable_v<::cuda::get_stream_t, typename __properties::__env_type>,
                "Environment must contain a stream");

  // The operator is unary: it maps one input element (which might be a tuple) to one output
  // element. Checking the result is writable through the output iterator catches a mismatched
  // output range here instead of deep inside CUB.
  static_assert(::cuda::std::indirectly_unary_invocable<_TransformOp, typename __properties::__input_iter_type>,
                "The transform operator must be callable with the input iterator's value type");

  static_assert(
    ::cuda::std::indirectly_writable<
      typename __properties::__output_iter_type,
      ::cuda::std::indirect_result_t<
        _TransformOp&,
        ::cuda::std::projected<typename __properties::__input_iter_type, ::cuda::std::identity>>>,
    "The result of the transform operator must be writable through the output iterator");

  _CCCL_NVTX_RANGE_SCOPE("cuda::experimental::transform");

  for (auto&& [__comm, __env, __input_it, __num_items, __output_it] :
       ::cuda::std::ranges::views::zip(__comms, __envs, __input_iters, __num_items_range, __output_iters))
  {
    __CUDAX_MULTI_GPU_DISPATCH(
      __comm.logical_device(),
      CUB_NS_QUALIFIER::DeviceTransform::Transform,
      __input_it,
      __output_it,
      __num_items,
      __op,
      __env);
  }
}

//! @brief Apply a transform operator to a single input over one communicator.
//!
//! Applies `__op` to every element of `__input_iter` and writes the result to the corresponding
//! position of `__output_iter`. The operation is rank-local: `__comm` reads only the elements it
//! owns, and no value moves between ranks. The output therefore holds exactly `__num_items`
//! values, in the same order as the input.
//!
//! This convenience overload forwards one communicator, environment, input iterator, input size,
//! and output iterator to the range-based overload. It is intended for a thread or process that
//! owns one local GPU. See the range overload for a description of the algorithm.
//!
//! @snippet transform/single_comm_basic.cu transform_single_range
//!
//! `__input_iter` must refer to readable device-accessible storage for at least `__num_items`
//! values, and `__output_iter` must refer to writable device-accessible storage for the same
//! number of values. `__output_iter` may alias `__input_iter`, which transforms the input in
//! place. Any other overlap gives undefined behavior.
//!
//! The environment supplies the stream and optional memory resource for the local rank, and is
//! also forwarded to the underlying CUB algorithms.
//!
//! @param[in] __policy The result policy object. Currently must be `cudax::distributed`.
//! @param[in] __comm The communicator.
//! @param[in] __env The execution environment. Must contain a stream.
//! @param[in] __input_iter The local input iterator.
//! @param[in] __num_items The number of items in `__input_iter` to transform.
//! @param[out] __output_iter The local output iterator.
//! @param[in] __op The operator applied to every input element.
_CCCL_TEMPLATE(class _Policy, class _Comm, class _Env, class _InputIt, class _SizeT, class _OutputIt, class _TransformOp)
_CCCL_REQUIRES(__communicator<_Comm> _CCCL_AND ::cuda::std::random_access_iterator<_InputIt>
                 _CCCL_AND ::cuda::std::random_access_iterator<_OutputIt>)
_CCCL_HOST_API void transform(
  const __result_policy_base<_Policy>& __policy,
  _Comm&& __comm,
  _Env&& __env,
  _InputIt __input_iter,
  _SizeT __num_items,
  _OutputIt __output_iter,
  _TransformOp __op)
{
  ::cuda::experimental::transform(
    __policy,
    ::cuda::std::span<::cuda::std::remove_reference_t<_Comm>, 1>{::cuda::std::addressof(__comm), 1},
    ::cuda::std::span<::cuda::std::remove_reference_t<_Env>, 1>{::cuda::std::addressof(__env), 1},
    ::cuda::std::span<_InputIt, 1>{::cuda::std::addressof(__input_iter), 1},
    ::cuda::std::span<_SizeT, 1>{::cuda::std::addressof(__num_items), 1},
    ::cuda::std::span<_OutputIt, 1>{::cuda::std::addressof(__output_iter), 1},
    ::cuda::std::move(__op));
}
} // namespace cuda::experimental

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_TRANSFORM_TRANSFORM_H
