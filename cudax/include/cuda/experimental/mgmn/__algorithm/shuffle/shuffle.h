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

#ifndef _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SHUFFLE_SHUFFLE_H
#define _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SHUFFLE_SHUFFLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__nvtx/nvtx.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__iterator/distance.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/span>

#include <cuda/experimental/__utility/result_policy.cuh>
#include <cuda/experimental/mgmn/__algorithm/common.h>
#include <cuda/experimental/mgmn/__algorithm/shuffle/execute.h>
#include <cuda/experimental/mgmn/__concepts/common.h>
#include <cuda/experimental/mgmn/__concepts/has_all_gather.h>
#include <cuda/experimental/mgmn/__concepts/has_all_reduce.h>
#include <cuda/experimental/mgmn/__concepts/has_all_to_all.h>
#include <cuda/experimental/mgmn/__concepts/has_all_to_all_v.h>

#include <random>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)
_CCCL_BEGIN_NAMESPACE_CUDA_MGMN
namespace __detail::__hashgraph_shuffle
{
_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

template <class _Policy, class _Comms, class _Envs, class _Inputs, class _Sizes, class _Outputs, class _URBG>
_CCCL_HOST_API void __dispatch(
  const ::cuda::experimental::__result_policy_base<_Policy>&,
  _Comms&& __comms,
  _Envs&& __envs,
  _Inputs&& __inputs,
  _Sizes&& __sizes,
  _Outputs&& __outputs,
  _URBG&& __generator)
{
  static_assert(::cuda::std::same_as<_Policy, ::cuda::experimental::distributed_t>,
                "Only distributed shuffle results are supported");
  static_assert(::cuda::std::ranges::sized_range<_Comms>);
  using _Env      = ::cuda::std::ranges::range_value_t<_Envs>;
  using _Comm     = ::cuda::std::ranges::range_value_t<_Comms>;
  using _InputIt  = ::cuda::std::ranges::range_value_t<_Inputs>;
  using _OutputIt = ::cuda::std::ranges::range_value_t<_Outputs>;
  static_assert(::cuda::std::__is_callable_v<::cuda::get_stream_t, _Env>, "Environment must contain a stream");
  static_assert(::cuda::std::same_as<::cuda::std::iter_value_t<_InputIt>, ::cuda::std::iter_value_t<_OutputIt>>,
                "Shuffle input and output value types must match");
  static_assert(__has_all_gather<_Comm, __detail::__hashgraph_shuffle::__index*>);
  static_assert(__has_all_reduce<_Comm, __detail::__hashgraph_shuffle::__index*>);
  static_assert(__has_all_to_all<_Comm, __detail::__hashgraph_shuffle::__index*>);
  static_assert(
    __has_all_to_all_v<_Comm, __detail::__hashgraph_shuffle::__record<::cuda::std::iter_value_t<_InputIt>>*>);
  const auto __n = ::cuda::std::ranges::size(__comms);
  _CCCL_VERIFY(static_cast<decltype(__n)>(::cuda::std::ranges::distance(__envs)) == __n
                 && static_cast<decltype(__n)>(::cuda::std::ranges::distance(__inputs)) == __n
                 && static_cast<decltype(__n)>(::cuda::std::ranges::distance(__sizes)) == __n
                 && static_cast<decltype(__n)>(::cuda::std::ranges::distance(__outputs)) == __n,
               "Shuffle local rank ranges must have matching lengths");
  if (__n == 0)
  {
    return;
  }
  _CCCL_NVTX_RANGE_SCOPE("cuda::experimental::mgmn::shuffle_copy");
  // Host engines must work without requiring --expt-relaxed-constexpr.
  ::std::uniform_int_distribution<__detail::__hashgraph_shuffle::__index> __distribution;
  const auto __seed = __distribution(__generator);
  ::cuda::experimental::mgmn::__detail::__hashgraph_shuffle::__execute(
    __comms, __envs, __inputs, __sizes, __outputs, __seed);
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT
} // namespace __detail::__hashgraph_shuffle

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

//! @brief Copy a HashGraph permutation of a distributed sequence to distributed output.
//!
//! Communicators, execution environments, input iterators, sizes and output iterators
//! describe the local ranks in lockstep. Every communicator rank participates, even
//! when empty. Each output receives exactly its corresponding input's number of
//! elements. Inputs and outputs must be device-accessible, random-access ranges of
//! the same trivially copyable value type. Output ranges must not overlap each other
//! or input ranges. The single-rank overload can be called concurrently for separate
//! local ranks; use this range overload when one host thread owns multiple GPUs.
//!
//! Each environment supplies a stream and optional memory resource. Communicators
//! must support grouped all-gather, sum all-reduce, all-to-all and variable all-to-all.
//! Work is ordered on the supplied streams; host metadata processing synchronizes
//! intermediate phases. The caller synchronizes its streams before reading outputs.
//!
//! This experimental port preserves the existing HashGraph permutation generator:
//! seed-dependent keys, hash partitioning, exchange and CSR ordering. Atomic insertion
//! order can vary between runs. It does not guarantee uniform sampling of all
//! permutations, fixed-seed repeatability, or the same ordering as thrust::shuffle.
//!
//! The generator is consumed once through a uniform uint64 distribution per call
//! with local communicators, including globally empty inputs. Separate participants
//! must supply generators that produce the same seed. This agrees with one range call
//! per process or concurrently issued single-communicator calls with separate engines.
//! An empty communicator range consumes no generator state.
//!
//! @param __policy Must be @c cudax::distributed.
//! @param __comms Range of local communicators from one communicator world.
//! @param __envs Range of environments, each containing a stream for its rank.
//! @param __inputs Range of input iterators.
//! @param __sizes Range of nonnegative integral local sizes.
//! @param __outputs Range of output iterators, each with its matching local size.
//! @param __generator Uniform random bit generator used to select the common seed.
_CCCL_TEMPLATE(class _Policy, class _Comms, class _Envs, class _Inputs, class _Sizes, class _Outputs, class _URBG)
_CCCL_REQUIRES(
  __range_of_communicators<_Comms> _CCCL_AND ::cuda::std::ranges::forward_range<_Envs> _CCCL_AND
    __detail::__range_of_random_access_iterators<_Inputs> _CCCL_AND ::cuda::std::ranges::forward_range<_Sizes> _CCCL_AND
      __detail::__range_of_random_access_iterators<_Outputs>)
_CCCL_HOST_API void shuffle_copy(
  const ::cuda::experimental::__result_policy_base<_Policy>& __policy,
  _Comms&& __comms,
  _Envs&& __envs,
  _Inputs&& __inputs,
  _Sizes&& __sizes,
  _Outputs&& __outputs,
  _URBG&& __generator)
{
  ::cuda::experimental::mgmn::__detail::__hashgraph_shuffle::__dispatch(
    __policy, __comms, __envs, __inputs, __sizes, __outputs, ::cuda::std::forward<_URBG>(__generator));
}

//! @brief Shuffle a distributed sequence in place using HashGraph.
//!
//! The range and generator contracts are those of @c shuffle_copy. Input values
//! are staged into owned storage before any output is written, so in-place use is
//! safe. Every local range retains its original size. This retains HashGraph's
//! permutation semantics and does not promise uniformity or repeatable ordering.
_CCCL_TEMPLATE(class _Policy, class _Comms, class _Envs, class _Inputs, class _Sizes, class _URBG)
_CCCL_REQUIRES(
  __range_of_communicators<_Comms> _CCCL_AND ::cuda::std::ranges::forward_range<_Envs> _CCCL_AND
    __detail::__range_of_random_access_iterators<_Inputs> _CCCL_AND ::cuda::std::ranges::forward_range<_Sizes>)
_CCCL_HOST_API void shuffle(
  const ::cuda::experimental::__result_policy_base<_Policy>& __policy,
  _Comms&& __comms,
  _Envs&& __envs,
  _Inputs&& __inputs,
  _Sizes&& __sizes,
  _URBG&& __generator)
{
  // The engine stages all inputs before its first write to caller-owned storage.
  ::cuda::experimental::mgmn::__detail::__hashgraph_shuffle::__dispatch(
    __policy, __comms, __envs, __inputs, __sizes, __inputs, ::cuda::std::forward<_URBG>(__generator));
}

//! @brief Copy a distributed HashGraph permutation for one local communicator rank.
//!
//! Equivalent to a one-element range call. Every rank participates concurrently;
//! use the range overload when several local ranks are driven by one host thread.
//! See @c shuffle_copy for storage, execution and randomization requirements.
_CCCL_TEMPLATE(class _Policy, class _Comm, class _Env, class _InputIt, class _Size, class _OutputIt, class _URBG)
_CCCL_REQUIRES(__communicator<_Comm> _CCCL_AND ::cuda::std::random_access_iterator<_InputIt>
                 _CCCL_AND ::cuda::std::random_access_iterator<_OutputIt>)
_CCCL_HOST_API void shuffle_copy(
  const ::cuda::experimental::__result_policy_base<_Policy>& __policy,
  _Comm&& __comm,
  _Env&& __env,
  _InputIt __input,
  _Size __size,
  _OutputIt __output,
  _URBG&& __generator)
{
  ::cuda::experimental::mgmn::shuffle_copy(
    __policy,
    ::cuda::std::span<::cuda::std::remove_reference_t<_Comm>, 1>{::cuda::std::addressof(__comm), 1},
    ::cuda::std::span<::cuda::std::remove_reference_t<_Env>, 1>{::cuda::std::addressof(__env), 1},
    ::cuda::std::span<_InputIt, 1>{::cuda::std::addressof(__input), 1},
    ::cuda::std::span<_Size, 1>{::cuda::std::addressof(__size), 1},
    ::cuda::std::span<_OutputIt, 1>{::cuda::std::addressof(__output), 1},
    ::cuda::std::forward<_URBG>(__generator));
}

//! @brief Shuffle one local communicator rank's distributed input in place.
//!
//! Equivalent to the single-communicator @c shuffle_copy overload with staged
//! in-place output. Separate local ranks must call concurrently with separate
//! generators producing the same seed; see the range overload for details.
_CCCL_TEMPLATE(class _Policy, class _Comm, class _Env, class _InputIt, class _Size, class _URBG)
_CCCL_REQUIRES(__communicator<_Comm> _CCCL_AND ::cuda::std::random_access_iterator<_InputIt>)
_CCCL_HOST_API void shuffle(
  const ::cuda::experimental::__result_policy_base<_Policy>& __policy,
  _Comm&& __comm,
  _Env&& __env,
  _InputIt __input,
  _Size __size,
  _URBG&& __generator)
{
  ::cuda::experimental::mgmn::shuffle(
    __policy,
    ::cuda::std::span<::cuda::std::remove_reference_t<_Comm>, 1>{::cuda::std::addressof(__comm), 1},
    ::cuda::std::span<::cuda::std::remove_reference_t<_Env>, 1>{::cuda::std::addressof(__env), 1},
    ::cuda::std::span<_InputIt, 1>{::cuda::std::addressof(__input), 1},
    ::cuda::std::span<_Size, 1>{::cuda::std::addressof(__size), 1},
    ::cuda::std::forward<_URBG>(__generator));
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT
_CCCL_END_NAMESPACE_CUDA_MGMN
// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SHUFFLE_SHUFFLE_H
