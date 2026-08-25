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

#include <cuda/__functional/operator_properties.h>
#include <cuda/__nvtx/nvtx.h>
#include <cuda/__runtime/ensure_current_context.h>
#include <cuda/__stream/get_stream.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__ranges/zip_view.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/span>

#include <cuda/experimental/__multi_gpu/algorithm/common.h>
#include <cuda/experimental/__multi_gpu/concepts.h>
#include <cuda/experimental/__utility/result_policy.cuh>

#include <vector>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental
{
namespace __detail::__reduce
{
template <class _Buffer, class _Comm, class _Env, class _InputIt, class _SizeT, class _Tp, class _BinaryOp>
[[nodiscard]] _CCCL_HOST_API _Buffer __local_reduction(
  const ::cuda::std::int32_t __ROOT_RANK,
  _Comm&& __comm,
  const _Env& __env,
  _InputIt __input_it,
  const _SizeT& __num_items,
  const _Tp& __init,
  const _BinaryOp& __op,
  const _Tp& __ident)
{
  const auto& __logical_device = __comm.logical_device();
  // Workaround for the case where:
  //
  // 1. The stream is the NULL stream.
  // 2. The resource is the default per-device memory resource.
  // 3. There is no current context set.
  //
  // In this case cuMemAllocFromPool fails with INVALID_CONTEXT because the driver cannot pick
  // an appropriate context to tie the allocation to.
  const auto _                = ::cuda::__ensure_current_context{__logical_device.context()};
  ::cuda::stream_ref __stream = ::cuda::get_stream(__env);
  auto __resource = ::cuda::experimental::__detail::__resource_from_env(__env, __logical_device.underlying_device());

  // Allocate enough storage so that we can use the buffer directly in an in-place comm all
  // gather/all reduce call. Those calls require that the receive buffer is of size nranks *
  // sendcount.
  auto __buff = ::cuda::experimental::__detail::__make_safe_uninitialized_buffer<_Tp>(
    __stream, ::cuda::std::move(__resource), __comm.size(), __env);
  static_assert(::cuda::std::same_as<decltype(__buff), _Buffer>);

  const auto __rank = __comm.rank();

  __CUDAX_MULTI_GPU_DISPATCH(
    __logical_device,
    CUB_NS_QUALIFIER::DeviceReduce::Reduce,
    __input_it,
    // Similarly to above, prepare for the comm calls later. In order for those to be
    // in-place, the sendbuff = recvbuff + rank, so we need to place our partial result
    // there
    __buff.begin() + __rank,
    __num_items,
    __op,
    __rank == __ROOT_RANK ? __init : __ident,
    __env);

  return __buff;
}

template <class _CommRange, class _OutputItRange, class _BinaryOp, class _Buffer>
_CCCL_HOST_API void __direct_reduction(
  _CommRange&& __comms, _OutputItRange&& __outputs, const _BinaryOp& __op, ::std::vector<_Buffer>* __partials)
{
  auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();

  for (auto&& [__comm, __local, __out_it] : ::cuda::std::ranges::views::zip(__comms, *__partials, __outputs))
  {
    __comm.all_reduce(
      __guard,
      __local.data() + __comm.rank(),
      ::cuda::std::to_address(__out_it),
      /*__count=*/1,
      __op,
      __local.stream());
  }
}

template <class _CommRange, class _EnvRange, class _OutputItRange, class _BinaryOp, class _Buffer>
_CCCL_HOST_API void __two_stage_gather_reduction(
  _CommRange&& __comms,
  _EnvRange&& __envs,
  _OutputItRange&& __outputs,
  const _BinaryOp& __op,
  ::std::vector<_Buffer>* __partials)
{
  {
    auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();

    for (auto&& [__comm, __local] : ::cuda::std::ranges::views::zip(__comms, *__partials))
    {
      auto* const __ptr = __local.data();

      __comm.all_gather(__guard, __ptr + __comm.rank(), __ptr, /*__count=*/1, __local.stream());
    }
  }

  for (auto&& [__comm, __env, __buffer, __out] :
       ::cuda::std::ranges::views::zip(__comms, __envs, *__partials, __outputs))
  {
    __CUDAX_MULTI_GPU_DISPATCH(
      __comm.logical_device(),
      CUB_NS_QUALIFIER::DeviceReduce::Reduce,
      __buffer.begin(),
      __out,
      __buffer.size(),
      __op,
      CUB_NS_QUALIFIER::detail::reduce::no_init,
      __env);
  }
}
} // namespace __detail::__reduce

//! @brief Reduce each input range over its communicator and write one result per output
//! iterator.
//!
//! Performs one reduction per communicator in parallel across devices. The communicators,
//! environments, input ranges and output iterators are iterated in lockstep, so for the
//! i-th element of each range the i-th input range is reduced with `__op` seeded by
//! `__init` on the i-th communicator's devices, the partial results are combined across
//! all ranks , and the final value is written to each output iterator. After this
//! routine returns, all output iterators will contain the same value.
//!
//! The passed environments are also passed directly to CUB reductions, and may therefore
//! contain any parameters recognized by CUB.
//!
//! `__init` and `__ident` must have the same value across all ranks calling this routine.
//!
//! This routine is used when the current thread or process owns multiple local GPUs. For
//! example, consider a scenario where there are 8 GPUs and 4 processes such that each
//! process owns 2 GPUs. Then the user would call this routine on each process, passing in both
//! local arrays:
//!
//! @snippet reduce/range_basic.cu reduce
//!
//! All ranges must have the same length. The algorithm will cap iteration to the shortest
//! length, but this should not be relied upon and may change at any time, for any reason. So
//! differing lengths is effectively undefined behavior.
//!
//! Each size in `__num_items_range` may be a plain integral value, or a deferred argument such
//! as `cuda::args::deferred`. A deferred size holds a pointer or iterator to device memory and
//! is read in stream order when the reduction runs, so its value need not be known on the host
//! at the time of the call. This lets a preceding device-side computation produce the number of
//! items to reduce:
//!
//! @snippet reduce/range_deferred.cu reduce_deferred
//!
//! A deferred size must be valid and reachable in stream order on the device that reduces it.
//!
//! The identity element should survive reduction with any other value, returning the original
//! value unchanged. For example, for integers/floats and `cuda::std::plus`, the identity
//! element is 0. For maximum and minimum, the identity values are INT_MIN, and INT_MAX
//! respectively.
//!
//! @param[in] __policy The result policy object. Currently must be `cudax::broadcasted`.
//! @param[in] __comms The range of communicators.
//! @param[in] __envs The range of execution environments. The execution environment must
//!                   contain a stream.
//! @param[in] __input_iters The range of per-communicator input iterators to reduce.
//! @param[in] __num_items_range A range of sizes per input iterator to reduce. Each size may be
//!                              an integral value or a deferred argument.
//! @param[out] __output_iters The range of output iterators receiving the per-communicator results.
//! @param[in] __init The initial value seeding each reduction.
//! @param[in] __op The binary reduction operator.
//! @param[in] __ident The identity element to be used in case of empty ranges.
_CCCL_TEMPLATE(
  class _Policy,
  class _CommRange,
  class _EnvRange,
  class _InputIterRange,
  class _SizeTRange,
  class _OutputIterRange,
  class _Tp       = ::cuda::std::iter_value_t<::cuda::std::ranges::range_reference_t<_InputIterRange>>,
  class _BinaryOp = ::cuda::std::plus<>)
_CCCL_REQUIRES(__range_of_communicators<_CommRange> _CCCL_AND ::cuda::std::ranges::forward_range<_EnvRange> _CCCL_AND
                 __detail::__range_of_random_access_iterators<_InputIterRange>
                   _CCCL_AND ::cuda::std::ranges::forward_range<_SizeTRange> _CCCL_AND
                     __detail::__range_of_output_iters<_OutputIterRange, _Tp>)
_CCCL_HOST_API void reduce(
  [[maybe_unused]] const __result_policy_base<_Policy>& __policy,
  _CommRange&& __comms,
  _EnvRange&& __envs,
  _InputIterRange&& __input_iters,
  _SizeTRange&& __num_items_range,
  _OutputIterRange&& __output_iters,
  _Tp __init     = {},
  _BinaryOp __op = {},
  _Tp __ident    = ::cuda::identity_element<_BinaryOp, _Tp>())
{
  static_assert(::cuda::std::ranges::sized_range<_CommRange>);
  static_assert(::cuda::std::same_as<_Policy, broadcasted_t>,
                "Only broadcasted results are currently supported. Please open an issue at "
                "github.com/NVIDIA/cccl/issue requesting support for your specified policy.");

  using __properties =
    ::cuda::experimental::__detail::__in_range_out_it_properties<_InputIterRange, _OutputIterRange, _EnvRange>;

  static_assert(::cuda::std::__indirectly_binary_reducible<_BinaryOp, _Tp, typename __properties::__input_iter_type>);

  // Could use ::cuda::std::invocable here, but it is overkill (compile-time wise). We know
  // that get_stream_t is a normal CPO and normally callable.
  static_assert(::cuda::std::__is_callable_v<::cuda::get_stream_t, typename __properties::__env_type>,
                "Environment must contain a stream");

  const auto __num_local = ::cuda::std::ranges::size(__comms);

  if (!__num_local)
  {
    return;
  }

  _CCCL_NVTX_RANGE_SCOPE("cuda::experimental::reduce");

  auto __partials = ::std::vector<typename __properties::__buffer_type>{};

  __partials.reserve(__num_local);
  // TODO(jfaibussowit): can just be ranges::zip | ranges::transform | ranges::to() (and then
  // we don't need to do the env, and buffer type deduction upfront)
  for (auto&& [__comm, __env, __input_it, __num_items] :
       ::cuda::std::ranges::views::zip(__comms, __envs, __input_iters, __num_items_range))
  {
    __partials.emplace_back(
      ::cuda::experimental::__detail::__reduce::__local_reduction<typename __properties::__buffer_type>(
        /*__ROOT_RANK=*/0, __comm, __env, __input_it, __num_items, __init, __op, __ident));
  }

  if constexpr (::cuda::experimental::__has_all_reduce<::cuda::std::ranges::range_value_t<_CommRange>,
                                                       typename __properties::__output_type*,
                                                       _BinaryOp>)
  {
    ::cuda::experimental::__detail::__reduce::__direct_reduction(__comms, __output_iters, __op, &__partials);
  }
  else
  {
    ::cuda::experimental::__detail::__reduce::__two_stage_gather_reduction(
      __comms, __envs, __output_iters, __op, &__partials);
  }
}

//! @brief Reduce a single input range over a single communicator using the given execution
//! environment.
//!
//! Convenience wrapper that forwards a single `(communicator, environment, input iterator,
//! input size, output iterator)` to the range-based overload. See the range overload for
//! further discussion.
//!
//! @snippet reduce/single_comm_basic.cu reduce_single_range
//!
//! `__num_items` may be a plain integral value, or a deferred argument such as
//! `cuda::args::deferred`. A deferred size holds a pointer or iterator to device memory and is
//! read in stream order when the reduction runs, so its value need not be known on the host at
//! the time of the call:
//!
//! @snippet reduce/single_comm_deferred.cu reduce_single_range_deferred
//!
//! @param[in] __policy The result policy object. Currently must be `cudax::broadcasted`.
//! @param[in] __comm The communicator.
//! @param[in] __env The execution environment. Must contain a stream.
//! @param[in] __input_iter The input iterator to reduce.
//! @param[in] __num_items The number of items in `__input_iter` to reduce. May be an integral
//!                        value or a deferred argument.
//! @param[out] __output_iter The output iterator receiving the result.
//! @param[in] __init The initial value seeding the reduction.
//! @param[in] __op The binary reduction operator.
//! @param[in] __ident The identity element to be used in case of empty ranges.
_CCCL_TEMPLATE(
  class _Policy,
  class _Comm,
  class _Env,
  class _InputIt,
  class _SizeT,
  class _OutputIt,
  class _Tp       = ::cuda::std::iter_value_t<_InputIt>,
  class _BinaryOp = ::cuda::std::plus<>)
_CCCL_REQUIRES(__communicator<_Comm> _CCCL_AND ::cuda::std::random_access_iterator<_InputIt>
                 _CCCL_AND ::cuda::std::output_iterator<_OutputIt, _Tp>)
_CCCL_HOST_API void reduce(
  const __result_policy_base<_Policy>& __policy,
  _Comm&& __comm,
  _Env&& __env,
  _InputIt __input_iter,
  _SizeT __num_items,
  _OutputIt __output_iter,
  _Tp __init     = {},
  _BinaryOp __op = {},
  _Tp __ident    = ::cuda::identity_element<_BinaryOp, _Tp>())
{
  ::cuda::experimental::reduce(
    __policy,
    ::cuda::std::span<::cuda::std::remove_reference_t<_Comm>, 1>{::cuda::std::addressof(__comm), 1},
    ::cuda::std::span<::cuda::std::remove_reference_t<_Env>, 1>{::cuda::std::addressof(__env), 1},
    ::cuda::std::span<_InputIt, 1>{::cuda::std::addressof(__input_iter), 1},
    ::cuda::std::span<_SizeT, 1>{::cuda::std::addressof(__num_items), 1},
    ::cuda::std::span<_OutputIt, 1>{::cuda::std::addressof(__output_iter), 1},
    ::cuda::std::move(__init),
    ::cuda::std::move(__op),
    ::cuda::std::move(__ident));
}
} // namespace cuda::experimental

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_REDUCE_H
