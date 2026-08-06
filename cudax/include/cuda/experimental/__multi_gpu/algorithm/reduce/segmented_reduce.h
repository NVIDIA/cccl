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

#ifndef _CUDA_EXPERIMENTAL___MULTI_GPU_SEGMENTED_REDUCE_H
#define _CUDA_EXPERIMENTAL___MULTI_GPU_SEGMENTED_REDUCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/device_transform.cuh>

#include <cuda/__functional/operator_properties.h>
#include <cuda/__runtime/ensure_current_context.h>
#include <cuda/__stream/get_stream.h>
#include <cuda/std/__bit/integral.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__functional/identity.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__iterator/readable_traits.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__ranges/zip_view.h>
#include <cuda/std/__tuple_dir/tie.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/__multi_gpu/algorithm/common.h>
#include <cuda/experimental/__multi_gpu/concepts.h>
#include <cuda/experimental/__utility/result_policy.cuh>

#include <vector>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental
{
namespace __detail::__segmented_reduce
{
template <bool __has_direct_reduction,
          class _Buffer,
          class _Comm,
          class _Env,
          class _InputIt,
          class _OffsetBeginIt,
          class _OffsetEndIt,
          class _Tp,
          class _BinaryOp>
[[nodiscard]] _CCCL_HOST_API _Buffer __local_reduction(
  ::cuda::std::int32_t __ROOT_RANK,
  _Comm&& __comm,
  const _Env& __env,
  _InputIt __input_it,
  ::cuda::std::size_t __num_segments,
  _OffsetBeginIt __offsets_begin_it,
  _OffsetEndIt __offsets_end_it,
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

  // One partial per segment. The butterfly fallback folds in place, but needs extra room for
  // the other ranks' data so we allocate it here
  auto __buff = ::cuda::experimental::__detail::__make_safe_uninitialized_buffer<_Tp>(
    __stream, ::cuda::std::move(__resource), __has_direct_reduction ? __num_segments : 2 * __num_segments, __env);
  static_assert(::cuda::std::same_as<decltype(__buff), _Buffer>);

  const auto __rank = __comm.rank();

  __CUDAX_MULTI_GPU_DISPATCH(
    __logical_device,
    CUB_NS_QUALIFIER::DeviceSegmentedReduce::Reduce,
    __input_it,
    __buff.begin(),
    __num_segments,
    __offsets_begin_it,
    __offsets_end_it,
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
    // TODO(jfaibussowit):
    //
    // We need to handle non contiguous output iterators properly here
    __comm.all_reduce(
      __guard, __local.data(), ::cuda::std::to_address(__out_it), __local.size(), __op, __local.stream());
  }
}

template <class _CommRange, class _EnvRange, class _BinaryOp, class _Pred, class _Buffer>
_CCCL_HOST_API void __exchange_and_fold(
  _CommRange&& __comms,
  _EnvRange&& __envs,
  ::cuda::std::size_t __num_segments,
  const _BinaryOp& __op,
  _Pred __participates,
  ::cuda::std::int32_t __peer_mask,
  ::std::vector<_Buffer>* __local_buffers)
{
  {
    auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();

    for (auto&& [__comm, __buf] : ::cuda::std::ranges::views::zip(__comms, *__local_buffers))
    {
      if (const auto __rank = __comm.rank(); __participates(__rank))
      {
        const auto __peer = __rank ^ __peer_mask;
        auto __send       = __buf.subspan(0, __num_segments);
        auto __recv       = __buf.subspan(__num_segments, __num_segments);

        __comm.send(__guard, __send.data(), __num_segments, __peer, __buf.stream());
        __comm.recv(__guard, __recv.data(), __num_segments, __peer, __buf.stream());
      }
    }
  }

  for (auto&& [__comm, __env, __buf] : ::cuda::std::ranges::views::zip(__comms, __envs, *__local_buffers))
  {
    if (__participates(__comm.rank()))
    {
      auto __send = __buf.subspan(0, __num_segments);
      auto __recv = __buf.subspan(__num_segments, __num_segments);

      __CUDAX_MULTI_GPU_DISPATCH(
        __comm.logical_device(),
        CUB_NS_QUALIFIER::DeviceTransform::Transform,
        ::cuda::std::make_tuple(__send.data(), __recv.data()),
        __send.data(),
        __num_segments,
        __op,
        __env);
    }
  }
}

template <class _CommRange, class _Buffer>
_CCCL_HOST_API void __broadcast_to_excess(
  _CommRange&& __comms,
  ::cuda::std::size_t __num_segments,
  ::std::vector<_Buffer>* __partials,
  ::cuda::std::int32_t __excess,
  ::cuda::std::int32_t __pow2)
{
  auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();

  for (auto&& [__comm, __local] : ::cuda::std::ranges::views::zip(__comms, *__partials))
  {
    const auto __rank = __comm.rank();

    if (__rank < __excess)
    {
      // It actually doesn't matter which rank we pick here, so long as __rank < __pow2, since
      // all ranks inside the butterfly will have a copy of the results.
      //
      // We choose the first __excess ranks purely to stay symmetrical with the original
      // folding step when the excess ranks dropped out.
      //
      // And furthermore, we don't even have send from different ranks at all. We could just
      // choose a single rank that broadcasts the results to all the excess ranks, but the idea
      // here is that surely overlapping the comms among multiple GPUs is faster.
      //
      // Another potential optimization here is to choose a rank that is "closest" to the
      // receiving rank (for example, on the same board or node), but this requires us to
      // inspect the network topology. Something to consider in future optimizations.
      __comm.send(__guard, __local.data(), __num_segments, __rank + __pow2, __local.stream());
    }
    else if (__rank >= __pow2)
    {
      __comm.recv(__guard, __local.data(), __num_segments, __rank - __pow2, __local.stream());
    }
  }
}

template <class _CommRange, class _EnvRange, class _OutputItRange, class _BinaryOp, class _Buffer>
_CCCL_HOST_API void __butterfly_reduction(
  _CommRange&& __comms,
  _EnvRange&& __envs,
  ::cuda::std::size_t __num_segments,
  _OutputItRange&& __outputs,
  const _BinaryOp& __op,
  ::std::vector<_Buffer>* __partials)
{
  // k: comm size (__comm_size)
  // n: local size (__num_segments)
  //
  // Combine the per-rank partials with a butterfly all-reduce. We implement such a complicated
  // algorithm is because a naive all-gather and local reduce requires `k*n` local
  // storage. This is not so bad when `n=1` (like in normal reduce), but here `n` is
  // `__num_segments` and potentially quite large. Butterfly reductions require only `2n`
  // temporary storage and finish in `log2(k)` steps. The tradeoff is that each step
  // communicates roughly `n*log(k)` bytes which is somewhat high.
  //
  // Every rank starts with its own `__num_segments` partials and ends with the full
  // reduction. Round `d` exchanges the whole accumulator with peer `rank ^ d` and folds the
  // received values in place, so after `log2(k)` rounds every rank has combined every other
  // rank's contribution.
  //
  // `__comm_size` is not required to be a power of two. The largest power of two `__pow2 <=
  // __comm_size` is used for the butterfly; the excess `__comm_size - __pow2` high ranks fold
  // their partials into the matching low ranks first and receive the final result at the end.
  //
  // `bit_floor` requires an unsigned type, so the rank count is widened to `uint32_t` for it and
  // the result is brought back to the signed rank type the communicator uses.
  const auto __comm_size = ::cuda::std::ranges::begin(__comms)->size();
  const auto __pow2 =
    static_cast<::cuda::std::int32_t>(::cuda::std::bit_floor(static_cast<::cuda::std::uint32_t>(__comm_size)));
  const auto __excess = __comm_size - __pow2;

  if (__excess)
  {
    // Reduce to a power-of-two number of participants. Ranks in `[__pow2, __size)` (which
    // should be `__excess` ranks) fold their partials into ranks `[0, __excess)` and then sit
    // out the butterfly.
    const auto __participates = [=](::cuda::std::int32_t __rank) {
      return __rank < __excess || __rank >= __pow2;
    };

    ::cuda::experimental::__detail::__segmented_reduce::__exchange_and_fold(
      __comms, __envs, __num_segments, __op, __participates, __pow2, __partials);
  }

  for (::cuda::std::int32_t __peer_mask = 1; __peer_mask < __pow2; __peer_mask <<= 1)
  {
    // The butterfly proper, over the low `__pow2` ranks.
    const auto __participates = [=](::cuda::std::int32_t __rank) {
      return __rank < __pow2;
    };

    ::cuda::experimental::__detail::__segmented_reduce::__exchange_and_fold(
      __comms, __envs, __num_segments, __op, __participates, __peer_mask, __partials);
  }

  if (__excess)
  {
    // Every rank in `[0, __pow2)` now holds the full result. Send it back to the excess ranks
    // that dropped out above.
    ::cuda::experimental::__detail::__segmented_reduce::__broadcast_to_excess(
      __comms, __num_segments, __partials, __excess, __pow2);
  }

  // Finally, copy to __out_it. This could actually just be a DeviceCopy if OutputIt is known
  // to be contiguous.
  //
  // We could also do an optimization where the final round reduces into __out_it directly, but
  // this requires pretty careful handling. The exchange and fold would need to handle a third
  // parameter for the transform output. We would also potentially need special handling in the
  // broadcast_to_excess() step since __out_it would need to be contiguous in order to be
  // written to directly.
  //
  // All in all *probably* not worth the pain of implementing.
  for (auto&& [__comm, __env, __local, __out_it] :
       ::cuda::std::ranges::views::zip(__comms, __envs, *__partials, __outputs))
  {
    __CUDAX_MULTI_GPU_DISPATCH(
      __comm.logical_device(),
      CUB_NS_QUALIFIER::DeviceTransform::Transform,
      __local.data(),
      __out_it,
      __num_segments,
      ::cuda::std::identity{},
      __env);
  }
}
} // namespace __detail::__segmented_reduce

_CCCL_TEMPLATE(
  class _Policy,
  class _CommRange,
  class _EnvRange,
  class _InputIterRange,
  class _OffsetBeginIterRange,
  class _OffsetEndIterRange,
  class _OutputIterRange,
  class _Tp       = ::cuda::std::iter_value_t<::cuda::std::ranges::range_reference_t<_InputIterRange>>,
  class _BinaryOp = ::cuda::std::plus<>)
_CCCL_REQUIRES(__range_of_communicators<_CommRange> _CCCL_AND //
               ::cuda::std::ranges::forward_range<_EnvRange> _CCCL_AND //
                 __detail::__range_of_random_access_iterators<_InputIterRange> _CCCL_AND //
               ::cuda::std::ranges::forward_range<_OffsetBeginIterRange> _CCCL_AND //
               ::cuda::std::ranges::forward_range<_OffsetEndIterRange> _CCCL_AND //
                 __detail::__range_of_output_iters<_OutputIterRange, _Tp>)
_CCCL_HOST_API void segmented_reduce(
  [[maybe_unused]] const __result_policy_base<_Policy>& __policy,
  _CommRange&& __comms,
  _EnvRange&& __envs,
  _InputIterRange&& __input_iters,
  ::cuda::std::size_t __num_segments,
  _OffsetBeginIterRange&& __offset_begin_iters,
  _OffsetEndIterRange&& __offset_end_iters,
  _OutputIterRange&& __output_iters,
  _Tp __init     = {},
  _BinaryOp __op = {},
  _Tp __ident    = ::cuda::identity_element<_BinaryOp, _Tp>())
{
  static_assert(::cuda::std::ranges::sized_range<_CommRange>);

  // NOTE: if we want to support cudax::distributed:
  //
  // 1. The direct reduction just becomes comm.reduce_scatter().
  // 2. The backup implementation should implement Rabenseifners method described in
  //    "Optimization of Collective Communication Operations in MPICH"
  //    (https://web.cels.anl.gov/~thakur/papers/ijhpca-coll.pdf)
  //
  // Ring is also possible but not really that optimal as __comm_size increases:
  //
  // k: number of ranks
  // n: local array size
  //
  //                  steps        bytes/rank      memory
  // butterfly        log2(k)      n*log2(k)       2n
  // Rabenseifner     2*log2(k)    ~2n             1.5n
  // ring             2(k-1)       ~2n             n + n/k
  static_assert(::cuda::std::same_as<_Policy, broadcasted_t>,
                "Only broadcasted results are currently supported. Please open an issue at "
                "github.com/NVIDIA/cccl/issue requesting support for your specified policy.");

  using __properties =
    ::cuda::experimental::__detail::__in_range_out_it_properties<_InputIterRange, _OutputIterRange, _EnvRange>;

  const auto __num_local = ::cuda::std::ranges::size(__comms);

  if (!__num_local)
  {
    return;
  }

  _CCCL_NVTX_RANGE_SCOPE("cuda::experimental::segmented_reduce");

  auto __partials                      = ::std::vector<typename __properties::__buffer_type>{};
  constexpr bool __comm_has_all_reduce = ::cuda::experimental::
    __has_all_reduce<::cuda::std::ranges::range_value_t<_CommRange>, typename __properties::__output_type*, _BinaryOp>;

  __partials.reserve(__num_local);
  for (auto&& [__comm, __env, __input_it, __offsets_begin_it, __offsets_end_it] :
       ::cuda::std::ranges::views::zip(__comms, __envs, __input_iters, __offset_begin_iters, __offset_end_iters))
  {
    __partials.emplace_back(
      ::cuda::experimental::__detail::__segmented_reduce::__local_reduction<__comm_has_all_reduce,
                                                                            typename __properties::__buffer_type>(
        /*__ROOT_RANK=*/0,
        __comm,
        __env,
        __input_it,
        __num_segments,
        __offsets_begin_it,
        __offsets_end_it,
        __init,
        __op,
        __ident));
  }

  if constexpr (__comm_has_all_reduce)
  {
    ::cuda::experimental::__detail::__segmented_reduce::__direct_reduction(__comms, __output_iters, __op, &__partials);
  }
  else
  {
    ::cuda::experimental::__detail::__segmented_reduce::__butterfly_reduction(
      __comms, __envs, __num_segments, __output_iters, __op, &__partials);
  }
}
} // namespace cuda::experimental

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_SEGMENTED_REDUCE_H
