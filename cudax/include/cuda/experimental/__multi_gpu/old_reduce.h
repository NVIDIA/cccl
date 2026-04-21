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

#ifndef _CUDA_EXPERIMENTAL___MULTI_GPU_OLD_REDUCE_H
#define _CUDA_EXPERIMENTAL___MULTI_GPU_OLD_REDUCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/device_reduce.cuh>

#include <thrust/system/cuda/detail/dispatch.h>
#include <thrust/system/cuda/detail/util.h>

#include <cuda/__container/buffer.h>
#include <cuda/__device/device_ref.h>
#include <cuda/__event/event.h>
#include <cuda/__functional/call_or.h>
#include <cuda/__iterator/counting_iterator.h>
#include <cuda/__iterator/transform_iterator.h>
#include <cuda/__memory_pool/device_memory_pool.h>
#include <cuda/__memory_resource/get_memory_resource.h>
#include <cuda/__runtime/ensure_current_context.h>
#include <cuda/__stream/get_stream.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/__utility/no_init.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__concepts/assignable.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__iterator/readable_traits.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/repeat_view.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__ranges/zip_view.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/type_identity.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/cstdint>
#include <cuda/std/inplace_vector>
#include <cuda/std/span>

#include <cuda/experimental/__device/logical_device.cuh>
#include <cuda/experimental/__multi_gpu/concepts.h>
#include <cuda/experimental/__multi_gpu/thread_group.h>

#include <stdexcept>
#include <vector>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental
{
namespace __detail
{
template <class _Buffer, class _Env>
struct __partial_redop_info
{
  _Buffer __buffer;
  _Env __env;
  logical_device __logical_device;
  ::cuda::stream_ref __stream;
  ::cuda::event __event;
};

template <class __buffer_type, class __env_type, class _EnvRange, class _InputRange, class _Tp, class _BinaryOp>
void __local_reduction(_EnvRange& __env_ranges, _InputRange& __input_shards, const _Tp& __identity, _BinaryOp __op)
{
  ::std::vector<__partial_redop_info<__buffer_type, __env_type>> __tmps;

  if constexpr (::cuda::std::ranges::sized_range<_InputRange>)
  {
    const auto __in_size = ::cuda::std::ranges::size(__input_shards);

    __tmps.reserve(__in_size);
    if constexpr (::cuda::std::ranges::sized_range<_EnvRange>)
    {
      _CCCL_ASSERT(::cuda::std::ranges::size(__env_ranges) >= __in_size,
                   "Must have as at least as many environments as inputs");
    }
  }

  for (auto&& [__env, __shard] : ::cuda::std::ranges::views::zip(__env_ranges, __input_shards))
  {
    const logical_device& __logical_device = __shard.device();
    const device_ref __device              = __logical_device.underlying_device();
    const auto _                           = __ensure_current_context{__logical_device.context()};

    const stream_ref __stream = ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{::cudaStream_t{}}, __env);

    auto&& __resource =
      ::cuda::__call_or(::cuda::mr::get_memory_resource, ::cuda::device_default_memory_pool(__device), __env);

    const auto __num_items = ::cuda::std::ranges::size(__shard);

    auto __buff = __buffer_type{__stream, ::cuda::std::move(__resource), /*__size*/ 1, ::cuda::no_init, __env};

    auto __status = ::cudaError_t{};
    THRUST_INDEX_TYPE_DISPATCH(
      __status,
      cub::DeviceReduce::Reduce,
      __num_items,
      (__shard.begin(), __buff.begin(), __num_items_fixed, __op, __identity, __env));
    thrust::cuda_cub::throw_on_error(__status, /*msg=*/"performing gpu-local reduction");

    __tmps.emplace_back(::cuda::std::move(__buff), __env, __logical_device, __stream, __stream.record_event());
  }

  return __tmps;
}

template <class _Iter, ::cuda::std::size_t _Capacity>
struct __deref_iterator_table
{
  ::cuda::std::inplace_vector<_Iter, _Capacity> __iters;

  constexpr __deref_iterator_table(::cuda::std::span<const _Iter> __span)
      : __iters{__span.begin(), __span.end()}
  {}

  [[nodiscard]] _CCCL_DEVICE_API constexpr ::cuda::std::iter_reference_t<_Iter>
  operator()(::cuda::std::uint8_t __i) const noexcept
  {
    return *__iters[__i];
  }
};

template <class _BufferIter, class _Env, class _Tp, class _BinaryOp, class _OutputIt>
void __reduce_temporaries(
  ::cuda::std::span<const _BufferIter> __tmps,
  const logical_device& __device,
  const _Env& __env,
  const _Tp& __identity,
  const _Tp& __init,
  _BinaryOp __op,
  _OutputIt __output)
{
  const auto _ = __ensure_current_context{__device.context()};

  while (!__tmps.empty())
  {
    constexpr auto __capacity = 8;
    const auto __num_items    = __tmps.size();
    const auto __is_last      = __num_items <= __capacity;
    auto __slice              = __is_last ? __tmps : __tmps.first(__capacity);
    auto __values             = ::cuda::make_transform_iterator(
      ::cuda::make_counting_iterator(::cuda::std::size_t{0}), __deref_iterator_table<_BufferIter, __capacity>{__slice});

    auto __status = ::cudaError_t{};
    THRUST_INDEX_TYPE_DISPATCH(
      __status,
      cub::DeviceReduce::Reduce,
      __num_items,
      (__values.begin(), __output, __num_items_fixed, __op, __is_last ? __init : __identity, __env));
    thrust::cuda_cub::throw_on_error(__status, /*msg=*/"performing gpu-local reduction");

    __tmps = __tmps.last(__tmps.size() - __slice.size());
  }
}

template <class _Buffer, class _Env, class _OutputIt>
[[nodiscard]] const __partial_redop_info<_Buffer, _Env>&
__get_matching_tmp(::cuda::std::span<const __partial_redop_info<_Buffer, _Env>> __tmps, _OutputIt __output)
{
  const logical_device __device = __get_device(__output);

  for (auto&& __t : __tmps)
  {
    if (__t.__logical_device == __device)
    {
      return __t;
    }
  }

  _CCCL_THROW(::std::invalid_argument, "Output iterator has no matching device iterator");
}
} // namespace __detail

_CCCL_TEMPLATE(class _EnvRange, class _InputRange, class _OutputIt, class _Tp, class _BinaryOp)
_CCCL_REQUIRES(::cuda::std::ranges::range<_EnvRange> _CCCL_AND __range_of_shards<_InputRange>)
void reduce(const thread_group& __group,
            _EnvRange&& __env_ranges,
            _InputRange&& __input_shards,
            _OutputIt __output,
            _Tp __init,
            ::cuda::std::type_identity_t<_Tp> __identity,
            _BinaryOp __op)
{
  using __shard_type    = ::cuda::std::remove_cvref_t<::cuda::std::ranges::range_reference_t<_InputRange>>;
  using __env_type      = ::cuda::std::remove_cvref_t<::cuda::std::ranges::range_reference_t<_EnvRange>>;
  using __result_type   = ::cuda::std::iter_value_t<_OutputIt>;
  using __resource_type = ::cuda::std::remove_cvref_t<decltype(::cuda::__call_or(
    ::cuda::mr::get_memory_resource,
    ::cuda::device_default_memory_pool(::cuda::std::declval<::cuda::device_ref>()),
    ::cuda::std::declval<__env_type>()))>;
  using __buffer_type   = ::cuda::__buffer_type_for_props<__result_type, typename __resource_type::default_queries>;

  static_assert(::cuda::std::assignable_from<::cuda::std::iter_reference_t<_OutputIt>,
                                             ::cuda::std::iter_reference_t<typename __shard_type::iterator_type>>);
  static_assert(::cuda::std::assignable_from<::cuda::std::iter_reference_t<_OutputIt>, _Tp>);

  auto __tmps = __detail::__local_reduction<__buffer_type, __env_type>(__env_ranges, __input_shards, __identity, __op);
  const auto& __output_info  = __detail::__get_matching_tmp(__tmps, __output);
  const auto __output_stream = __output_info.__stream;
  auto __gathered            = ::std::vector<decltype(__tmps)*>(__group.size());

  __group.all_gather(&__tmps, __gathered);

  auto __flat_iters = ::std::vector<typename __buffer_type::iterator>{};

  // This will under reserve if any thread in the group has more than 1 shard, but is the
  // correct guess in the case of single-thread or thread-per-gpu.
  __flat_iters.reserve(::cuda::std::max(__group.size(), __tmps.size()));
  for (const auto* const __ref_ptr : __gathered)
  {
    for (const auto& __redop_tmp : *__ref_ptr)
    {
      __flat_iters.emplace_back(__redop_tmp.__buff.begin());
      __output_stream.wait_event(__redop_tmp.__event);
    }
  }

  __detail::__reduce_temporaries(
    __flat_iters,
    __output_info.__logical_device,
    __output_info.__env,
    __identity,
    __init,
    ::cuda::std::move(__op),
    ::cuda::std::move(__output));

  const auto __finish_event = __output_stream.record_event();

  __group.barrier();

  for (const auto* const __ref_ptr : __gathered)
  {
    for (const auto& __redop_tmp : *__ref_ptr)
    {
      // This is probably not right.
      //
      // The problem here is that the final reduction is still in flight and we want the
      // buffers to live long enough for all of them to complete (in effect, this is an
      // all-to-all dependency). We are relying on the fact that the memory resources
      // themselves don't scribble anything into the buffers when they are deallocated.
      __redop_tmp.__buff.stream().wait_event(__finish_event);
    }
  }

  // Must barrier here to ensure the pointers are all still alive. Could maybe make copies or
  // use shared pointer so we can avoid this barrier.
  __group.barrier();
}

_CCCL_TEMPLATE(class _InputRange, class _OutputIt, class _Tp)
_CCCL_REQUIRES(__range_of_shards<_InputRange>)
void reduce(const thread_group& __group, _InputRange&& __input_shards, _OutputIt __output, _Tp __identity)
{
  using __result_type = ::cuda::std::iter_value_t<_OutputIt>;

  reduce(__group,
         ::cuda::std::ranges::views::repeat(::cuda::std::execution::env<>{}),
         ::cuda::std::forward<_InputRange>(__input_shards),
         ::cuda::std::move(__output),
         /*__init*/ __result_type{},
         ::cuda::std::move(__identity),
         ::cuda::std::plus<>{});
}
} // namespace cuda::experimental

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_REDUCE_H
