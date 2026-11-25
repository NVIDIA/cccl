// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_EXCHANGE_H
#define _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_EXCHANGE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__container/buffer.h>
#include <cuda/__driver/driver_api.h>
#include <cuda/__memory_resource/legacy_pinned_memory_resource.h>
#include <cuda/launch>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_valid_expansion.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/span>

#include <cuda/experimental/mgmn/__algorithm/sort/context.h>
#include <cuda/experimental/mgmn/__algorithm/sort/device.h>
#include <cuda/experimental/mgmn/__algorithm/sort/local.h>
#include <cuda/experimental/mgmn/__algorithm/sort/select.h>

#include <stdexcept>
#include <vector>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)
_CCCL_BEGIN_NAMESPACE_CUDA_MGMN
namespace __detail::__sort
{
_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

[[nodiscard]] _CCCL_HOST_API inline __index_type
__receive_offset(__index_type __begin, __index_type __count, __index_type& __cursor) noexcept
{
  if (__count == 0)
  {
    return __cursor;
  }
  const auto __offset    = __cursor + __begin % 16;
  __cursor               = __offset + __count;
  const auto __remainder = __cursor % 16;
  if (__remainder != 0)
  {
    __cursor += 16 - __remainder;
  }
  return __offset;
}

template <class _Comm, class _Tp>
using __all_gather_result_t = decltype(::cuda::std::declval<_Comm&>().all_gather(
  ::cuda::std::declval<typename ::cuda::std::remove_cvref_t<_Comm>::group_guard_type&>(),
  ::cuda::std::declval<const _Tp*>(),
  ::cuda::std::declval<_Tp*>(),
  ::cuda::std::declval<::cuda::std::size_t>(),
  ::cuda::std::declval<::cuda::stream_ref>()));

template <class _Comm, class _Tp>
_CCCL_HOST_API void __all_gather(
  _Comm& __comm, const _Tp* __source, _Tp* __result, ::cuda::std::size_t __count, ::cuda::stream_ref __stream)
{
  auto __guard = __comm.group_guard();
  if constexpr (::cuda::std::_IsValidExpansion<__all_gather_result_t, _Comm, _Tp>::value)
  {
    __comm.all_gather(__guard, __source, __result, __count, __stream);
  }
  else
  {
    for (int __peer = 0; __peer < __comm.size(); ++__peer)
    {
      if (__peer == __comm.rank())
      {
        ::cuda::__driver::__memcpyAsync(__result + __count * __peer, __source, __count * sizeof(_Tp), __stream.get());
      }
      else
      {
        __comm.send(__guard, __source, __count, __peer, __stream);
        __comm.recv(__guard, __result + __count * __peer, __count, __peer, __stream);
      }
    }
  }
}

#if defined(_CUDAX_SORT_HAS_NCCL_DEVICE)
// Count, capability flags, window bytes and resident grid share one agreement.
inline constexpr int __device_metadata_width = 4;
inline constexpr int __device_supported = 1, __device_wrong_context = 2, __device_create = 4;

template <class _Comm, class _Env, class _Key, class _Compare>
_CCCL_HOST_API void __sort_device(
  _Comm& __comm,
  const _Env& __env,
  _Key* __keys,
  __index_type __size,
  const ::std::vector<__index_type>& __targets,
  bool __uniform,
  const long long* __metadata,
  bool __gin,
  _Compare __cmp)
{
  const auto __stream = ::cuda::get_stream(__env);
  const __scratch_allocator<_Env> __allocate{__env, __stream};
  const int __ranks = __comm.size(), __rank = __comm.rank();
  const auto __width      = static_cast<::cuda::std::size_t>(__ranks) + 1;
  __index_type __capacity = 1;
  for (int __peer = 0; __peer < __ranks; ++__peer)
  {
    __capacity = ::cuda::std::max(__capacity, __targets[__peer + 1] - __targets[__peer]);
  }
  __device_layout __layout{};
  __layout.__source_     = __layout.template __append<__selection_message<_Key>>(__width);
  __layout.__gathered_   = __layout.template __append<__selection_message<_Key>>(__width * __ranks);
  __layout.__local_cuts_ = __layout.template __append<__index_type>(__width);
  __layout.__cuts_       = __layout.template __append<__index_type>(__width * __ranks);
  __layout.__targets_    = __layout.template __append<__index_type>(__width);
  __layout.__sorted_     = __layout.template __append<_Key>(__capacity);
  __layout.__received_   = __layout.template __append<_Key>(__capacity + __index_type{30} * __ranks);
  ::cuda::experimental::mgmn::sort_context __temporary;
  auto* __persistent = ::cuda::__call_or(::cuda::experimental::mgmn::__sort_context_query{}, &__temporary, __env);
  auto& __owner      = (__persistent ? *__persistent : __temporary).__window();
  // The local flag was included in the collective metadata. Keep that invariant
  // explicit here too, so an empty temporary context never reaches the reuse path.
  int __actions = __owner ? 0 : __device_create;
  bool __grow   = false;
  int __blocks  = __gin ? __transfer_blocks : __lsa_transfer_blocks;
  for (int __peer = 0; __peer < __ranks; ++__peer)
  {
    const auto __flags = __metadata[__device_metadata_width * __peer + 1];
    _CCCL_ASSERT(__flags & __device_supported, "Device backend requires collective capability agreement");
    __actions |= static_cast<int>(__flags);
    const auto __bytes = ::cuda::std::bit_cast<::cuda::std::uint64_t>(__metadata[__device_metadata_width * __peer + 2]);
    __grow |= __bytes < __layout.__bytes_;
    __blocks = ::cuda::std::min(__blocks, static_cast<int>(__metadata[__device_metadata_width * __peer + 3]));
  }
  if (__gin)
  {
    // Small transfers favor fewer queue operations; larger ones benefit from
    // concurrent puts. Capacity is the collective maximum, so all ranks use
    // the same grid even when their local input counts differ.
    constexpr ::cuda::std::size_t __chunk_bytes = 1 << 20;
    const auto __wanted                         = ::cuda::std::max<::cuda::std::size_t>(
      8, (static_cast<::cuda::std::size_t>(__capacity) * sizeof(_Key) - 1) / __chunk_bytes + 1);
    __blocks =
      ::cuda::std::min(__blocks, static_cast<int>(::cuda::std::min<::cuda::std::size_t>(__transfer_blocks, __wanted)));
  }
  if (__actions & __device_wrong_context)
  {
    _CCCL_THROW(::std::invalid_argument, "Sort context belongs to a different NCCL communicator or CUDA device");
  }
  // Creation and growth are collective even if only one rank reset its context.
  if (__actions & __device_create)
  {
    __owner.reset();
    __owner.reset(new __device_window{__comm.native_handle(), __stream.get(), __layout.__bytes_, __gin});
  }
  else
  {
    __owner->__set_stream(__stream.get());
    if (__grow)
    {
      __owner->__reserve(__layout.__bytes_);
    }
  }
  auto& __window = *__owner;
  // Barrier sessions retain their epochs across calls. Reset signals only with
  // newly created resources; resetting them on reuse would strand those epochs.
  if (__gin && (__actions & __device_create))
  {
    ::cuda::launch(
      __stream,
      ::cuda::make_config(::cuda::block_dims<256>(), ::cuda::grid_dims(__window.__device_.ginContextCount)),
      ::cuda::experimental::mgmn::__detail::__sort::__reset_gin_signals<::ncclDevComm>,
      __window.__device_);

    // Every rank must finish its reset before any rank can send a signal.
    // These buffers are needed only when a communicator is newly created.
    auto __capability_source = __allocate.template operator()<int>(1);
    auto __capability_result = __allocate.template operator()<int>(__ranks);
    ::cuda::__driver::__memsetAsync(
      __capability_source.data(), static_cast<unsigned char>(0), sizeof(int), __stream.get());
    ::cuda::experimental::mgmn::__detail::__sort::__all_gather(
      __comm, __capability_source.data(), __capability_result.data(), 1, __stream);
    ::cuda::__driver::__streamSynchronize(__stream.get());
  }
  auto __states    = __allocate.template operator()<__selection_state<_Key>>(__width);
  auto* __sorted   = __window.template __at<_Key>(__layout.__sorted_);
  auto* __received = __window.template __at<_Key>(__layout.__received_);
  ::cuda::__driver::__memcpyAsync(
    __window.template __at<__index_type>(__layout.__targets_),
    __targets.data(),
    __width * sizeof(__index_type),
    __stream.get());
  ::cuda::experimental::mgmn::__detail::__sort::__local_sort(__keys, __sorted, __size, __cmp, __env);
  if (__gin)
  {
    ::cuda::launch(
      __stream,
      ::cuda::make_config(::cuda::block_dims<256>(), ::cuda::grid_dims(1)),
      ::cuda::experimental::mgmn::__detail::__sort::__device_select<true, _Key, _Compare>,
      __window.__device_,
      __window.__world_barrier_,
      __window.__window_,
      __layout,
      __size,
      __states.data(),
      __uniform,
      __cmp);
  }
  else
  {
    ::cuda::launch(
      __stream,
      ::cuda::make_config(::cuda::block_dims<256>(), ::cuda::grid_dims(1)),
      ::cuda::experimental::mgmn::__detail::__sort::__device_select<false, _Key, _Compare>,
      __window.__device_,
      __window.__world_barrier_,
      __window.__window_,
      __layout,
      __size,
      __states.data(),
      __uniform,
      __cmp);
  }

  ::cuda::std::span<__index_type> __cuts{
    static_cast<__index_type*>(__window.__host_buffer(1, __width * __ranks * sizeof(__index_type))), __width * __ranks};
  ::cuda::__driver::__memcpyAsync(
    __cuts.data(),
    __window.template __at<__index_type>(__layout.__cuts_),
    __width * __ranks * sizeof(__index_type),
    __stream.get());
  if (__gin)
  {
    ::cuda::launch(
      __stream,
      ::cuda::make_config(::cuda::block_dims<256>(), ::cuda::grid_dims(__blocks)),
      ::cuda::experimental::mgmn::__detail::__sort::__device_redistribute<true, _Key>,
      __window.__device_,
      __window.__world_barrier_,
      __window.__window_,
      __layout);
  }
  else if (::cuda::experimental::mgmn::__detail::__sort::__use_rotated_redistribution<_Key>(__size))
  {
    ::cuda::launch(
      __stream,
      ::cuda::make_config(::cuda::block_dims<256>(), ::cuda::grid_dims(__blocks)),
      ::cuda::experimental::mgmn::__detail::__sort::__device_redistribute_rotated<_Key>,
      __window.__device_,
      __window.__world_barrier_,
      __window.__window_,
      __layout);
  }
  else
  {
    ::cuda::launch(
      __stream,
      ::cuda::make_config(::cuda::block_dims<256>(), ::cuda::grid_dims(__blocks)),
      ::cuda::experimental::mgmn::__detail::__sort::__device_redistribute<false, _Key>,
      __window.__device_,
      __window.__world_barrier_,
      __window.__window_,
      __layout);
  }

  ::cuda::__driver::__streamSynchronize(__stream.get());
  ::std::vector<__run> __runs;
  __index_type __cursor = 0;
  for (int __peer = 0; __peer < __ranks; ++__peer)
  {
    const auto __begin  = __cuts[__peer * __width + __rank];
    const auto __count  = __cuts[__peer * __width + __rank + 1] - __begin;
    const auto __offset = ::cuda::experimental::mgmn::__detail::__sort::__receive_offset(__begin, __count, __cursor);
    if (__count)
    {
      __runs.push_back({__offset, __count});
    }
  }
  ::cuda::experimental::mgmn::__detail::__sort::__merge_received(
    __received, __size, ::cuda::std::move(__runs), __keys, __cmp, __env);
  ::cuda::__driver::__streamSynchronize(__stream.get());
}
#endif // _CUDAX_SORT_HAS_NCCL_DEVICE

template <class _Comm, class _Env, class _Key, class _Compare>
_CCCL_HOST_API void __sort_host(
  _Comm& __comm,
  const _Env& __env,
  _Key* __keys,
  __index_type __size,
  const ::std::vector<__index_type>& __targets,
  bool __uniform,
  _Compare __cmp)
{
  const auto __stream = ::cuda::get_stream(__env);
  const __scratch_allocator<_Env> __allocate{__env, __stream};
  const int __rank = __comm.rank(), __ranks = __comm.size();
  auto __sorted = __allocate.template operator()<_Key>(::cuda::std::max<::cuda::std::int64_t>(__size, 1));
  ::cuda::experimental::mgmn::__detail::__sort::__local_sort(__keys, __sorted.data(), __size, __cmp, __env);
  if (__ranks == 1)
  {
    ::cuda::__driver::__memcpyAsync(__keys, __sorted.data(), __size * sizeof(_Key), __stream.get());
    return;
  }

  const ::cuda::std::size_t __width = static_cast<::cuda::std::size_t>(__ranks) + 1;
  auto __source                     = __allocate.template operator()<__selection_message<_Key>>(__width);
  auto __gathered                   = __allocate.template operator()<__selection_message<_Key>>(__width * __ranks);
  auto __states                     = __allocate.template operator()<__selection_state<_Key>>(__ranks + 1);
  auto __device_targets             = __allocate.template operator()<::cuda::std::int64_t>(__ranks + 1);
  auto __unfinished                 = __allocate.template operator()<int>(1);
  ::cuda::host_buffer<int> __host_unfinished{
    __stream, ::cuda::mr::legacy_pinned_memory_resource{__stream.device()}, 1, ::cuda::no_init};
  ::cuda::host_buffer<::cuda::std::int64_t> __cuts{
    __stream,
    ::cuda::mr::legacy_pinned_memory_resource{__stream.device()},
    static_cast<::cuda::std::size_t>(__ranks + 1) * __ranks,
    ::cuda::no_init};
  ::cuda::__driver::__memcpyAsync(
    __device_targets.data(), __targets.data(), (__ranks + 1) * sizeof(::cuda::std::int64_t), __stream.get());
  auto __local_cuts    = __allocate.template operator()<::cuda::std::int64_t>(__width);
  auto __gathered_cuts = __allocate.template operator()<::cuda::std::int64_t>(__width * __ranks);
  auto __step          = [&](auto __phase) {
    ::cuda::launch(
      __stream,
      ::cuda::make_config(::cuda::block_dims<256>(), ::cuda::grid_dims(1)),
      ::cuda::experimental::mgmn::__detail::__sort::__select<decltype(__phase)::value, _Key, _Compare>,
      __sorted.data(),
      __size,
      __rank,
      __ranks,
      __source.data(),
      __gathered.data(),
      __states.data(),
      __unfinished.data(),
      __cmp,
      __device_targets.data(),
      __uniform,
      __local_cuts.data());
  };
  using __phase = __selection_phase;
  __step(::cuda::std::integral_constant<__phase, __phase::__initialize>{});
  ::cuda::experimental::mgmn::__detail::__sort::__all_gather(
    __comm, __source.data(), __gathered.data(), __width, __stream);
  __step(::cuda::std::integral_constant<__phase, __phase::__first_count>{});
  do
  {
    ::cuda::experimental::mgmn::__detail::__sort::__all_gather(
      __comm, __source.data(), __gathered.data(), __width, __stream);
    __step(::cuda::std::integral_constant<__phase, __phase::__advance>{});
    ::cuda::__driver::__memcpyAsync(__host_unfinished.data(), __unfinished.data(), sizeof(int), __stream.get());
    ::cuda::__driver::__streamSynchronize(__stream.get());
  } while (__host_unfinished[0]);
  __step(::cuda::std::integral_constant<__phase, __phase::__export_cuts>{});
  ::cuda::experimental::mgmn::__detail::__sort::__all_gather(
    __comm, __local_cuts.data(), __gathered_cuts.data(), __ranks + 1, __stream);
  ::cuda::__driver::__memcpyAsync(
    __cuts.data(),
    __gathered_cuts.data(),
    ::cuda::std::size_t(__ranks + 1) * __ranks * sizeof(long long),
    __stream.get());
  ::cuda::__driver::__streamSynchronize(__stream.get());
  auto __cut = [&](int __boundary, int __peer) -> ::cuda::std::int64_t {
    return __cuts[::cuda::std::size_t(__peer) * (__ranks + 1) + __boundary];
  };

  auto __received = __allocate.template operator()<_Key>(__size + ::cuda::std::int64_t{30} * __ranks);
  ::std::vector<__run> __runs;
  ::cuda::std::int64_t __cursor = 0;
  {
    auto __guard = __comm.group_guard();
    for (int __peer = 0; __peer < __ranks; ++__peer)
    {
      const ::cuda::std::int64_t __begin = __cut(__rank, __peer), __count = __cut(__rank + 1, __peer) - __begin;
      const ::cuda::std::int64_t __offset =
        ::cuda::experimental::mgmn::__detail::__sort::__receive_offset(__begin, __count, __cursor);
      if (__count)
      {
        __runs.push_back({__offset, __count});
      }
      if (__peer == __rank)
      {
        if (__count)
        {
          ::cuda::__driver::__memcpyAsync(
            __received.data() + __offset, __sorted.data() + __begin, __count * sizeof(_Key), __stream.get());
        }
      }
      else
      {
        const ::cuda::std::int64_t __send_begin  = __cut(__peer, __rank),
                                   __send_count  = __cut(__peer + 1, __rank) - __send_begin;
        const ::cuda::std::int64_t __send_prefix = __send_begin % 16;
        const ::cuda::std::int64_t __recv_prefix = __begin % 16;
        if (__send_count)
        {
          __comm.send(
            __guard, __sorted.data() + __send_begin - __send_prefix, __send_count + __send_prefix, __peer, __stream);
        }
        if (__count)
        {
          __comm.recv(__guard, __received.data() + __offset - __recv_prefix, __count + __recv_prefix, __peer, __stream);
        }
      }
    }
  }
  // Makes receive data and async scratch allocations visible to merge streams.
  ::cuda::__driver::__streamSynchronize(__stream.get());
  ::cuda::experimental::mgmn::__detail::__sort::__merge_received(
    __received.data(), __size, ::cuda::std::move(__runs), __keys, __cmp, __env);
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT
} // namespace __detail::__sort
_CCCL_END_NAMESPACE_CUDA_MGMN
// NOLINTEND(bugprone-reserved-identifier)
#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_EXCHANGE_H
