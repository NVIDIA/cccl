// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_EXECUTE_H
#define _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_EXECUTE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/device_merge.cuh>
#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <cuda/__memory_resource/allocation_alignment.h>
#include <cuda/__runtime/api_wrapper.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__exception/cuda_error.h>
#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__iterator/distance.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__type_traits/is_valid_expansion.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/swap.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <cuda/experimental/mgmn/__algorithm/common.h>
#include <cuda/experimental/mgmn/__algorithm/sort/context.h>
#include <cuda/experimental/mgmn/__algorithm/sort/device.h>
#include <cuda/experimental/mgmn/__algorithm/sort/select.h>

#include <condition_variable>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)
_CCCL_BEGIN_NAMESPACE_CUDA_MGMN
namespace __detail::__sort
{
_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

_CCCL_HOST_API inline void __check(::cudaError_t __error)
{
  if (__error != ::cudaSuccess)
  {
    _CCCL_THROW(::cuda::cuda_error, __error, "executing distributed sort");
  }
}

template <class _Tp>
class __pinned_buffer
{
  _Tp* __ptr_   = nullptr;
  bool __owned_ = false;

public:
  _CCCL_HOST_API explicit __pinned_buffer(::cuda::std::size_t __count, _Tp* __borrowed = nullptr)
      : __ptr_{__borrowed}
      , __owned_{__borrowed == nullptr}
  {
    if (__owned_)
    {
      _CCCL_TRY_RUNTIME_API(::cudaMallocHost, "allocating sort metadata", &__ptr_, __count * sizeof(_Tp));
    }
  }
  _CCCL_HOST_API ~__pinned_buffer()
  {
    if (__owned_)
    {
      _CCCL_ASSERT_RUNTIME_API(::cudaFreeHost, "freeing sort metadata", __ptr_);
    }
  }
  __pinned_buffer(const __pinned_buffer&)            = delete;
  __pinned_buffer& operator=(const __pinned_buffer&) = delete;
  [[nodiscard]] _CCCL_HOST_API _Tp* __data() const noexcept
  {
    return __ptr_;
  }
  [[nodiscard]] _CCCL_HOST_API _Tp& operator[](::cuda::std::size_t __idx) const noexcept
  {
    return __ptr_[__idx];
  }
};

class __merge_streams
{
  ::std::vector<::cudaStream_t> __streams_;
  _CCCL_HOST_API void __release() noexcept
  {
    for (const auto __stream : __streams_)
    {
      // Join outstanding merges on exception paths before freeing their inputs.
      _CCCL_ASSERT_RUNTIME_API(::cudaStreamSynchronize, "waiting for sort merge", __stream);
      _CCCL_ASSERT_RUNTIME_API(::cudaStreamDestroy, "destroying sort merge stream", __stream);
    }
    __streams_.clear();
  }

public:
  _CCCL_HOST_API explicit __merge_streams(::cuda::std::size_t __count)
  {
    __streams_.reserve(__count);
    _CCCL_TRY
    {
      for (::cuda::std::size_t __idx = 0; __idx < __count; ++__idx)
      {
        ::cudaStream_t __stream;
        _CCCL_TRY_RUNTIME_API(
          ::cudaStreamCreateWithFlags, "creating sort merge stream", &__stream, cudaStreamNonBlocking);
        __streams_.push_back(__stream);
      }
    }
    _CCCL_CATCH_ALL
    {
      __release();
      _CCCL_RETHROW;
    }
  }
  _CCCL_HOST_API ~__merge_streams()
  {
    __release();
  }
  __merge_streams(const __merge_streams&)            = delete;
  __merge_streams& operator=(const __merge_streams&) = delete;
  [[nodiscard]] _CCCL_HOST_API ::cudaStream_t operator[](::cuda::std::size_t __idx) const noexcept
  {
    return __streams_[__idx];
  }
  _CCCL_HOST_API void __wait(::cuda::std::size_t __count) const
  {
    for (::cuda::std::size_t __idx = 0; __idx < __count; ++__idx)
    {
      _CCCL_TRY_RUNTIME_API(::cudaStreamSynchronize, "waiting for sort merge", __streams_[__idx]);
    }
  }
};

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
        ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaMemcpyAsync(
          __result + __count * __peer, __source, __count * sizeof(_Tp), ::cudaMemcpyDeviceToDevice, __stream.get()));
      }
      else
      {
        __comm.send(__guard, __source, __count, __peer, __stream);
        __comm.recv(__guard, __result + __count * __peer, __count, __peer, __stream);
      }
    }
  }
}

template <class _Key, class _Compare, class _Env>
_CCCL_HOST_API void
__local_sort(_Key* __keys, _Key* __sorted_output, ::cuda::std::int64_t __count, _Compare __cmp, const _Env& __env)
{
  if (__count == 0)
  {
    return;
  }
  if constexpr (::cuda::std::is_arithmetic_v<_Key>
                && (::cuda::std::is_same_v<_Compare, ::cuda::std::less<>>
                    || ::cuda::std::is_same_v<_Compare, ::cuda::std::less<_Key>>) )
  {
    ::cuda::experimental::mgmn::__detail::__sort::__check(
      CUB_NS_QUALIFIER::DeviceRadixSort::SortKeys(__keys, __sorted_output, __count, 0, sizeof(_Key) * 8, __env));
  }
  else if constexpr (::cuda::std::is_arithmetic_v<_Key>
                     && (::cuda::std::is_same_v<_Compare, ::cuda::std::greater<>>
                         || ::cuda::std::is_same_v<_Compare, ::cuda::std::greater<_Key>>) )
  {
    ::cuda::experimental::mgmn::__detail::__sort::__check(CUB_NS_QUALIFIER::DeviceRadixSort::SortKeysDescending(
      __keys, __sorted_output, __count, 0, sizeof(_Key) * 8, __env));
  }
  else
  {
    ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaMemcpyAsync(
      __sorted_output, __keys, __count * sizeof(_Key), ::cudaMemcpyDeviceToDevice, ::cuda::get_stream(__env).get()));
    ::cuda::experimental::mgmn::__detail::__sort::__check(
      CUB_NS_QUALIFIER::DeviceMergeSort::SortKeys(__sorted_output, __count, __cmp, __env));
  }
}

template <class _Env>
class __scratch_allocator
{
  const _Env& __env_;
  ::cuda::stream_ref __stream_;
  ::cuda::experimental::mgmn::__detail::__resource_type_for<_Env> __resource_;

public:
  _CCCL_HOST_API __scratch_allocator(const _Env& __env, ::cuda::stream_ref __stream)
      : __env_(__env)
      , __stream_(__stream)
      , __resource_(::cuda::experimental::mgmn::__detail::__resource_from_env(__env, __stream.__logical_device()))
  {}
  template <class _Tp>
  [[nodiscard]] _CCCL_HOST_API auto operator()(::cuda::std::size_t __count) const
  {
    // Sixteen-key offsets preserve the transfer alignment requested here.
    const auto __requested = ::cuda::__call_or(::cuda::allocation_alignment, alignof(_Tp), __env_);
    ::cuda::__validate_allocation_alignment(__requested, alignof(_Tp));
    using _Alignment        = ::cuda::std::execution::prop<::cuda::allocation_alignment_t, ::cuda::std::size_t>;
    const auto __buffer_env = ::cuda::std::execution::env<_Alignment, const _Env&>{
      _Alignment{::cuda::allocation_alignment, ::cuda::std::max(::cuda::std::size_t{128}, __requested)}, __env_};
    return ::cuda::experimental::mgmn::__detail::__make_safe_uninitialized_buffer<_Tp>(
      __stream_, __resource_, __count, __buffer_env);
  }
};

struct __run
{
  ::cuda::std::int64_t __offset_, __count_;
};

template <class _Key, class _Compare, class _Env>
_CCCL_HOST_API void __merge_received(
  _Key* __received,
  ::cuda::std::int64_t __size,
  ::std::vector<__run> __runs,
  _Key* __keys,
  _Compare __cmp,
  const _Env& __env)
{
  const auto __stream = ::cuda::get_stream(__env);
  if (__runs.size() <= 1)
  {
    if (__size)
    {
      ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaMemcpyAsync(
        __keys, __received + __runs[0].__offset_, __size * sizeof(_Key), ::cudaMemcpyDeviceToDevice, __stream.get()));
    }
    return;
  }
  if (__runs.size() == 2)
  {
    const auto __a = __runs[0], __b = __runs[1];
    ::cuda::experimental::mgmn::__detail::__sort::__check(CUB_NS_QUALIFIER::DeviceMerge::MergeKeys(
      __received + __a.__offset_, __a.__count_, __received + __b.__offset_, __b.__count_, __keys, __cmp, __env));
    return;
  }
  const __scratch_allocator<_Env> __allocate{__env, __stream};
  auto __scratch       = __allocate.template operator()<_Key>(::cuda::std::max<::cuda::std::int64_t>(__size, 1));
  _Key* __merge_input  = __received;
  _Key* __merge_output = __scratch.data();
  if (__runs.size() > 1)
  {
    // The independent merge streams must wait for the caller-stream allocation.
    ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaStreamSynchronize(__stream.get()));
  }
  __merge_streams __streams{__runs.size() / 2};
  while (__runs.size() > 1)
  {
    // The final merge can write directly to the caller's disjoint output.
    // Earlier rounds alternate between the receive buffer and scratch.
    if (__runs.size() == 2)
    {
      __merge_output = __keys;
    }
    ::std::vector<__run> __next;
    ::cuda::std::int64_t __sorted_output_offset = 0;
    const ::cuda::std::size_t __pairs           = __runs.size() / 2;
    for (::cuda::std::size_t __pair = 0; __pair < __pairs; ++__pair)
    {
      const auto __a = __runs[2 * __pair], __b = __runs[2 * __pair + 1];
      // Override only the stream; preserve resource, tuning and all other
      // CUB environment queries. Each concurrent merge owns its scratch.
      auto __merge_env =
        ::cuda::std::execution::env<::cuda::stream_ref, const _Env&>{::cuda::stream_ref{__streams[__pair]}, __env};
      ::cuda::experimental::mgmn::__detail::__sort::__check(CUB_NS_QUALIFIER::DeviceMerge::MergeKeys(
        __merge_input + __a.__offset_,
        __a.__count_,
        __merge_input + __b.__offset_,
        __b.__count_,
        __merge_output + __sorted_output_offset,
        __cmp,
        __merge_env));
      __next.push_back({__sorted_output_offset, __a.__count_ + __b.__count_});
      __sorted_output_offset += __a.__count_ + __b.__count_;
    }
    if (__runs.size() % 2)
    {
      const auto __tail = __runs.back();
      ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaMemcpyAsync(
        __merge_output + __sorted_output_offset,
        __merge_input + __tail.__offset_,
        __tail.__count_ * sizeof(_Key),
        ::cudaMemcpyDeviceToDevice,
        __streams[0]));
      __next.push_back({__sorted_output_offset, __tail.__count_});
    }
    __streams.__wait(__pairs);
    __runs.swap(__next);
    ::cuda::std::swap(__merge_input, __merge_output);
  }
  // Every merge has completed before the temporary inputs are released.
  // The final round already wrote the caller's output, so no copy is needed.
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
  auto& __owner      = __persistent->__window();
  int __actions      = 0;
  bool __grow        = false;
  int __blocks       = __gin ? __transfer_blocks : __lsa_transfer_blocks;
  for (int __peer = 0; __peer < __ranks; ++__peer)
  {
    const auto __flags = __metadata[__device_metadata_width * __peer + 1];
    if (!(__flags & __device_supported))
    {
      _CCCL_THROW(::std::runtime_error, "Distributed sort requires NCCL device API support and GIN for non-LSA peers");
    }
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
    ::cuda::experimental::mgmn::__detail::__sort::
      __reset_gin_signals<<<__window.__device_.ginContextCount, 256, 0, __stream.get()>>>(__window.__device_);
    ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaGetLastError());
    // Every rank must finish its reset before any rank can send a signal.
    // These buffers are needed only when a communicator is newly created.
    auto __capability_source = __allocate.template operator()<int>(1);
    auto __capability_result = __allocate.template operator()<int>(__ranks);
    ::cuda::experimental::mgmn::__detail::__sort::__check(
      ::cudaMemsetAsync(__capability_source.data(), 0, sizeof(int), __stream.get()));
    ::cuda::experimental::mgmn::__detail::__sort::__all_gather(
      __comm, __capability_source.data(), __capability_result.data(), 1, __stream);
    ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaStreamSynchronize(__stream.get()));
  }
  auto __states    = __allocate.template operator()<__selection_state<_Key>>(__width);
  auto* __sorted   = __window.template __at<_Key>(__layout.__sorted_);
  auto* __received = __window.template __at<_Key>(__layout.__received_);
  ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaMemcpyAsync(
    __window.template __at<__index_type>(__layout.__targets_),
    __targets.data(),
    __width * sizeof(__index_type),
    ::cudaMemcpyHostToDevice,
    __stream.get()));
  ::cuda::experimental::mgmn::__detail::__sort::__local_sort(__keys, __sorted, __size, __cmp, __env);
  if (__gin)
  {
    ::cuda::experimental::mgmn::__detail::__sort::__device_select<true, _Key><<<1, 256, 0, __stream.get()>>>(
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
    ::cuda::experimental::mgmn::__detail::__sort::__device_select<false, _Key><<<1, 256, 0, __stream.get()>>>(
      __window.__device_,
      __window.__world_barrier_,
      __window.__window_,
      __layout,
      __size,
      __states.data(),
      __uniform,
      __cmp);
  }
  ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaGetLastError());
  __pinned_buffer<__index_type> __cuts{
    __width * __ranks, static_cast<__index_type*>(__window.__host_buffer(1, __width * __ranks * sizeof(__index_type)))};
  ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaMemcpyAsync(
    __cuts.__data(),
    __window.template __at<__index_type>(__layout.__cuts_),
    __width * __ranks * sizeof(__index_type),
    ::cudaMemcpyDeviceToHost,
    __stream.get()));
  if (__gin)
  {
    ::cuda::experimental::mgmn::__detail::__sort::__device_redistribute<true, _Key>
      <<<__blocks, 256, 0, __stream.get()>>>(
        __window.__device_, __window.__world_barrier_, __window.__window_, __layout);
  }
  else if (::cuda::experimental::mgmn::__detail::__sort::__use_rotated_redistribution<_Key>(__size))
  {
    ::cuda::experimental::mgmn::__detail::__sort::__device_redistribute_rotated<_Key>
      <<<__blocks, 256, 0, __stream.get()>>>(
        __window.__device_, __window.__world_barrier_, __window.__window_, __layout);
  }
  else
  {
    ::cuda::experimental::mgmn::__detail::__sort::__device_redistribute<false, _Key>
      <<<__blocks, 256, 0, __stream.get()>>>(
        __window.__device_, __window.__world_barrier_, __window.__window_, __layout);
  }
  ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaGetLastError());
  ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaStreamSynchronize(__stream.get()));
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
  ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaStreamSynchronize(__stream.get()));
}
#endif // _CUDAX_SORT_HAS_NCCL_DEVICE

template <class _Comm, class _Env, class _Key, class _Size, class _Compare>
_CCCL_HOST_API void __sort_one(_Comm& __comm, const _Env& __env, _Key* __keys, _Size __input_count, _Compare __cmp)
{
#if defined(_CUDAX_SORT_HAS_NCCL_DEVICE)
  ::std::unique_lock<::std::mutex> __context_lock;
  auto* __persistent = ::cuda::__call_or(
    ::cuda::experimental::mgmn::__sort_context_query{},
    static_cast<::cuda::experimental::mgmn::sort_context*>(nullptr),
    __env);
  if (__persistent)
  {
    // Serialize the metadata collectives too, not just workspace accesses.
    __context_lock = __persistent->__lock();
  }
#endif
  static_assert(::cuda::std::is_integral_v<_Size>, "sort count must be an integer");
  const ::cuda::stream_ref __stream = ::cuda::get_stream(__env);
  const auto __context              = ::cuda::__ensure_current_context{__stream};
  const int __rank = __comm.rank(), __ranks = __comm.size();
  if (__ranks <= 0 || __rank < 0 || __rank >= __ranks)
  {
    _CCCL_THROW(::std::invalid_argument, "Invalid communicator rank/size");
  }
  const __scratch_allocator<_Env> __allocate{__env, __stream};

  // Agree on the caller's fixed output counts before selection. Invalid local
  // counts are exchanged too, so a negative count cannot strand other ranks.
  const ::cuda::std::uint64_t __limit =
    static_cast<::cuda::std::uint64_t>(::cuda::std::numeric_limits<::cuda::std::int64_t>::max()) / sizeof(_Key);
  const ::cuda::std::uint64_t __padding = ::cuda::std::uint64_t{30} * __ranks;
  const bool __invalid =
    (::cuda::std::is_unsigned_v<_Size> ? false : (__input_count < 0))
    || static_cast<::cuda::std::uint64_t>(__input_count) > __limit - __padding
    || (__input_count != 0 && __keys == nullptr);
  ::cuda::std::size_t __metadata_width = 1;
#if defined(_CUDAX_SORT_HAS_NCCL_DEVICE)
  bool __gin = false;
  if constexpr (::cuda::std::is_same_v<decltype(__comm.native_handle()), ::ncclComm_t>)
  {
    if (__ranks > 1)
    {
      __metadata_width = __device_metadata_width;
    }
  }
#endif
  const auto __metadata_count = __metadata_width * (static_cast<::cuda::std::size_t>(__ranks) + 1);
  long long* __cached_counts  = nullptr;
#if defined(_CUDAX_SORT_HAS_NCCL_DEVICE)
  if (__metadata_width == __device_metadata_width && __persistent && __persistent->__window())
  {
    __cached_counts =
      static_cast<long long*>(__persistent->__window()->__host_buffer(0, __metadata_count * sizeof(long long)));
  }
#endif
  __pinned_buffer<long long> __counts{__metadata_count, __cached_counts};
  auto* __local_metadata = __counts.__data() + __metadata_width * __ranks;
  __local_metadata[0]    = __invalid ? -1 : static_cast<long long>(__input_count);
#if defined(_CUDAX_SORT_HAS_NCCL_DEVICE)
  if constexpr (::cuda::std::is_same_v<decltype(__comm.native_handle()), ::ncclComm_t>)
  {
    if (__ranks > 1)
    {
      ::ncclCommProperties __properties = NCCL_COMM_PROPERTIES_INITIALIZER;
      const auto __property_status      = ::ncclCommQueryProperties(__comm.native_handle(), &__properties);
      __gin                             = ::ncclTeamLsa(__comm.native_handle()).nRanks != __ranks;
      int __ordinal                     = 0;
      ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaGetDevice(&__ordinal));
      // Cross-rank CTA barriers require a grid that can reside on every GPU.
      // Query the compiled specialization; the collective takes the rank minimum.
      int __active_blocks = 0, __multiprocessors = 0;
      // Ragged ranks may choose different LSA kernels. Query the exact local
      // choice before agreeing on a common resident grid and barrier slots.
      const bool __rotate = !__invalid && !__gin
                         && ::cuda::experimental::mgmn::__detail::__sort::__use_rotated_redistribution<_Key>(
                           static_cast<::cuda::std::uint64_t>(__input_count));
      const auto __kernel =
        __gin      ? ::cuda::experimental::mgmn::__detail::__sort::__device_redistribute<true, _Key>
        : __rotate ? ::cuda::experimental::mgmn::__detail::__sort::__device_redistribute_rotated<_Key>
                   : ::cuda::experimental::mgmn::__detail::__sort::__device_redistribute<false, _Key>;
      const auto __occupancy_status =
        ::cudaOccupancyMaxActiveBlocksPerMultiprocessor(&__active_blocks, __kernel, 256, 0);
      const auto __device_status =
        ::cudaDeviceGetAttribute(&__multiprocessors, ::cudaDevAttrMultiProcessorCount, __ordinal);
      const bool __resident = __occupancy_status == ::cudaSuccess && __device_status == ::cudaSuccess
                           && __active_blocks > 0 && __multiprocessors > 0;
      const int __blocks =
        __resident
          ? ::cuda::std::min(__gin ? __transfer_blocks : __lsa_transfer_blocks, __active_blocks * __multiprocessors)
          : 0;
      __local_metadata[3] = __blocks;
      const auto* __owner = __persistent ? __persistent->__window().get() : nullptr;
      // Send failures as flags so every rank first participates in the count
      // agreement. Empty inputs still return without creating a device context.
      __local_metadata[1] =
        (__resident && __property_status == ::ncclSuccess && __properties.deviceApiSupport
             && (!__gin || __properties.ginType != NCCL_GIN_TYPE_NONE)
           ? __device_supported
           : 0)
        | (__owner && !__owner->__matches(__comm.native_handle(), __ordinal, __gin) ? __device_wrong_context : 0)
        | (!__owner ? __device_create : 0);
      __local_metadata[2] =
        ::cuda::std::bit_cast<long long>(static_cast<::cuda::std::uint64_t>(__owner ? __owner->__capacity_ : 0));
    }
  }
#endif
  auto __source_count  = __allocate.template operator()<long long>(__metadata_width);
  auto __device_counts = __allocate.template operator()<long long>(__metadata_width * __ranks);
  ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaMemcpyAsync(
    __source_count.data(),
    __local_metadata,
    __metadata_width * sizeof(long long),
    ::cudaMemcpyHostToDevice,
    __stream.get()));
  ::cuda::experimental::mgmn::__detail::__sort::__all_gather(
    __comm, __source_count.data(), __device_counts.data(), __metadata_width, __stream);
  ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaMemcpyAsync(
    __counts.__data(),
    __device_counts.data(),
    __metadata_width * __ranks * sizeof(long long),
    ::cudaMemcpyDeviceToHost,
    __stream.get()));
  ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaStreamSynchronize(__stream.get()));
  ::std::vector<::cuda::std::int64_t> __targets(__ranks + 1, 0);
  bool __uniform = true;
  for (int __r = 0; __r < __ranks; ++__r)
  {
    const auto __count = __counts[__metadata_width * __r];
    if (__count < 0 || __count > ::cuda::std::numeric_limits<::cuda::std::int64_t>::max() - __targets[__r])
    {
      _CCCL_THROW(::std::invalid_argument, "Invalid distributed sort input count or total size overflow");
    }
    __targets[__r + 1] = __targets[__r] + __count;
    __uniform &= __count == __counts[0];
  }
  if (__targets.back() == 0)
  {
    return;
  }
  const ::cuda::std::int64_t __size = __counts[__metadata_width * __rank];
#if defined(_CUDAX_SORT_HAS_NCCL_DEVICE)
  if constexpr (::cuda::std::is_same_v<decltype(__comm.native_handle()), ::ncclComm_t>)
  {
    if (__ranks > 1)
    {
      ::cuda::experimental::mgmn::__detail::__sort::__sort_device(
        __comm, __env, __keys, __size, __targets, __uniform, __counts.__data(), __gin, __cmp);
      return;
    }
  }
#endif
  auto __sorted = __allocate.template operator()<_Key>(::cuda::std::max<::cuda::std::int64_t>(__size, 1));
  ::cuda::experimental::mgmn::__detail::__sort::__local_sort(__keys, __sorted.data(), __size, __cmp, __env);
  if (__ranks == 1)
  {
    ::cuda::experimental::mgmn::__detail::__sort::__check(
      ::cudaMemcpyAsync(__keys, __sorted.data(), __size * sizeof(_Key), ::cudaMemcpyDeviceToDevice, __stream.get()));
    return;
  }

  const ::cuda::std::size_t __width = static_cast<::cuda::std::size_t>(__ranks) + 1;
  auto __source                     = __allocate.template operator()<__selection_message<_Key>>(__width);
  auto __gathered                   = __allocate.template operator()<__selection_message<_Key>>(__width * __ranks);
  auto __states                     = __allocate.template operator()<__selection_state<_Key>>(__ranks + 1);
  auto __device_targets             = __allocate.template operator()<::cuda::std::int64_t>(__ranks + 1);
  auto __unfinished                 = __allocate.template operator()<int>(1);
  __pinned_buffer<int> __host_unfinished{1};
  __pinned_buffer<::cuda::std::int64_t> __cuts{static_cast<::cuda::std::size_t>(__ranks + 1) * __ranks};
  ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaMemcpyAsync(
    __device_targets.data(),
    __targets.data(),
    (__ranks + 1) * sizeof(::cuda::std::int64_t),
    ::cudaMemcpyHostToDevice,
    __stream.get()));
  auto __local_cuts    = __allocate.template operator()<::cuda::std::int64_t>(__width);
  auto __gathered_cuts = __allocate.template operator()<::cuda::std::int64_t>(__width * __ranks);
  auto __step          = [&](auto __phase) {
    ::cuda::experimental::mgmn::__detail::__sort::__select<decltype(__phase)::value><<<1, 256, 0, __stream.get()>>>(
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
    ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaGetLastError());
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
    ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaMemcpyAsync(
      __host_unfinished.__data(), __unfinished.data(), sizeof(int), ::cudaMemcpyDeviceToHost, __stream.get()));
    ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaStreamSynchronize(__stream.get()));
  } while (__host_unfinished[0]);
  __step(::cuda::std::integral_constant<__phase, __phase::__export_cuts>{});
  ::cuda::experimental::mgmn::__detail::__sort::__all_gather(
    __comm, __local_cuts.data(), __gathered_cuts.data(), __ranks + 1, __stream);
  ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaMemcpyAsync(
    __cuts.__data(),
    __gathered_cuts.data(),
    ::cuda::std::size_t(__ranks + 1) * __ranks * sizeof(long long),
    ::cudaMemcpyDeviceToHost,
    __stream.get()));
  ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaStreamSynchronize(__stream.get()));
  auto __cut = [&](int __boundary, int __peer) -> ::cuda::std::int64_t {
    return __cuts[::cuda::std::size_t(__peer) * (__ranks + 1) + __boundary];
  };

  auto __received = __allocate.template operator()<_Key>(__size + __padding);
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
          ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaMemcpyAsync(
            __received.data() + __offset,
            __sorted.data() + __begin,
            __count * sizeof(_Key),
            ::cudaMemcpyDeviceToDevice,
            __stream.get()));
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
  ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaStreamSynchronize(__stream.get()));
  ::cuda::experimental::mgmn::__detail::__sort::__merge_received(
    __received.data(), __size, ::cuda::std::move(__runs), __keys, __cmp, __env);
}
template <class _CommRange, class _EnvRange, class _InputRange, class _SizeRange, class _Compare = ::cuda::std::less<>>
_CCCL_HOST_API void __execute_range(
  _CommRange&& __comms, _EnvRange&& __envs, _InputRange&& __inputs, _SizeRange&& __sizes, _Compare __cmp = {})
{
  const auto __count = ::cuda::std::ranges::distance(__comms);
  if (__count == 0)
  {
    return;
  }
  if (::cuda::std::ranges::distance(__envs) != __count || ::cuda::std::ranges::distance(__inputs) != __count
      || ::cuda::std::ranges::distance(__sizes) != __count)
  {
    _CCCL_THROW(::std::invalid_argument, "Distributed sort ranges must have the same length");
  }
#if defined(_CUDAX_SORT_HAS_NCCL_DEVICE)
  if (__count > 1)
  {
    ::std::vector<::cuda::experimental::mgmn::sort_context*> __contexts;
    for (const auto& __env : __envs)
    {
      auto* __context = ::cuda::__call_or(
        ::cuda::experimental::mgmn::__sort_context_query{},
        static_cast<::cuda::experimental::mgmn::sort_context*>(nullptr),
        __env);
      if (__context)
      {
        for (auto* __previous : __contexts)
        {
          if (__previous == __context)
          {
            _CCCL_THROW(::std::invalid_argument, "Each local sort rank requires a distinct context");
          }
        }
        __contexts.push_back(__context);
      }
    }
  }
#endif
  auto __c = ::cuda::std::ranges::begin(__comms);
  auto __e = ::cuda::std::ranges::begin(__envs);
  auto __i = ::cuda::std::ranges::begin(__inputs);
  auto __n = ::cuda::std::ranges::begin(__sizes);
  if (__count == 1)
  {
    ::cuda::experimental::mgmn::__detail::__sort::__sort_one(*__c, *__e, ::cuda::std::to_address(*__i), *__n, __cmp);
    return;
  }
  // Blocking metadata decisions require concurrent participation by local
  // ranks. Each worker has independent state and selects its stream's device.
  ::std::vector<::std::exception_ptr> __errors(__count);
  ::std::vector<::std::thread> __workers;
  __workers.reserve(__count);
  ::std::mutex __mutex;
  ::std::condition_variable __ready;
  int __start = 0;
  _CCCL_TRY
  {
    for (::cuda::std::remove_cv_t<decltype(__count)> __r = 0; __r < __count; ++__r, ++__c, ++__e, ++__i, ++__n)
    {
      __workers.emplace_back([&, __c, __e, __i, __n, __r, __cmp] {
        {
          ::std::unique_lock<::std::mutex> __lock{__mutex};
          __ready.wait(__lock, [&] {
            return __start != 0;
          });
          if (__start == 2)
          {
            return;
          }
        }
        _CCCL_TRY
        {
          ::cuda::experimental::mgmn::__detail::__sort::__sort_one(
            *__c, *__e, ::cuda::std::to_address(*__i), *__n, __cmp);
        }
        _CCCL_CATCH_ALL
        {
          __errors[__r] = ::std::current_exception();
        }
      });
    }
  }
  _CCCL_CATCH_ALL
  {
    // Do not leave already-created workers inside a collective when a
    // later host thread cannot be started.
    {
      const ::std::lock_guard<::std::mutex> __lock{__mutex};
      __start = 2;
    }
    __ready.notify_all();
    for (auto& __worker : __workers)
    {
      __worker.join();
    }
    _CCCL_RETHROW;
  }
  {
    const ::std::lock_guard<::std::mutex> __lock{__mutex};
    __start = 1;
  }
  __ready.notify_all();
  for (auto& __worker : __workers)
  {
    __worker.join();
  }
  for (auto __error : __errors)
  {
    if (__error)
    {
      ::std::rethrow_exception(__error);
    }
  }
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT
} // namespace __detail::__sort
_CCCL_END_NAMESPACE_CUDA_MGMN
// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>
#endif // _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_EXECUTE_H
