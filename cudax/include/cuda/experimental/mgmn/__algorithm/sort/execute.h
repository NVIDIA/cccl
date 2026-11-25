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

#include <cuda/__container/buffer.h>
#include <cuda/__driver/driver_api.h>
#include <cuda/__memory_resource/legacy_pinned_memory_resource.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__exception/cuda_error.h>
#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/__iterator/distance.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/span>

#include <cuda/experimental/mgmn/__algorithm/sort/exchange.h>

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
  ::cuda::host_buffer<long long> __counts_storage{
    __stream, ::cuda::mr::legacy_pinned_memory_resource{__stream.device()}};
  if (!__cached_counts)
  {
    __counts_storage = ::cuda::host_buffer<long long>{
      __stream, ::cuda::mr::legacy_pinned_memory_resource{__stream.device()}, __metadata_count, ::cuda::no_init};
  }
  ::cuda::std::span<long long> __counts{__cached_counts ? __cached_counts : __counts_storage.data(), __metadata_count};
  auto* __local_metadata = __counts.data() + __metadata_width * __ranks;
  __local_metadata[0]    = __invalid ? -1 : static_cast<long long>(__input_count);
#if defined(_CUDAX_SORT_HAS_NCCL_DEVICE)
  if constexpr (::cuda::std::is_same_v<decltype(__comm.native_handle()), ::ncclComm_t>)
  {
    if (__ranks > 1)
    {
      ::ncclCommProperties __properties = NCCL_COMM_PROPERTIES_INITIALIZER;
      const auto& __api                 = ::cuda::experimental::__nccl::__get_device_api();
      const auto __property_status =
        __api.__supports_lsa() ? __api.__query(__comm.native_handle(), &__properties) : ::ncclInvalidUsage;
      const bool __device_available = __property_status == ::ncclSuccess && __properties.deviceApiSupport;
      if (__device_available)
      {
        __gin = __api.__lsa(__comm.native_handle()).nRanks != __ranks;
      }
      const auto __ordinal = ::cuda::__driver::__ctxGetDevice();
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
      bool __resident = false;
      if (__device_available)
      {
        _CCCL_TRY
        {
          const auto __function = ::cuda::__get_cufunction_of(reinterpret_cast<const void*>(__kernel));
          const auto __occupancy_status =
            ::cuda::__driver::__occupancyMaxActiveBlocksPerMultiprocessorNoThrow(__active_blocks, __function, 256, 0);
          const auto __device_status = ::cuda::__driver::__deviceGetAttributeNoThrow(
            __multiprocessors, ::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, __ordinal);
          __resident = __occupancy_status.__ok() && __device_status.__ok() && __active_blocks > 0
                    && __multiprocessors > 0;
        }
        _CCCL_CATCH (const ::cuda::cuda_error& __error)
        {
          (void) __error;
          // Kernel availability is optional. Agree on the host backend instead
          // of throwing before the other ranks enter the metadata collective.
        }
        _CCCL_CATCH_FALLTHROUGH
      }
      const int __blocks =
        __resident
          ? ::cuda::std::min(__gin ? __transfer_blocks : __lsa_transfer_blocks, __active_blocks * __multiprocessors)
          : 0;
      __local_metadata[3] = __blocks;
      const auto* __owner = __persistent ? __persistent->__window().get() : nullptr;
      // Send failures as flags so every rank first participates in the count
      // agreement. Empty inputs still return without creating a device context.
      __local_metadata[1] =
        (__resident && __device_available
             && (!__gin || (__api.__supports_gin() && __properties.ginType != NCCL_GIN_TYPE_NONE))
           ? __device_supported
           : 0)
        | (__owner
               && !__owner->__matches(__comm.native_handle(), __ordinal, __device_available ? __gin : __owner->__gin_)
             ? __device_wrong_context
             : 0)
        | (!__owner ? __device_create : 0);
      __local_metadata[2] =
        ::cuda::std::bit_cast<long long>(static_cast<::cuda::std::uint64_t>(__owner ? __owner->__capacity_ : 0));
    }
  }
#endif
  auto __source_count  = __allocate.template operator()<long long>(__metadata_width);
  auto __device_counts = __allocate.template operator()<long long>(__metadata_width * __ranks);
  ::cuda::__driver::__memcpyAsync(
    __source_count.data(), __local_metadata, __metadata_width * sizeof(long long), __stream.get());
  ::cuda::experimental::mgmn::__detail::__sort::__all_gather(
    __comm, __source_count.data(), __device_counts.data(), __metadata_width, __stream);
  ::cuda::__driver::__memcpyAsync(
    __counts.data(), __device_counts.data(), __metadata_width * __ranks * sizeof(long long), __stream.get());
  ::cuda::__driver::__streamSynchronize(__stream.get());
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
      bool __use_device = true;
      int __actions     = 0;
      for (int __peer = 0; __peer < __ranks; ++__peer)
      {
        const auto __flags = __counts[__device_metadata_width * __peer + 1];
        __use_device &= (__flags & __device_supported) != 0;
        __actions |= static_cast<int>(__flags);
      }
      if (__actions & __device_wrong_context)
      {
        _CCCL_THROW(::std::invalid_argument, "Sort context belongs to a different communicator or device");
      }
      if (__persistent && __persistent->__window())
      {
        __persistent->__window()->__set_stream(__stream.get());
      }
      // Every rank makes the same choice before creating any device resources.
      // New headers do not imply that this communicator supports the device API.
      if (__use_device)
      {
        ::cuda::experimental::mgmn::__detail::__sort::__sort_device(
          __comm, __env, __keys, __size, __targets, __uniform, __counts.data(), __gin, __cmp);
        return;
      }
    }
  }
#endif
  ::cuda::experimental::mgmn::__detail::__sort::__sort_host(__comm, __env, __keys, __size, __targets, __uniform, __cmp);
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
