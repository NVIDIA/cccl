//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_RESOURCE_LOCALITY_DOMAIN_MEMORY_POOL_H
#define _CUDA___MEMORY_RESOURCE_LOCALITY_DOMAIN_MEMORY_POOL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()

#  include <cuda/__device/logical_device_ref.h>
#  include <cuda/__device/physical_device.h>
#  include <cuda/__memory_pool/device_memory_pool.h>
#  include <cuda/__utility/call_once.h>
#  include <cuda/__utility/raw_storage.h>
#  include <cuda/std/__memory/unique_ptr.h>
#  include <cuda/std/__utility/move.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// A single `once_flag` per device rather than one per domain. Whoever needs the pool of domain X on
// device Y almost always needs the pools of the other domains of Y as well, and the green contexts
// those pools sit behind are built for every domain at once by `device_ref::__locality_domains()`.
// Building all of them together therefore costs one extra driver call per domain, and saves a flag
// and a level of indirection per pool.
class __per_device_locality_pools
{
  __once_flag __once_{};
  __raw_storage_array<device_memory_pool_ref> __pools_{};

  static_assert(::cuda::std::is_trivially_destructible_v<device_memory_pool_ref>);

  ::cuda::std::size_t __num_domains_{};

  _CCCL_HOST_API void __init(::cuda::device_ref __device)
  {
    const auto __num_domains = __device.__locality_domains().size();

    // `device_memory_pool_ref` is not default constructible, so the array is built element by
    // element in raw storage.
    auto __tmp = ::cuda::__make_raw_storage_array<device_memory_pool_ref>(__num_domains);

    for (auto&& __domain : __device.__locality_domains())
    {
      const auto [__domain_id, __localized] = __domain.locality_domain();
      auto __location                       = ::CUmemLocation{};

      // We got here because the outer logical_device was localized, but now one of its sibling
      // locality domain is not? Something has gone wrong.
      _CCCL_VERIFY(__localized, "Non-localized locality domain in a localized-only path, we should never get here.");

#  if _CCCL_CTK_AT_LEAST(13, 4)
      __location.type                       = ::CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
      __location.localized.deviceId         = __device.get();
      __location.localized.localityDomainId = static_cast<unsigned int>(__domain_id);
#  else // ^^^ 13.4+ ^^^ / vvv 13.3- vvv
      _CCCL_VERIFY(false, "We should have taken the full-device memory pool path earlier.");
#  endif // ^^^ 13.3- ^^^

      ::cuda::std::__construct_at(
        __tmp.get() + __domain_id, ::cuda::__get_default_memory_pool(__location, ::CU_MEM_ALLOCATION_TYPE_PINNED));
      // This only works if __locality_domains() iterates with monotonically increasing domain
      // ID without holes. Otherwise we will need a more complicated strategy for remember
      // which domains to delete.
      __tmp.get_deleter().__count_ = __domain_id + 1;
    }

    // Commit only once every step succeeds, so a throw leaves this object as it was.
    __pools_       = ::cuda::std::move(__tmp);
    __num_domains_ = __num_domains;
  }

public:
  [[nodiscard]] _CCCL_HOST_API device_memory_pool_ref& __get(::cuda::device_ref __device, ::cuda::std::size_t __domain)
  {
    ::cuda::__call_once(__once_, [this, __device] {
      this->__init(__device);
    });

    _CCCL_ASSERT(__domain < __num_domains_, "locality domain id out of range");
    return __pools_[__domain];
  }
};

[[nodiscard]] _CCCL_HOST_API inline device_memory_pool_ref& __device_default_memory_pool(__logical_device_ref __device)
{
  const auto __underlying = __device.underlying_device();
  const auto __domain     = __device.locality_domain();

  if (!__domain.localized)
  {
    // Unlocalized logical devices just use the whole device
    return ::cuda::device_default_memory_pool(__underlying);
  }

  static const auto __pools_ =
    ::cuda::std::make_unique<__per_device_locality_pools[]>(::cuda::__physical_devices_count());

  return __pools_[static_cast<::cuda::std::size_t>(__underlying.get())].__get(__underlying, __domain.domain_id);
}

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif //_CUDA___MEMORY_RESOURCE_LOCALITY_DOMAIN_MEMORY_POOL_H
