//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_SUPPORT_LOGICAL_ENDPOINT_TEST_HELPER_H
#define TEST_SUPPORT_LOGICAL_ENDPOINT_TEST_HELPER_H

#include "test_macros.h"

#if _CCCL_CTK_AT_LEAST(13, 3)

#  include <cuda/atomic>
#  include <cuda/devices>
#  include <cuda/logical_endpoint>
#  include <cuda/memory_pool>
#  include <cuda/ptx>
#  include <cuda/std/chrono>
#  include <cuda/std/cstdint>

#  include <cuda_runtime_api.h>

namespace logical_endpoint_test
{
inline constexpr int cuda_driver_version_13_3                  = 13030;
inline constexpr cuda::std::uint64_t minimum_bytes             = 4096;
inline constexpr auto ready_timeout                            = cuda::std::chrono::seconds{1};
inline constexpr cuda::std::uint32_t payload_words             = 4;
inline constexpr cuda::std::uint32_t payload_bytes             = payload_words * sizeof(cuda::std::uint32_t);
inline constexpr cuda::std::uint32_t tx_granularity            = 16;
inline constexpr cuda::std::uint32_t status_success            = 1;
inline constexpr cuda::std::uint32_t status_unsupported        = 2;
inline constexpr cuda::std::uint32_t status_timeout            = 3;
inline constexpr cuda::std::uint64_t counted_counter_alignment = 256;
inline constexpr cuda::std::uint32_t ring_chunk_words          = 4;
inline constexpr cuda::std::uint32_t ring_chunk_bytes          = ring_chunk_words * sizeof(cuda::std::uint32_t);
inline constexpr cuda::std::uint32_t ring_chunk_count          = 2;
inline constexpr cuda::std::uint32_t ring_payload_words        = ring_chunk_words * ring_chunk_count;
inline constexpr cuda::std::uint32_t ring_payload_bytes        = ring_chunk_bytes * ring_chunk_count;
inline constexpr cuda::std::uint32_t ring_sync_bytes           = 16;
inline constexpr cuda::std::uint32_t ring_status_words         = ring_chunk_count + 1;
inline constexpr int wait_iterations                           = 1000000;

struct support_result
{
  bool supported{};
  const char* reason{};
  cuda::logical_endpoint_limits limits{};
};

[[nodiscard]] inline const char* runtime_unsupported_reason(int minimum_device_count = 1)
{
  int driver_version = 0;
  if (::cudaDriverGetVersion(&driver_version) != cudaSuccess)
  {
    return "CUDA driver version could not be queried";
  }
  if (driver_version < logical_endpoint_test::cuda_driver_version_13_3)
  {
    return "logical endpoints require a CUDA 13.3 driver";
  }

  int device_count = 0;
  if (minimum_device_count > 0
      && (::cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < minimum_device_count))
  {
    return "logical endpoint tests require a CUDA device";
  }
  return nullptr;
}

[[nodiscard]] inline cuda::std::uint64_t align_up(cuda::std::uint64_t value, cuda::std::uint64_t alignment)
{
  return ((value + alignment - 1) / alignment) * alignment;
}

[[nodiscard]] inline cuda::std::uint64_t endpoint_size(cuda::logical_endpoint_limits limits)
{
  return limits.bind_alignment == 0
         ? 0
         : logical_endpoint_test::align_up(logical_endpoint_test::minimum_bytes, limits.bind_alignment);
}

template <class Spec, class... Devices>
[[nodiscard]] support_result
probe_logical_endpoint_support(const Spec& spec, cuda::device_ref device, Devices... devices)
{
  if (const char* reason = logical_endpoint_test::runtime_unsupported_reason(0))
  {
    return {false, reason, {}};
  }

  const cuda::device_ref checked_devices[] = {device, devices...};
  for (const cuda::device_ref checked_device : checked_devices)
  {
    if (!spec.is_supported(checked_device))
    {
      return {false, "logical endpoints are not supported by the selected device", {}};
    }
  }

  return {true, nullptr, spec.limits()};
}

template <class... Devices>
[[nodiscard]] bool memory_pools_supported(cuda::device_ref device, Devices... devices)
{
  const cuda::device_ref checked_devices[] = {device, devices...};
  for (const cuda::device_ref checked_device : checked_devices)
  {
    if (!cuda::device_attributes::memory_pools_supported(checked_device))
    {
      return false;
    }
  }
  return true;
}

template <class... Devices>
[[nodiscard]] bool fabric_memory_pools_supported(cuda::device_ref device, Devices... devices)
{
  if (!logical_endpoint_test::memory_pools_supported(device, devices...))
  {
    return false;
  }

  const cuda::device_ref checked_devices[] = {device, devices...};
  for (const cuda::device_ref checked_device : checked_devices)
  {
    if ((cuda::device_attributes::memory_pool_supported_handle_types(checked_device) & cudaMemHandleTypeFabric) == 0)
    {
      return false;
    }
  }
  return true;
}

[[nodiscard]] inline cuda::memory_pool_properties fabric_memory_pool_properties()
{
  cuda::memory_pool_properties properties{};
  properties.allocation_handle_type = cudaMemHandleTypeFabric;
  return properties;
}

template <class... Devices>
[[nodiscard]] bool fabric_ptx_supported(cuda::device_ref device, Devices... devices)
{
#  if __cccl_ptx_isa >= 930
  const cuda::device_ref checked_devices[] = {device, devices...};
  for (const cuda::device_ref checked_device : checked_devices)
  {
    if (checked_device.attribute(cuda::device_attributes::compute_capability) < cuda::compute_capability{100})
    {
      return false;
    }
  }
  return true;
#  else // ^^^ __cccl_ptx_isa >= 930 ^^^ / vvv __cccl_ptx_isa < 930
  (void) device;
  ((void) devices, ...);
  return false;
#  endif // __cccl_ptx_isa < 930
}

#  if __cccl_ptx_isa >= 930
__device__ bool wait_for_mbarrier_completion(cuda::std::uint64_t* barrier)
{
  for (int iteration = 0; iteration < logical_endpoint_test::wait_iterations; ++iteration)
  {
    if (cuda::ptx::mbarrier_try_wait_parity(cuda::ptx::sem_acquire, cuda::ptx::scope_cta, barrier, 0))
    {
      return true;
    }
  }
  return false;
}

__device__ bool wait_for_remote_flag(cuda::std::uint32_t* flag)
{
  cuda::atomic_ref<cuda::std::uint32_t, cuda::thread_scope_system> ref(*flag);
  for (int iteration = 0; iteration < logical_endpoint_test::wait_iterations; ++iteration)
  {
    if (ref.load(cuda::memory_order_relaxed) != 0)
    {
      return true;
    }
  }
  return false;
}

__device__ bool wait_for_counter(cuda::std::uint64_t* counter, cuda::std::uint64_t expected)
{
  cuda::atomic_ref<cuda::std::uint64_t, cuda::thread_scope_system> ref(*counter);
  for (int iteration = 0; iteration < logical_endpoint_test::wait_iterations; ++iteration)
  {
    if (ref.load(cuda::memory_order_relaxed) == expected)
    {
      return true;
    }
  }
  return false;
}
#  endif // __cccl_ptx_isa >= 930

__global__ void fabric_try_put_smoke_kernel(
  cuda::unicast_logical_endpoint_ref endpoint, cuda::std::uint64_t endpoint_offset, cuda::std::uint32_t* status)
{
#  if __cccl_ptx_isa >= 930
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_100,
    (__shared__ alignas(16) cuda::std::uint32_t src_words[logical_endpoint_test::payload_words];
     __shared__ alignas(8) cuda::std::uint64_t barrier;

     if (threadIdx.x == 0) {
       src_words[0] = 0x13572468u;
       src_words[1] = 0x24681357u;
       src_words[2] = 0xdeadbeefu;
       src_words[3] = 0xcafef00du;

       cuda::ptx::mbarrier_init(&barrier, 1u);
       cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
       cuda::ptx::fabric_try_put(
         cuda::ptx::space_shared,
         cuda::ptx::sem_relaxed,
         cuda::ptx::scope_sys,
         endpoint.native_handle(),
         endpoint_offset,
         src_words,
         logical_endpoint_test::payload_bytes,
         &barrier);
       cuda::ptx::fabric_submit();
       cuda::ptx::mbarrier_arrive_expect_tx(
         cuda::ptx::sem_relaxed,
         cuda::ptx::scope_cta,
         cuda::ptx::space_shared,
         &barrier,
         logical_endpoint_test::payload_bytes / logical_endpoint_test::tx_granularity);

       for (int iteration = 0; iteration < 1000000; ++iteration)
       {
         if (cuda::ptx::mbarrier_try_wait_parity(cuda::ptx::sem_acquire, cuda::ptx::scope_cta, &barrier, 0))
         {
           *status = logical_endpoint_test::status_success;
           return;
         }
       }

       *status = logical_endpoint_test::status_timeout;
     }),
    (if (threadIdx.x == 0) { *status = logical_endpoint_test::status_unsupported; }))
#  else // ^^^ __cccl_ptx_isa >= 930 ^^^ / vvv __cccl_ptx_isa < 930
  if (threadIdx.x == 0)
  {
    *status = logical_endpoint_test::status_unsupported;
  }
#  endif // __cccl_ptx_isa < 930
}

__global__ void fabric_try_put_counted_smoke_kernel(
  cuda::unicast_logical_endpoint_ref endpoint,
  cuda::std::uint64_t endpoint_offset,
  cuda::std::uint64_t counter_offset,
  cuda::std::uint32_t* status)
{
#  if __cccl_ptx_isa >= 930
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_100,
    (__shared__ alignas(16) cuda::std::uint32_t src_words[logical_endpoint_test::payload_words];
     __shared__ alignas(8) cuda::std::uint64_t barrier;

     if (threadIdx.x == 0) {
       src_words[0] = 0x13572468u;
       src_words[1] = 0x24681357u;
       src_words[2] = 0xdeadbeefu;
       src_words[3] = 0xcafef00du;

       cuda::ptx::mbarrier_init(&barrier, 1u);
       cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
       cuda::ptx::fabric_try_put_counted(
         cuda::ptx::space_shared,
         cuda::ptx::sem_relaxed,
         cuda::ptx::scope_sys,
         endpoint.native_handle(),
         endpoint_offset,
         counter_offset,
         src_words,
         logical_endpoint_test::payload_bytes,
         &barrier);
       cuda::ptx::fabric_submit();
       cuda::ptx::mbarrier_arrive_expect_tx(
         cuda::ptx::sem_relaxed,
         cuda::ptx::scope_cta,
         cuda::ptx::space_shared,
         &barrier,
         logical_endpoint_test::payload_bytes / logical_endpoint_test::tx_granularity);

       for (int iteration = 0; iteration < 1000000; ++iteration)
       {
         if (cuda::ptx::mbarrier_try_wait_parity(cuda::ptx::sem_acquire, cuda::ptx::scope_cta, &barrier, 0))
         {
           *status = logical_endpoint_test::status_success;
           return;
         }
       }

       *status = logical_endpoint_test::status_timeout;
     }),
    (if (threadIdx.x == 0) { *status = logical_endpoint_test::status_unsupported; }))
#  else // ^^^ __cccl_ptx_isa >= 930 ^^^ / vvv __cccl_ptx_isa < 930
  if (threadIdx.x == 0)
  {
    *status = logical_endpoint_test::status_unsupported;
  }
#  endif // __cccl_ptx_isa < 930
}

__global__ void fabric_ring_put_kernel(
  cuda::unicast_logical_endpoint_ref endpoint, cuda::std::uint32_t rank, cuda::std::uint32_t* status)
{
#  if __cccl_ptx_isa >= 930
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_100,
    (__shared__ alignas(16) cuda::std::uint32_t src_words[logical_endpoint_test::ring_chunk_words];
     __shared__ alignas(8) cuda::std::uint64_t barrier;

     if (threadIdx.x == 0) {
       const auto chunk = static_cast<cuda::std::uint32_t>(blockIdx.x);
       src_words[0]     = 0x13570000u + rank;
       src_words[1]     = 0x24680000u + chunk;
       src_words[2]     = 0xdead0000u + rank;
       src_words[3]     = 0xcafe0000u + chunk;

       cuda::ptx::mbarrier_init(&barrier, 1u);
       cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
       cuda::ptx::fabric_try_put(
         cuda::ptx::space_shared,
         cuda::ptx::sem_relaxed,
         cuda::ptx::scope_sys,
         endpoint.native_handle(),
         static_cast<cuda::std::uint64_t>(chunk) * logical_endpoint_test::ring_chunk_bytes,
         src_words,
         logical_endpoint_test::ring_chunk_bytes,
         &barrier);
       cuda::ptx::fabric_submit();
       cuda::ptx::mbarrier_arrive_expect_tx(
         cuda::ptx::sem_relaxed,
         cuda::ptx::scope_cta,
         cuda::ptx::space_shared,
         &barrier,
         logical_endpoint_test::ring_chunk_bytes / logical_endpoint_test::tx_granularity);

       status[chunk] = logical_endpoint_test::wait_for_mbarrier_completion(&barrier)
                       ? logical_endpoint_test::status_success
                       : logical_endpoint_test::status_timeout;
     }),
    (if (threadIdx.x == 0) { status[blockIdx.x] = logical_endpoint_test::status_unsupported; }))
#  else // ^^^ __cccl_ptx_isa >= 930 ^^^ / vvv __cccl_ptx_isa < 930
  if (threadIdx.x == 0)
  {
    status[blockIdx.x] = logical_endpoint_test::status_unsupported;
  }
#  endif // __cccl_ptx_isa < 930
}

__global__ void fabric_ring_signal_flag_kernel(
  cuda::unicast_logical_endpoint_ref endpoint,
  cuda::std::uint64_t flag_offset,
  cuda::std::uint32_t* local_flag,
  cuda::std::uint32_t* status)
{
#  if __cccl_ptx_isa >= 930
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_100,
    (__shared__ alignas(16) cuda::std::uint32_t src_words[logical_endpoint_test::ring_chunk_words];
     __shared__ alignas(8) cuda::std::uint64_t barrier;

     if (threadIdx.x == 0) {
       src_words[0] = 1;
       src_words[1] = 0;
       src_words[2] = 0;
       src_words[3] = 0;

       cuda::ptx::mbarrier_init(&barrier, 1u);
       cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
       cuda::ptx::fabric_try_put(
         cuda::ptx::space_shared,
         cuda::ptx::sem_relaxed,
         cuda::ptx::scope_sys,
         endpoint.native_handle(),
         flag_offset,
         src_words,
         logical_endpoint_test::ring_sync_bytes,
         &barrier);
       cuda::ptx::fabric_submit();
       cuda::ptx::mbarrier_arrive_expect_tx(
         cuda::ptx::sem_relaxed,
         cuda::ptx::scope_cta,
         cuda::ptx::space_shared,
         &barrier,
         logical_endpoint_test::ring_sync_bytes / logical_endpoint_test::tx_granularity);

       if (!logical_endpoint_test::wait_for_mbarrier_completion(&barrier)
           || !logical_endpoint_test::wait_for_remote_flag(local_flag))
       {
         status[logical_endpoint_test::ring_chunk_count] = logical_endpoint_test::status_timeout;
         return;
       }

       status[logical_endpoint_test::ring_chunk_count] = logical_endpoint_test::status_success;
     }),
    (if (threadIdx.x == 0) {
      status[logical_endpoint_test::ring_chunk_count] = logical_endpoint_test::status_unsupported;
    }))
#  else // ^^^ __cccl_ptx_isa >= 930 ^^^ / vvv __cccl_ptx_isa < 930
  if (threadIdx.x == 0)
  {
    status[logical_endpoint_test::ring_chunk_count] = logical_endpoint_test::status_unsupported;
  }
#  endif // __cccl_ptx_isa < 930
}

__global__ void fabric_ring_put_counted_kernel(
  cuda::unicast_logical_endpoint_ref endpoint,
  cuda::std::uint64_t counter_offset,
  cuda::std::uint64_t expected_bytes,
  cuda::std::uint32_t rank,
  cuda::std::uint64_t* local_counter,
  cuda::std::uint32_t* status)
{
#  if __cccl_ptx_isa >= 930
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_100,
    (__shared__ alignas(16) cuda::std::uint32_t src_words[logical_endpoint_test::ring_chunk_words];
     __shared__ alignas(8) cuda::std::uint64_t barrier;

     if (threadIdx.x == 0) {
       const auto chunk = static_cast<cuda::std::uint32_t>(blockIdx.x);
       src_words[0]     = 0x13570000u + rank;
       src_words[1]     = 0x24680000u + chunk;
       src_words[2]     = 0xdead0000u + rank;
       src_words[3]     = 0xcafe0000u + chunk;

       cuda::ptx::mbarrier_init(&barrier, 1u);
       cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
       cuda::ptx::fabric_try_put_counted(
         cuda::ptx::space_shared,
         cuda::ptx::sem_relaxed,
         cuda::ptx::scope_sys,
         endpoint.native_handle(),
         static_cast<cuda::std::uint64_t>(chunk) * logical_endpoint_test::ring_chunk_bytes,
         counter_offset,
         src_words,
         logical_endpoint_test::ring_chunk_bytes,
         &barrier);
       cuda::ptx::fabric_submit();
       cuda::ptx::mbarrier_arrive_expect_tx(
         cuda::ptx::sem_relaxed,
         cuda::ptx::scope_cta,
         cuda::ptx::space_shared,
         &barrier,
         logical_endpoint_test::ring_chunk_bytes / logical_endpoint_test::tx_granularity);

       if (!logical_endpoint_test::wait_for_mbarrier_completion(&barrier))
       {
         status[chunk] = logical_endpoint_test::status_timeout;
         return;
       }

       status[chunk] = logical_endpoint_test::status_success;
       if (blockIdx.x == 0)
       {
         status[logical_endpoint_test::ring_chunk_count] =
           logical_endpoint_test::wait_for_counter(local_counter, expected_bytes)
             ? logical_endpoint_test::status_success
             : logical_endpoint_test::status_timeout;
       }
     }),
    (if (threadIdx.x == 0) { status[blockIdx.x] = logical_endpoint_test::status_unsupported; }))
#  else // ^^^ __cccl_ptx_isa >= 930 ^^^ / vvv __cccl_ptx_isa < 930
  if (threadIdx.x == 0)
  {
    status[blockIdx.x] = logical_endpoint_test::status_unsupported;
  }
#  endif // __cccl_ptx_isa < 930
}

__global__ void fabric_try_put_multimem_smoke_kernel(
  cuda::multicast_logical_endpoint_ref endpoint, cuda::std::uint64_t endpoint_offset, cuda::std::uint32_t* status)
{
#  if __cccl_ptx_isa >= 930
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_100,
    (__shared__ alignas(16) cuda::std::uint32_t src_words[logical_endpoint_test::payload_words];
     __shared__ alignas(8) cuda::std::uint64_t barrier;

     if (threadIdx.x == 0) {
       src_words[0] = 0x13572468u;
       src_words[1] = 0x24681357u;
       src_words[2] = 0xdeadbeefu;
       src_words[3] = 0xcafef00du;

       cuda::ptx::mbarrier_init(&barrier, 1u);
       cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
       cuda::ptx::fabric_try_put_multimem(
         cuda::ptx::space_shared,
         cuda::ptx::sem_relaxed,
         cuda::ptx::scope_sys,
         endpoint.native_handle(),
         endpoint_offset,
         src_words,
         logical_endpoint_test::payload_bytes,
         &barrier);
       cuda::ptx::fabric_submit();
       cuda::ptx::mbarrier_arrive_expect_tx(
         cuda::ptx::sem_relaxed,
         cuda::ptx::scope_cta,
         cuda::ptx::space_shared,
         &barrier,
         logical_endpoint_test::payload_bytes / logical_endpoint_test::tx_granularity);

       for (int iteration = 0; iteration < 1000000; ++iteration)
       {
         if (cuda::ptx::mbarrier_try_wait_parity(cuda::ptx::sem_acquire, cuda::ptx::scope_cta, &barrier, 0))
         {
           *status = logical_endpoint_test::status_success;
           return;
         }
       }

       *status = logical_endpoint_test::status_timeout;
     }),
    (if (threadIdx.x == 0) { *status = logical_endpoint_test::status_unsupported; }))
#  else // ^^^ __cccl_ptx_isa >= 930 ^^^ / vvv __cccl_ptx_isa < 930
  if (threadIdx.x == 0)
  {
    *status = logical_endpoint_test::status_unsupported;
  }
#  endif // __cccl_ptx_isa < 930
}
} // namespace logical_endpoint_test

#endif // _CCCL_CTK_AT_LEAST(13, 3)

#endif // TEST_SUPPORT_LOGICAL_ENDPOINT_TEST_HELPER_H
