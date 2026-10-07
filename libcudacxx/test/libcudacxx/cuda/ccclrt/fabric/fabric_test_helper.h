//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_FABRIC_TEST_HELPER_H
#define TEST_FABRIC_TEST_HELPER_H

#include <testing.cuh>

#if _CCCL_CUDACC_AT_LEAST(13, 4) && __cccl_ptx_isa >= 940

#  include <cuda/buffer>
#  include <cuda/devices>
#  include <cuda/fabric>
#  include <cuda/hierarchy>
#  include <cuda/launch>
#  include <cuda/logical_endpoint>
#  include <cuda/std/chrono>
#  include <cuda/std/cstddef>
#  include <cuda/std/cstdint>
#  include <cuda/std/span>
#  include <cuda/std/type_traits>
#  include <cuda/stream>

#  include <cstdio>

#  include "logical_endpoint_test_helper.h"

namespace fabric_test
{
constexpr cuda::std::size_t test_region_bytes       = 512;
constexpr cuda::std::uint64_t data_offset           = 32;
constexpr cuda::std::uint64_t data_bytes            = 64;
constexpr cuda::std::uint64_t transfer_bytes        = 32;
constexpr cuda::std::uint64_t flag_offset           = data_offset + data_bytes;
constexpr cuda::std::uint64_t counter_offset        = 256;
constexpr cuda::std::uint32_t initial_payload_value = 8;

TEST_NV_DIAG_SUPPRESS(static_var_with_dynamic_init)
// Collective callers must synchronize before using the initialized barrier.
TEST_DEVICE_FUNC inline cuda::shared_barrier& make_barrier(cuda::std::size_t rank = 0)
{
  __shared__ cuda::shared_barrier bar;
  if (rank == 0)
  {
    init(&bar, 1);
  }
  return bar;
}
_CCCL_END_NV_DIAG_SUPPRESS()

TEST_DEVICE_FUNC inline void wait_success(cuda::shared_barrier& bar, int count)
{
  const auto token = bar.arrive_tx(1, count);
  auto status      = bar.try_wait_for(token, cuda::std::chrono::seconds{1}, cuda::return_status);
  REQUIRE(status.complete());
  if (status.has_report())
  {
    printf("fabric completion reported %u errors\n", status.get_error_count(cuda::status_source::generic_fabric));
    REQUIRE(false);
  }
}

inline void require_runtime_support()
{
  if (const char* reason = logical_endpoint_test::runtime_unsupported_reason(2))
  {
    SKIP(reason);
  }
}

template <class Spec>
cuda::logical_endpoint_limits require_fabric_support(const Spec& spec, cuda::device_ref issuer, cuda::device_ref peer)
{
  logical_endpoint_test::support_result support;
  if constexpr (cuda::std::is_same_v<Spec, cuda::unicast_logical_endpoint_spec>)
  {
    support = logical_endpoint_test::probe_logical_endpoint_support(spec, peer);
  }
  else
  {
    support = logical_endpoint_test::probe_logical_endpoint_support(spec, issuer, peer);
  }
  if (!support.supported)
  {
    SKIP(support.reason);
  }
  if (!logical_endpoint_test::fabric_ptx_supported(issuer, peer))
  {
    SKIP("fabric runtime tests require SM100+ and PTX 9.4");
  }
  if (!logical_endpoint_test::memory_pools_supported(issuer, peer))
  {
    SKIP("memory pools unavailable");
  }
  return support.limits;
}

// TODO: Remove endpoint_storage once cuda::buffer is better integrated with logical endpoint binding.
struct endpoint_storage
{
  cuda::device_buffer<cuda::std::uint8_t> allocation;
  cuda::std::span<cuda::std::uint8_t> bound;

  endpoint_storage(cuda::stream_ref stream,
                   cuda::device_ref device,
                   cuda::logical_endpoint_limits limits,
                   cuda::std::size_t minimum_bytes = logical_endpoint_test::minimum_bytes)
      : allocation(cuda::make_device_buffer<cuda::std::uint8_t>(
          stream, device, allocation_size(limits, minimum_bytes), cuda::no_init))
  {
    // Pool allocation is stream ordered; complete it before binding on the host.
    stream.sync();
    auto bytes = logical_endpoint_test::align_up(minimum_bytes, limits.bind_alignment);
    auto address =
      logical_endpoint_test::align_up(reinterpret_cast<cuda::std::uintptr_t>(allocation.data()), limits.bind_alignment);
    REQUIRE(address + bytes <= reinterpret_cast<cuda::std::uintptr_t>(allocation.data()) + allocation.size());
    bound = {reinterpret_cast<cuda::std::uint8_t*>(address), bytes};
  }

private:
  static cuda::std::size_t allocation_size(cuda::logical_endpoint_limits limits, cuda::std::size_t minimum_bytes)
  {
    REQUIRE(limits.bind_alignment != 0);
    auto bytes = logical_endpoint_test::align_up(minimum_bytes, limits.bind_alignment);
    REQUIRE(bytes <= limits.max_size);
    return bytes + limits.bind_alignment;
  }
};

template <class T>
struct fabric_payload
{
  cuda::std::span<cuda::std::uint8_t> bytes;
  cuda::std::span<T> values;
  cuda::std::span<cuda::std::uint32_t> flag;
  cuda::std::span<cuda::std::uint64_t> counter;
};

template <class T>
fabric_payload<T> make_payload(endpoint_storage& storage)
{
  auto bytes = storage.bound.first(test_region_bytes);
  return {bytes,
          {reinterpret_cast<T*>(bytes.data() + data_offset), data_bytes / sizeof(T)},
          {reinterpret_cast<cuda::std::uint32_t*>(bytes.data() + flag_offset), 4},
          {reinterpret_cast<cuda::std::uint64_t*>(bytes.data() + counter_offset), 1}};
}

template <class T>
struct fabric_case
{
  using value_type               = T;
  static constexpr bool counted  = false;
  static constexpr bool signaled = false;

  TEST_DEVICE_FUNC static T initial_value(cuda::std::size_t, int = 0)
  {
    return T{initial_payload_value};
  }
};

template <class Case>
struct initialize_payload
{
  int device_index = 0;

  template <class Config>
  TEST_DEVICE_FUNC void operator()(Config, fabric_payload<typename Case::value_type> payload) const
  {
    for (auto& byte : payload.bytes)
    {
      byte = 0xa5;
    }
    for (cuda::std::size_t i = 0; i < payload.values.size(); ++i)
    {
      payload.values[i] = Case::initial_value(i, device_index);
    }
    for (auto& word : payload.flag)
    {
      word = 0;
    }
    payload.counter[0] = 0;
  }
};

template <class Case>
struct validate_payload
{
  cuda::std::uint64_t expected_counter = 0;
  bool signaled                        = false;
  int device_index                     = 0;

  template <class Config>
  TEST_DEVICE_FUNC void operator()(Config, fabric_payload<typename Case::value_type> payload) const
  {
    for (cuda::std::size_t i = 0; i < payload.values.size(); ++i)
    {
      CHECK(payload.values[i] == Case::expected_value(i, device_index));
    }
    CHECK(payload.counter[0] == expected_counter);
    for (cuda::std::size_t i = 0; i < payload.flag.size(); ++i)
    {
      CHECK(payload.flag[i] == cuda::std::uint32_t(signaled && i == 0));
    }
    // Also catch writes outside the payload, flag and counter.
    for (cuda::std::size_t offset = 0; offset < payload.bytes.size(); ++offset)
    {
      if ((offset >= data_offset && offset < data_offset + payload.values.size_bytes())
          || (offset >= flag_offset && offset < flag_offset + payload.flag.size_bytes())
          || (offset >= counter_offset && offset < counter_offset + sizeof(cuda::std::uint64_t)))
      {
        continue;
      }
      CHECK(payload.bytes[offset] == 0xa5);
    }
  }
};

template <class Kernel>
void run_unicast(Kernel kernel)
{
  require_runtime_support();
  cuda::device_ref issuer{0};
  cuda::device_ref target_device{1};
  auto spec = cuda::unicast_logical_endpoint_spec{
    target_device,
    Kernel::counted ? cuda::logical_endpoint_flag::counted_ops : cuda::logical_endpoint_flag::none,
    cuda::logical_endpoint_ipc_handle_type::none};
  auto limits = require_fabric_support(spec, issuer, target_device);

  cuda::stream stream{issuer};
  cuda::stream target_stream{target_device};
  endpoint_storage storage{target_stream, target_device, limits};
  auto payload = make_payload<typename Kernel::value_type>(storage);
  cuda::unicast_logical_endpoint endpoint{spec, storage.bound.size()};
  REQUIRE(endpoint.wait_ready_for(logical_endpoint_test::ready_timeout));
  endpoint.bind(target_device, 0, storage.bound.data(), storage.bound.size());

  auto config = cuda::make_config(cuda::make_hierarchy(cuda::grid_dims<1>(), cuda::block_dims<1>()));
  cuda::launch(target_stream, config, initialize_payload<Kernel>{}, payload);
  // Initialize on the owning GPU before the issuer accesses its endpoint.
  target_stream.sync();
  cuda::launch(stream, config, kernel, endpoint);
  stream.sync();

  cuda::launch(target_stream,
               config,
               validate_payload<Kernel>{Kernel::counted ? transfer_bytes : cuda::std::uint64_t{0}, Kernel::signaled},
               payload);
  target_stream.sync();
  endpoint.unbind(target_device, 0, storage.bound.size());
}
} // namespace fabric_test

#endif // _CCCL_CUDACC_AT_LEAST(13, 4) && __cccl_ptx_isa >= 940

#endif // TEST_FABRIC_TEST_HELPER_H
