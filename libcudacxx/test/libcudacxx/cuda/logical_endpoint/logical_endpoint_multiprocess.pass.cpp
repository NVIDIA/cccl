//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc
// UNSUPPORTED: libcpp-no-exceptions
// REQUIRES: linux
// Forking must happen before CUDA initialization, so use the test's own host-only main.
// ADDITIONAL_COMPILE_DEFINITIONS: _LIBCUDACXX_FORCE_INCLUDE_H

#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/launch>
#include <cuda/logical_endpoint>
#include <cuda/memory_pool>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/span>
#include <cuda/stream>

#include "logical_endpoint_test_helper.h"
#include "process_pair_test_helper.h"
#include "test_macros.h"

#if _CCCL_CTK_AT_LEAST(13, 3)

namespace
{
void require_fabric_ipc_support(int minimum_device_count)
{
  if (const char* reason = logical_endpoint_test::runtime_unsupported_reason(minimum_device_count))
  {
    process_pair_test::skip_unless(false, reason);
  }
#  if _CCCL_CTK_AT_LEAST(13, 4)
  process_pair_test::skip_unless(!cuda::__driver::__version_below(13, 4), "fabric IPC requires a CUDA 13.4 driver");
#  else // ^^^ _CCCL_CTK_AT_LEAST(13, 4) ^^^ / vvv _CCCL_CTK_BELOW(13, 4) vvv
  process_pair_test::skip_unless(false, "fabric IPC tests require CUDA Toolkit 13.4");
#  endif // _CCCL_CTK_BELOW(13, 4)
}

struct endpoint_config
{
  cuda::logical_endpoint_limits limits;
  cuda::std::uint64_t bytes;
};

template <class Spec, class... Devices>
endpoint_config require_endpoint_config(const Spec& spec, Devices... devices)
{
  auto support = logical_endpoint_test::probe_logical_endpoint_support(spec, devices...);
  process_pair_test::skip_unless(support.supported, support.reason);
  const auto bytes = logical_endpoint_test::endpoint_size(support.limits);
  assert(bytes >= logical_endpoint_test::minimum_bytes);
  assert(support.limits.max_size == 0 || bytes <= support.limits.max_size);
  return {support.limits, bytes};
}

template <class Buffer>
cuda::std::span<cuda::std::uint8_t> aligned_bind_region(Buffer& allocation, endpoint_config config)
{
  const auto address = reinterpret_cast<cuda::std::uintptr_t>(allocation.data());
  const auto aligned = logical_endpoint_test::align_up(address, config.limits.bind_alignment);
  assert(aligned + config.bytes <= address + allocation.size());
  return {reinterpret_cast<cuda::std::uint8_t*>(aligned), static_cast<size_t>(config.bytes)};
}

cuda::unicast_logical_endpoint import_unicast(const cuda::logical_endpoint_fabric_handle& handle)
{
  cuda::unicast_logical_endpoint imported{handle};
  assert(imported.wait_ready_for(logical_endpoint_test::ready_timeout));
  return imported;
}

void put_to_unicast(cuda::unicast_logical_endpoint& target)
{
  cuda::device_ref device{0};
  cuda::stream stream{device};
  auto status = cuda::make_device_buffer<cuda::std::uint32_t>(stream, device, 1, cuda::no_init);
  cuda::fill_bytes(stream, status, 0);
  auto config = cuda::make_config(cuda::make_hierarchy(cuda::grid_dims(1), cuda::block_dims<1>()));
  cuda::launch(
    stream, config, logical_endpoint_test::fabric_try_put_smoke_kernel, target, cuda::std::uint64_t{0}, status.data());
  stream.sync();

  cuda::std::uint32_t host_status = 0;
  cuda::copy_bytes(stream, status, cuda::std::span<cuda::std::uint32_t>{&host_status, 1});
  stream.sync();
  assert(host_status == logical_endpoint_test::status_success);
}

int test_unicast_import()
{
  return process_pair_test::run_process_pair(
    "unicast import",
    [](process_pair_test::ipc_channel& peer) {
      require_fabric_ipc_support(1);
      cuda::device_ref device{0};
      auto spec   = cuda::unicast_logical_endpoint_spec{device};
      auto config = require_endpoint_config(spec, device);
      cuda::unicast_logical_endpoint local{spec, config.bytes};
      assert(local.wait_ready_for(logical_endpoint_test::ready_timeout));

      peer.send(local.export_endpoint(cuda::fabric_handle));
      peer.sync(); // Keep the exporter alive until import has completed.
    },
    [](process_pair_test::ipc_channel& peer) {
      auto imported = import_unicast(peer.receive<cuda::logical_endpoint_fabric_handle>());
      peer.sync();
    });
}

int test_unicast_put()
{
  return process_pair_test::run_process_pair(
    "imported unicast put",
    [](process_pair_test::ipc_channel& peer) {
      require_fabric_ipc_support(1);
      cuda::device_ref device{0};
      process_pair_test::skip_unless(
        logical_endpoint_test::fabric_memory_pools_supported(device), "fabric memory pools are not supported");
      process_pair_test::skip_unless(
        logical_endpoint_test::fabric_ptx_supported(device), "fabric PTX requires an SM 100+ device and PTX ISA 9.3+");
      auto spec   = cuda::unicast_logical_endpoint_spec{device};
      auto config = require_endpoint_config(spec, device);

      cuda::stream stream{device};
      cuda::shared_device_memory_pool pool{device, logical_endpoint_test::fabric_memory_pool_properties()};
      auto allocation =
        cuda::make_buffer<cuda::std::uint8_t>(stream, pool, config.bytes + config.limits.bind_alignment, cuda::no_init);
      stream.sync();
      auto memory = aligned_bind_region(allocation, config);
      cuda::fill_bytes(stream, memory.first(logical_endpoint_test::payload_bytes), 0);
      stream.sync();

      cuda::unicast_logical_endpoint local{spec, config.bytes};
      assert(local.wait_ready_for(logical_endpoint_test::ready_timeout));
      local.bind(device, 0, memory.data(), memory.size());
      // On failure, isolated process teardown cleans up the binding and allocation.
      peer.send(local.export_endpoint(cuda::fabric_handle));
      peer.sync(); // The importer has completed its put before we inspect memory.

      cuda::std::uint32_t observed[logical_endpoint_test::payload_words]{};
      cuda::copy_bytes(stream,
                       memory.first(logical_endpoint_test::payload_bytes),
                       cuda::std::span<cuda::std::uint32_t>{observed, logical_endpoint_test::payload_words});
      stream.sync();
      assert(observed[0] == 0x13572468u && observed[1] == 0x24681357u && observed[2] == 0xdeadbeefu
             && observed[3] == 0xcafef00du);
      local.unbind(device, 0, memory.size());
    },
    [](process_pair_test::ipc_channel& peer) {
      auto imported = import_unicast(peer.receive<cuda::logical_endpoint_fabric_handle>());
      put_to_unicast(imported);
      peer.sync();
    });
}

int test_multicast_import()
{
  return process_pair_test::run_process_pair(
    "multicast import and bind",
    [](process_pair_test::ipc_channel& peer) {
      require_fabric_ipc_support(2);
      cuda::device_ref device{0};
      cuda::device_ref child_device{1};
      process_pair_test::skip_unless(logical_endpoint_test::fabric_memory_pools_supported(device, child_device),
                                     "fabric memory pools are not supported");
      auto spec   = cuda::multicast_logical_endpoint_spec{2};
      auto config = require_endpoint_config(spec, device, child_device);

      cuda::multicast_logical_endpoint local{spec, config.bytes};
      local.add_device(device);
      peer.send(local.export_endpoint(cuda::fabric_handle));
      peer.sync(); // Both devices must be added before either process binds memory.
      assert(local.wait_ready_for(logical_endpoint_test::ready_timeout));

      cuda::stream stream{device};
      cuda::shared_device_memory_pool pool{device, logical_endpoint_test::fabric_memory_pool_properties()};
      auto allocation =
        cuda::make_buffer<cuda::std::uint8_t>(stream, pool, config.bytes + config.limits.bind_alignment, cuda::no_init);
      stream.sync();
      auto memory = aligned_bind_region(allocation, config);
      local.bind(device, 0, memory.data(), memory.size());
      local.unbind(device, 0, memory.size());
      peer.sync();
    },
    [](process_pair_test::ipc_channel& peer) {
      cuda::multicast_logical_endpoint imported{peer.receive<cuda::logical_endpoint_fabric_handle>()};
      imported.add_device(cuda::device_ref{1});
      assert(imported.wait_ready_for(logical_endpoint_test::ready_timeout));
      peer.sync();
      peer.sync(); // Retain the import until the exporter's bind/unbind check finishes.
    });
}
} // namespace

#endif // _CCCL_CTK_AT_LEAST(13, 3)

int main(int, char**)
{
#if _CCCL_CTK_AT_LEAST(13, 3)
  return test_unicast_import() || test_unicast_put() || test_multicast_import();
#else // ^^^ _CCCL_CTK_AT_LEAST(13, 3) ^^^ / vvv _CCCL_CTK_BELOW(13, 3) vvv
  return 0;
#endif // _CCCL_CTK_BELOW(13, 3)
}
