//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <testing.cuh>

#if _CCCL_CTK_AT_LEAST(13, 3)

#  include <cuda/algorithm>
#  include <cuda/buffer>
#  include <cuda/devices>
#  include <cuda/launch>
#  include <cuda/logical_endpoint>
#  include <cuda/memory_pool>
#  include <cuda/std/cstdint>
#  include <cuda/std/limits>
#  include <cuda/std/span>
#  include <cuda/std/utility>
#  include <cuda/stream>

#  include <stdexcept>

#  include <cuda.h>
#  include <cuda_runtime_api.h>

#  include "logical_endpoint_test_helper.h"

namespace
{
template <class Spec, class... Devices>
[[nodiscard]] cuda::logical_endpoint_limits
validate_logical_endpoint_support(const Spec& spec, cuda::device_ref device, Devices... devices)
{
  auto support = logical_endpoint_test::probe_logical_endpoint_support(spec, device, devices...);
  if (!support.supported)
  {
    SKIP(support.reason);
  }
  return support.limits;
}

[[nodiscard]] cuda::unicast_logical_endpoint_spec
unicast_spec(cuda::device_ref device, cuda::logical_endpoint_flag flags = cuda::logical_endpoint_flag::none)
{
  return cuda::unicast_logical_endpoint_spec{device, flags, cuda::logical_endpoint_ipc_handle_type::none};
}

[[nodiscard]] cuda::multicast_logical_endpoint_spec
multicast_spec(unsigned int num_devices, cuda::logical_endpoint_flag flags = cuda::logical_endpoint_flag::none)
{
  return cuda::multicast_logical_endpoint_spec{num_devices, flags, cuda::logical_endpoint_ipc_handle_type::none};
}

[[nodiscard]] cuda::std::uint64_t smoke_size(cuda::logical_endpoint_limits limits)
{
  REQUIRE(limits.bind_alignment != 0);
  const cuda::std::uint64_t bytes = logical_endpoint_test::endpoint_size(limits);
  REQUIRE(bytes >= logical_endpoint_test::minimum_bytes);
  REQUIRE((bytes % limits.bind_alignment) == 0);
  if (limits.max_size != 0)
  {
    REQUIRE(bytes <= limits.max_size);
  }
  return bytes;
}

void check_ring_statuses(const cuda::std::uint32_t* statuses)
{
  for (cuda::std::uint32_t index = 0; index < logical_endpoint_test::ring_status_words; ++index)
  {
    CHECK(statuses[index] == logical_endpoint_test::status_success);
  }
}

void check_ring_payload(const cuda::std::uint32_t* observed, cuda::std::uint32_t rank)
{
  for (cuda::std::uint32_t chunk = 0; chunk < logical_endpoint_test::ring_chunk_count; ++chunk)
  {
    const auto base = chunk * logical_endpoint_test::ring_chunk_words;
    CHECK(observed[base + 0] == 0x13570000u + rank);
    CHECK(observed[base + 1] == 0x24680000u + chunk);
    CHECK(observed[base + 2] == 0xdead0000u + rank);
    CHECK(observed[base + 3] == 0xcafe0000u + chunk);
  }
}

void destroy_released_endpoint(cuda::logical_endpoint_id id)
{
  REQUIRE(::cuda::__driver::__logicalEndpointDestroyNoThrow(id.native_handle()) == cudaSuccess);
}

template <class... Devices>
void skip_if_memory_pools_are_unsupported(cuda::device_ref device, Devices... devices)
{
  if (!logical_endpoint_test::memory_pools_supported(device, devices...))
  {
    SKIP("stream-ordered memory pools are not supported");
  }
}

template <class... Devices>
void skip_if_fabric_ptx_smoke_is_unsupported(cuda::device_ref device, Devices... devices)
{
  if (!logical_endpoint_test::fabric_ptx_supported(device, devices...))
  {
    SKIP("fabric PTX logical endpoint smoke requires an SM 100+ device and PTX ISA 9.3+");
  }
}

void skip_if_vmm_allocations_are_unsupported(cuda::device_ref device)
{
  const auto native_device = cuda::__driver::__deviceGet(device.get());
  if (cuda::__driver::__deviceGetAttribute(CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, native_device) == 0)
  {
    SKIP("virtual memory management is not supported");
  }
}

[[nodiscard]] cuda::std::pair<cuda::device_ref, cuda::device_ref> multicast_test_devices()
{
  if (cuda::devices.size() < 2)
  {
    SKIP("multicast logical endpoint tests require at least two devices");
  }

  return {cuda::device_ref{0}, cuda::device_ref{1}};
}

[[nodiscard]] cuda::std::pair<cuda::device_ref, cuda::device_ref> two_unicast_test_devices()
{
  if (cuda::devices.size() < 2)
  {
    SKIP("unicast logical endpoint CFT ring tests require at least two devices");
  }

  return {cuda::device_ref{0}, cuda::device_ref{1}};
}

class generic_allocation
{
  CUmemGenericAllocationHandle handle_{};
  PFN_cuMemRelease_v10020 mem_release_{};
  cuda::std::uint64_t bytes_{};
  bool owns_handle_ = false;

public:
  generic_allocation(cuda::device_ref device, cuda::std::uint64_t requested_bytes)
  {
    auto mem_get_allocation_granularity = reinterpret_cast<PFN_cuMemGetAllocationGranularity_v10020>(
      cuda::__driver::__get_driver_entry_point("cuMemGetAllocationGranularity", 10, 2));
    auto mem_create =
      reinterpret_cast<PFN_cuMemCreate_v10020>(cuda::__driver::__get_driver_entry_point("cuMemCreate", 10, 2));
    mem_release_ =
      reinterpret_cast<PFN_cuMemRelease_v10020>(cuda::__driver::__get_driver_entry_point("cuMemRelease", 10, 2));

    CUmemAllocationProp allocation_prop{};
    allocation_prop.type                 = CU_MEM_ALLOCATION_TYPE_PINNED;
    allocation_prop.location.type        = CU_MEM_LOCATION_TYPE_DEVICE;
    allocation_prop.location.id          = cuda::__driver::__deviceGet(device.get());
    allocation_prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;

    size_t allocation_granularity = 0;
    REQUIRE(mem_get_allocation_granularity(&allocation_granularity, &allocation_prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM)
            == CUDA_SUCCESS);
    bytes_ = logical_endpoint_test::align_up(requested_bytes, static_cast<cuda::std::uint64_t>(allocation_granularity));

    REQUIRE(mem_create(&handle_, static_cast<size_t>(bytes_), &allocation_prop, 0) == CUDA_SUCCESS);
    owns_handle_ = true;
  }

  generic_allocation(const generic_allocation&)            = delete;
  generic_allocation& operator=(const generic_allocation&) = delete;

  ~generic_allocation()
  {
    if (owns_handle_)
    {
      static_cast<void>(mem_release_(handle_));
    }
  }

  [[nodiscard]] CUmemGenericAllocationHandle handle() const
  {
    return handle_;
  }

  [[nodiscard]] cuda::std::uint64_t bytes() const
  {
    return bytes_;
  }
};
} // namespace

C2H_CCCLRT_TEST("logical endpoint validates host-only state without driver calls", "[logical_endpoint]")
{
#  if TEST_HAS_EXCEPTIONS()
  CHECK_THROWS_AS(cuda::logical_endpoint_id_range{0}, std::invalid_argument);
#  endif // TEST_HAS_EXCEPTIONS()

  cuda::unicast_logical_endpoint unicast;
  cuda::multicast_logical_endpoint multicast;
  CHECK(!unicast.has_value());
  CHECK(!multicast.has_value());
  CHECK(unicast.size() == 0);
  CHECK(multicast.size() == 0);
  CHECK(unicast.bind_alignment() == 0);
  CHECK(multicast.bind_alignment() == 0);

  cuda::unicast_logical_endpoint moved_unicast{cuda::std::move(unicast)};
  cuda::multicast_logical_endpoint moved_multicast{cuda::std::move(multicast)};
  CHECK(!unicast.has_value());
  CHECK(!multicast.has_value());
  CHECK(!moved_unicast.has_value());
  CHECK(!moved_multicast.has_value());
}

C2H_CCCLRT_TEST("unicast logical endpoint lifecycle with caller-owned ID range", "[logical_endpoint]")
{
  cuda::device_ref device{0};
  auto spec = unicast_spec(device);

  auto limits = validate_logical_endpoint_support(spec, device);
  auto bytes  = smoke_size(limits);

  cuda::logical_endpoint_id_range ids{2};
  CHECK(ids.size() == 2);
  CHECK(ids[0] == ids.base_id());
  CHECK(ids[1] == ids.base_id() + 1);
  cuda::logical_endpoint_id_range ids_copy = ids;
  CHECK(ids_copy.base_id() == ids.base_id());
  CHECK(ids_copy.size() == ids.size());
  cuda::logical_endpoint_id_range ids_move{cuda::std::move(ids_copy)};
  CHECK(ids_copy.size() == 0);
  CHECK(ids_move.base_id() == ids.base_id());
  CHECK(ids_move.size() == ids.size());

  cuda::unicast_logical_endpoint endpoint{ids, 0, spec, bytes};
  CHECK(endpoint.has_value());
  CHECK(endpoint.id() == ids[0]);
  CHECK(endpoint.size() == bytes);
  CHECK(endpoint.bind_alignment() == limits.bind_alignment);
  REQUIRE(endpoint.wait_until_ready(logical_endpoint_test::ready_timeout));

  cuda::unicast_logical_endpoint moved{cuda::std::move(endpoint)};
  CHECK(!endpoint.has_value());
  CHECK(moved.has_value());
  CHECK(moved.id() == ids[0]);

  auto released = moved.release();
  CHECK(!moved.has_value());
  CHECK(released.first == ids[0]);
  REQUIRE(released.second.has_value());
  CHECK(released.second->base_id() == ids.base_id());
  CHECK(released.second->size() == ids.size());

  destroy_released_endpoint(released.first);
}

C2H_CCCLRT_TEST("logical endpoint retains caller-owned ID range after source range is destroyed", "[logical_endpoint]")
{
  cuda::device_ref device{0};
  auto spec = unicast_spec(device);

  auto limits = validate_logical_endpoint_support(spec, device);
  auto bytes  = smoke_size(limits);

  cuda::unicast_logical_endpoint endpoint;
  cuda::logical_endpoint_id expected_id{0};
  cuda::std::uint32_t expected_size = 0;
  {
    cuda::logical_endpoint_id_range ids{1};
    expected_id   = ids[0];
    expected_size = ids.size();
    endpoint      = cuda::unicast_logical_endpoint{ids, 0, spec, bytes};
  }

  CHECK(endpoint.has_value());
  CHECK(endpoint.id() == expected_id);
  REQUIRE(endpoint.wait_until_ready(logical_endpoint_test::ready_timeout));

  auto released = endpoint.release();
  CHECK(!endpoint.has_value());
  CHECK(released.first == expected_id);
  REQUIRE(released.second.has_value());
  CHECK(released.second->base_id() == expected_id);
  CHECK(released.second->size() == expected_size);

  destroy_released_endpoint(released.first);
}

C2H_CCCLRT_TEST("unicast logical endpoint lifecycle with internally reserved ID", "[logical_endpoint]")
{
  cuda::device_ref device{0};
  auto spec = unicast_spec(device);

  auto limits = validate_logical_endpoint_support(spec, device);
  auto bytes  = smoke_size(limits);

  cuda::unicast_logical_endpoint endpoint{spec, bytes};
  CHECK(endpoint.has_value());
  CHECK(endpoint.size() == bytes);
  REQUIRE(endpoint.wait_until_ready(logical_endpoint_test::ready_timeout));

  auto released = endpoint.release();
  CHECK(!endpoint.has_value());
  REQUIRE(released.second.has_value());
  CHECK(released.second->size() == 1);

  destroy_released_endpoint(released.first);
}

C2H_CCCLRT_TEST("unicast logical endpoint release from explicit ID has no retained ID range", "[logical_endpoint]")
{
  cuda::device_ref device{0};
  auto spec = unicast_spec(device);

  auto limits = validate_logical_endpoint_support(spec, device);
  auto bytes  = smoke_size(limits);

  cuda::logical_endpoint_id_range ids{1};
  cuda::unicast_logical_endpoint endpoint{ids[0], spec, bytes};
  CHECK(endpoint.has_value());
  CHECK(endpoint.id() == ids[0]);
  REQUIRE(endpoint.wait_until_ready(logical_endpoint_test::ready_timeout));

  auto released = endpoint.release();
  CHECK(!endpoint.has_value());
  CHECK(released.first == ids[0]);
  CHECK(!released.second.has_value());

  destroy_released_endpoint(released.first);
}

C2H_CCCLRT_TEST("unicast logical endpoint honors reported maximum size", "[logical_endpoint]")
{
  cuda::device_ref device{0};
  auto spec   = unicast_spec(device);
  auto limits = validate_logical_endpoint_support(spec, device);

  if (limits.max_size == 0)
  {
    SKIP("no maximum logical endpoint size is reported");
  }

  cuda::unicast_logical_endpoint at_max{spec, limits.max_size};
  REQUIRE(at_max.wait_until_ready(logical_endpoint_test::ready_timeout));

#  if TEST_HAS_EXCEPTIONS()
  REQUIRE(limits.max_size < cuda::std::numeric_limits<cuda::std::uint64_t>::max());
  CHECK_THROWS_AS((cuda::unicast_logical_endpoint{spec, limits.max_size + 1}), std::invalid_argument);
#  endif // TEST_HAS_EXCEPTIONS()
}

C2H_CCCLRT_TEST("unicast logical endpoint binds memory pool allocation by address", "[logical_endpoint]")
{
  cuda::device_ref device{0};
  auto spec = unicast_spec(device);
  skip_if_memory_pools_are_unsupported(device);

  auto limits                 = validate_logical_endpoint_support(spec, device);
  const auto alignment        = limits.bind_alignment;
  const auto bytes            = smoke_size(limits);
  const auto allocation_bytes = bytes + alignment;
  cuda::stream stream{device};
  auto resource = cuda::device_default_memory_pool(device);

  {
    auto allocation = cuda::make_buffer<cuda::std::uint8_t>(stream, resource, allocation_bytes, cuda::no_init);
    stream.sync();
    const auto allocation_addr = reinterpret_cast<cuda::std::uintptr_t>(allocation.data());
    const auto bind_addr       = logical_endpoint_test::align_up(allocation_addr, alignment);
    void* bind_ptr             = reinterpret_cast<void*>(bind_addr);
    cuda::unicast_logical_endpoint endpoint{spec, bytes};

    REQUIRE(bind_addr + bytes <= allocation_addr + allocation_bytes);
    REQUIRE(endpoint.wait_until_ready(logical_endpoint_test::ready_timeout));
    cuda::unicast_logical_endpoint_ref ref = endpoint;
    ref.bind(device, 0, bind_ptr, bytes);
    ref.unbind(device, 0, bytes);
  }
  stream.sync();
}

C2H_CCCLRT_TEST("unicast logical endpoint binds generic allocation handle", "[logical_endpoint]")
{
  cuda::device_ref device{0};
  auto spec = unicast_spec(device);
  skip_if_vmm_allocations_are_unsupported(device);

  auto limits = validate_logical_endpoint_support(spec, device);
  auto bytes  = smoke_size(limits);

  generic_allocation allocation{device, bytes};
  if (limits.max_size != 0)
  {
    REQUIRE(allocation.bytes() <= limits.max_size);
  }

  cuda::unicast_logical_endpoint endpoint{spec, allocation.bytes()};
  REQUIRE(endpoint.wait_until_ready(logical_endpoint_test::ready_timeout));
  endpoint.bind(device, 0, allocation.handle(), 0, allocation.bytes());
  endpoint.unbind(device, 0, allocation.bytes());
}

C2H_CCCLRT_TEST("unicast logical endpoint supports device-side fabric put to a bound endpoint", "[logical_endpoint]")
{
  cuda::device_ref device{0};
  auto spec = unicast_spec(device);
  skip_if_memory_pools_are_unsupported(device);
  skip_if_fabric_ptx_smoke_is_unsupported(device);

  auto limits                 = validate_logical_endpoint_support(spec, device);
  const auto alignment        = limits.bind_alignment;
  const auto bytes            = smoke_size(limits);
  const auto allocation_bytes = bytes + alignment;
  cuda::stream stream{device};
  auto resource = cuda::device_default_memory_pool(device);

  {
    auto allocation = cuda::make_buffer<cuda::std::uint8_t>(stream, resource, allocation_bytes, cuda::no_init);
    auto status     = cuda::make_device_buffer<cuda::std::uint32_t>(stream, device, 1, cuda::no_init);
    stream.sync();

    const auto allocation_addr = reinterpret_cast<cuda::std::uintptr_t>(allocation.data());
    const auto bind_addr       = logical_endpoint_test::align_up(allocation_addr, alignment);
    void* bind_ptr             = reinterpret_cast<void*>(bind_addr);
    cuda::unicast_logical_endpoint local{spec, bytes};

    REQUIRE(bind_addr + bytes <= allocation_addr + allocation_bytes);
    REQUIRE(local.wait_until_ready(logical_endpoint_test::ready_timeout));
    local.bind(device, 0, bind_ptr, bytes);
    cuda::fill_bytes(stream,
                     cuda::std::span<cuda::std::uint8_t>{
                       static_cast<cuda::std::uint8_t*>(bind_ptr), logical_endpoint_test::payload_bytes},
                     0);
    cuda::fill_bytes(stream, status, 0);

    auto config = cuda::make_config(cuda::make_hierarchy(cuda::grid_dims(1), cuda::block_dims<1>()));
    cuda::launch(
      stream, config, logical_endpoint_test::fabric_try_put_smoke_kernel, local, cuda::std::uint64_t{0}, status.data());
    stream.sync();

    cuda::std::uint32_t host_status = 0;
    cuda::std::uint32_t observed[logical_endpoint_test::payload_words]{};
    cuda::copy_bytes(stream, status, cuda::std::span<cuda::std::uint32_t>{&host_status, 1});
    cuda::copy_bytes(stream,
                     cuda::std::span<cuda::std::uint32_t>{
                       static_cast<cuda::std::uint32_t*>(bind_ptr), logical_endpoint_test::payload_words},
                     cuda::std::span<cuda::std::uint32_t>{observed, logical_endpoint_test::payload_words});
    stream.sync();

    CHECK(host_status == logical_endpoint_test::status_success);
    CHECK(observed[0] == 0x13572468u);
    CHECK(observed[1] == 0x24681357u);
    CHECK(observed[2] == 0xdeadbeefu);
    CHECK(observed[3] == 0xcafef00du);

    local.unbind(device, 0, bytes);
  }
  stream.sync();
}

C2H_CCCLRT_TEST("unicast logical endpoint supports counted device-side fabric put to a bound endpoint",
                "[logical_endpoint]")
{
  cuda::device_ref device{0};
  auto spec = unicast_spec(device, cuda::logical_endpoint_flag::counted_ops);
  skip_if_memory_pools_are_unsupported(device);
  skip_if_fabric_ptx_smoke_is_unsupported(device);

  auto limits               = validate_logical_endpoint_support(spec, device);
  const auto alignment      = limits.bind_alignment;
  const auto bytes          = smoke_size(limits);
  const auto counter_offset = logical_endpoint_test::align_up(
    logical_endpoint_test::payload_bytes, logical_endpoint_test::counted_counter_alignment);
  const auto allocation_bytes = bytes + alignment;
  cuda::stream stream{device};
  auto resource = cuda::device_default_memory_pool(device);

  REQUIRE(counter_offset + sizeof(cuda::std::uint64_t) <= bytes);

  {
    auto allocation = cuda::make_buffer<cuda::std::uint8_t>(stream, resource, allocation_bytes, cuda::no_init);
    auto status     = cuda::make_device_buffer<cuda::std::uint32_t>(stream, device, 1, cuda::no_init);
    stream.sync();

    const auto allocation_addr = reinterpret_cast<cuda::std::uintptr_t>(allocation.data());
    const auto bind_addr       = logical_endpoint_test::align_up(allocation_addr, alignment);
    void* bind_ptr             = reinterpret_cast<void*>(bind_addr);
    cuda::unicast_logical_endpoint local{spec, bytes};

    REQUIRE(bind_addr + bytes <= allocation_addr + allocation_bytes);
    REQUIRE(local.wait_until_ready(logical_endpoint_test::ready_timeout));
    local.bind(device, 0, bind_ptr, bytes);
    cuda::fill_bytes(stream,
                     cuda::std::span<cuda::std::uint8_t>{
                       static_cast<cuda::std::uint8_t*>(bind_ptr), counter_offset + sizeof(cuda::std::uint64_t)},
                     0);
    cuda::fill_bytes(stream, status, 0);

    auto config = cuda::make_config(cuda::make_hierarchy(cuda::grid_dims(1), cuda::block_dims<1>()));
    cuda::launch(
      stream,
      config,
      logical_endpoint_test::fabric_try_put_counted_smoke_kernel,
      local,
      cuda::std::uint64_t{0},
      counter_offset,
      status.data());
    stream.sync();

    cuda::std::uint32_t host_status = 0;
    cuda::std::uint32_t observed[logical_endpoint_test::payload_words]{};
    cuda::std::uint64_t observed_counter = 0;
    cuda::copy_bytes(stream, status, cuda::std::span<cuda::std::uint32_t>{&host_status, 1});
    cuda::copy_bytes(stream,
                     cuda::std::span<cuda::std::uint32_t>{
                       static_cast<cuda::std::uint32_t*>(bind_ptr), logical_endpoint_test::payload_words},
                     cuda::std::span<cuda::std::uint32_t>{observed, logical_endpoint_test::payload_words});
    cuda::copy_bytes(
      stream,
      cuda::std::span<cuda::std::uint64_t>{
        reinterpret_cast<cuda::std::uint64_t*>(static_cast<cuda::std::uint8_t*>(bind_ptr) + counter_offset), 1},
      cuda::std::span<cuda::std::uint64_t>{&observed_counter, 1});
    stream.sync();

    CHECK(host_status == logical_endpoint_test::status_success);
    CHECK(observed[0] == 0x13572468u);
    CHECK(observed[1] == 0x24681357u);
    CHECK(observed[2] == 0xdeadbeefu);
    CHECK(observed[3] == 0xcafef00du);
    CHECK(observed_counter == logical_endpoint_test::payload_bytes);

    local.unbind(device, 0, bytes);
  }
  stream.sync();
}

C2H_CCCLRT_TEST("unicast logical endpoint supports programming-guide CFT ring with flags", "[logical_endpoint]")
{
  auto [device, peer_device] = two_unicast_test_devices();
  auto spec                  = unicast_spec(device);
  auto peer_spec             = unicast_spec(peer_device);
  skip_if_memory_pools_are_unsupported(device, peer_device);
  skip_if_fabric_ptx_smoke_is_unsupported(device, peer_device);

  auto limits               = validate_logical_endpoint_support(spec, device);
  auto peer_limits          = validate_logical_endpoint_support(peer_spec, peer_device);
  const auto alignment      = limits.bind_alignment;
  const auto peer_alignment = peer_limits.bind_alignment;
  const auto bytes          = smoke_size(limits);
  const auto peer_bytes     = smoke_size(peer_limits);
  const auto sync_offset    = logical_endpoint_test::align_up(
    logical_endpoint_test::ring_payload_bytes, logical_endpoint_test::counted_counter_alignment);
  const auto peer_sync_offset = logical_endpoint_test::align_up(
    logical_endpoint_test::ring_payload_bytes, logical_endpoint_test::counted_counter_alignment);
  const auto allocation_bytes      = bytes + alignment;
  const auto peer_allocation_bytes = peer_bytes + peer_alignment;
  cuda::stream stream{device};
  cuda::stream peer_stream{peer_device};
  auto resource      = cuda::device_default_memory_pool(device);
  auto peer_resource = cuda::device_default_memory_pool(peer_device);

  REQUIRE(sync_offset + logical_endpoint_test::ring_sync_bytes <= bytes);
  REQUIRE(peer_sync_offset + logical_endpoint_test::ring_sync_bytes <= peer_bytes);

  {
    auto allocation = cuda::make_buffer<cuda::std::uint8_t>(stream, resource, allocation_bytes, cuda::no_init);
    auto peer_allocation =
      cuda::make_buffer<cuda::std::uint8_t>(peer_stream, peer_resource, peer_allocation_bytes, cuda::no_init);
    auto status = cuda::make_device_buffer<cuda::std::uint32_t>(
      stream, device, logical_endpoint_test::ring_status_words, cuda::no_init);
    auto peer_status = cuda::make_device_buffer<cuda::std::uint32_t>(
      peer_stream, peer_device, logical_endpoint_test::ring_status_words, cuda::no_init);
    stream.sync();
    peer_stream.sync();

    const auto allocation_addr      = reinterpret_cast<cuda::std::uintptr_t>(allocation.data());
    const auto bind_addr            = logical_endpoint_test::align_up(allocation_addr, alignment);
    void* bind_ptr                  = reinterpret_cast<void*>(bind_addr);
    const auto peer_allocation_addr = reinterpret_cast<cuda::std::uintptr_t>(peer_allocation.data());
    const auto peer_bind_addr       = logical_endpoint_test::align_up(peer_allocation_addr, peer_alignment);
    void* peer_bind_ptr             = reinterpret_cast<void*>(peer_bind_addr);
    auto* flag = reinterpret_cast<cuda::std::uint32_t*>(static_cast<cuda::std::uint8_t*>(bind_ptr) + sync_offset);
    auto* peer_flag =
      reinterpret_cast<cuda::std::uint32_t*>(static_cast<cuda::std::uint8_t*>(peer_bind_ptr) + peer_sync_offset);
    cuda::unicast_logical_endpoint endpoint{spec, bytes};
    cuda::unicast_logical_endpoint peer_endpoint{peer_spec, peer_bytes};

    REQUIRE(bind_addr + bytes <= allocation_addr + allocation_bytes);
    REQUIRE(peer_bind_addr + peer_bytes <= peer_allocation_addr + peer_allocation_bytes);
    REQUIRE(endpoint.wait_until_ready(logical_endpoint_test::ready_timeout));
    REQUIRE(peer_endpoint.wait_until_ready(logical_endpoint_test::ready_timeout));
    endpoint.bind(device, 0, bind_ptr, bytes);
    peer_endpoint.bind(peer_device, 0, peer_bind_ptr, peer_bytes);

    cuda::fill_bytes(stream,
                     cuda::std::span<cuda::std::uint8_t>{static_cast<cuda::std::uint8_t*>(bind_ptr),
                                                         sync_offset + logical_endpoint_test::ring_sync_bytes},
                     0);
    cuda::fill_bytes(peer_stream,
                     cuda::std::span<cuda::std::uint8_t>{static_cast<cuda::std::uint8_t*>(peer_bind_ptr),
                                                         peer_sync_offset + logical_endpoint_test::ring_sync_bytes},
                     0);
    cuda::fill_bytes(stream, status, 0);
    cuda::fill_bytes(peer_stream, peer_status, 0);

    auto payload_config = cuda::make_config(
      cuda::make_hierarchy(cuda::grid_dims(logical_endpoint_test::ring_chunk_count), cuda::block_dims<1>()));
    auto signal_config = cuda::make_config(cuda::make_hierarchy(cuda::grid_dims(1), cuda::block_dims<1>()));
    cuda::launch(
      stream,
      payload_config,
      logical_endpoint_test::fabric_ring_put_kernel,
      peer_endpoint,
      cuda::std::uint32_t{0},
      status.data());
    cuda::launch(
      peer_stream,
      payload_config,
      logical_endpoint_test::fabric_ring_put_kernel,
      endpoint,
      cuda::std::uint32_t{1},
      peer_status.data());
    cuda::launch(
      stream,
      signal_config,
      logical_endpoint_test::fabric_ring_signal_flag_kernel,
      peer_endpoint,
      peer_sync_offset,
      flag,
      status.data());
    cuda::launch(
      peer_stream,
      signal_config,
      logical_endpoint_test::fabric_ring_signal_flag_kernel,
      endpoint,
      sync_offset,
      peer_flag,
      peer_status.data());
    stream.sync();
    peer_stream.sync();

    cuda::std::uint32_t host_status[logical_endpoint_test::ring_status_words]{};
    cuda::std::uint32_t peer_host_status[logical_endpoint_test::ring_status_words]{};
    cuda::std::uint32_t observed[logical_endpoint_test::ring_payload_words]{};
    cuda::std::uint32_t peer_observed[logical_endpoint_test::ring_payload_words]{};
    cuda::std::uint32_t observed_flag      = 0;
    cuda::std::uint32_t observed_peer_flag = 0;
    cuda::copy_bytes(
      stream, status, cuda::std::span<cuda::std::uint32_t>{host_status, logical_endpoint_test::ring_status_words});
    cuda::copy_bytes(peer_stream,
                     peer_status,
                     cuda::std::span<cuda::std::uint32_t>{peer_host_status, logical_endpoint_test::ring_status_words});
    cuda::copy_bytes(stream,
                     cuda::std::span<cuda::std::uint32_t>{
                       static_cast<cuda::std::uint32_t*>(bind_ptr), logical_endpoint_test::ring_payload_words},
                     cuda::std::span<cuda::std::uint32_t>{observed, logical_endpoint_test::ring_payload_words});
    cuda::copy_bytes(peer_stream,
                     cuda::std::span<cuda::std::uint32_t>{
                       static_cast<cuda::std::uint32_t*>(peer_bind_ptr), logical_endpoint_test::ring_payload_words},
                     cuda::std::span<cuda::std::uint32_t>{peer_observed, logical_endpoint_test::ring_payload_words});
    cuda::copy_bytes(
      stream, cuda::std::span<cuda::std::uint32_t>{flag, 1}, cuda::std::span<cuda::std::uint32_t>{&observed_flag, 1});
    cuda::copy_bytes(peer_stream,
                     cuda::std::span<cuda::std::uint32_t>{peer_flag, 1},
                     cuda::std::span<cuda::std::uint32_t>{&observed_peer_flag, 1});
    stream.sync();
    peer_stream.sync();

    check_ring_statuses(host_status);
    check_ring_statuses(peer_host_status);
    check_ring_payload(observed, cuda::std::uint32_t{1});
    check_ring_payload(peer_observed, cuda::std::uint32_t{0});
    CHECK(observed_flag == 1);
    CHECK(observed_peer_flag == 1);

    endpoint.unbind(device, 0, bytes);
    peer_endpoint.unbind(peer_device, 0, peer_bytes);
  }
  stream.sync();
  peer_stream.sync();
}

C2H_CCCLRT_TEST("unicast logical endpoint supports programming-guide counted CFT ring", "[logical_endpoint]")
{
  auto [device, peer_device] = two_unicast_test_devices();
  auto spec                  = unicast_spec(device, cuda::logical_endpoint_flag::counted_ops);
  auto peer_spec             = unicast_spec(peer_device, cuda::logical_endpoint_flag::counted_ops);
  skip_if_memory_pools_are_unsupported(device, peer_device);
  skip_if_fabric_ptx_smoke_is_unsupported(device, peer_device);

  auto limits               = validate_logical_endpoint_support(spec, device);
  auto peer_limits          = validate_logical_endpoint_support(peer_spec, peer_device);
  const auto alignment      = limits.bind_alignment;
  const auto peer_alignment = peer_limits.bind_alignment;
  const auto bytes          = smoke_size(limits);
  const auto peer_bytes     = smoke_size(peer_limits);
  const auto counter_offset = logical_endpoint_test::align_up(
    logical_endpoint_test::ring_payload_bytes, logical_endpoint_test::counted_counter_alignment);
  const auto allocation_bytes      = bytes + alignment;
  const auto peer_allocation_bytes = peer_bytes + peer_alignment;
  cuda::stream stream{device};
  cuda::stream peer_stream{peer_device};
  auto resource      = cuda::device_default_memory_pool(device);
  auto peer_resource = cuda::device_default_memory_pool(peer_device);

  REQUIRE(counter_offset + sizeof(cuda::std::uint64_t) <= bytes);
  REQUIRE(counter_offset + sizeof(cuda::std::uint64_t) <= peer_bytes);

  {
    auto allocation = cuda::make_buffer<cuda::std::uint8_t>(stream, resource, allocation_bytes, cuda::no_init);
    auto peer_allocation =
      cuda::make_buffer<cuda::std::uint8_t>(peer_stream, peer_resource, peer_allocation_bytes, cuda::no_init);
    auto status = cuda::make_device_buffer<cuda::std::uint32_t>(
      stream, device, logical_endpoint_test::ring_status_words, cuda::no_init);
    auto peer_status = cuda::make_device_buffer<cuda::std::uint32_t>(
      peer_stream, peer_device, logical_endpoint_test::ring_status_words, cuda::no_init);
    stream.sync();
    peer_stream.sync();

    const auto allocation_addr      = reinterpret_cast<cuda::std::uintptr_t>(allocation.data());
    const auto bind_addr            = logical_endpoint_test::align_up(allocation_addr, alignment);
    void* bind_ptr                  = reinterpret_cast<void*>(bind_addr);
    const auto peer_allocation_addr = reinterpret_cast<cuda::std::uintptr_t>(peer_allocation.data());
    const auto peer_bind_addr       = logical_endpoint_test::align_up(peer_allocation_addr, peer_alignment);
    void* peer_bind_ptr             = reinterpret_cast<void*>(peer_bind_addr);
    auto* counter = reinterpret_cast<cuda::std::uint64_t*>(static_cast<cuda::std::uint8_t*>(bind_ptr) + counter_offset);
    auto* peer_counter =
      reinterpret_cast<cuda::std::uint64_t*>(static_cast<cuda::std::uint8_t*>(peer_bind_ptr) + counter_offset);
    cuda::unicast_logical_endpoint endpoint{spec, bytes};
    cuda::unicast_logical_endpoint peer_endpoint{peer_spec, peer_bytes};

    REQUIRE(bind_addr + bytes <= allocation_addr + allocation_bytes);
    REQUIRE(peer_bind_addr + peer_bytes <= peer_allocation_addr + peer_allocation_bytes);
    REQUIRE(endpoint.wait_until_ready(logical_endpoint_test::ready_timeout));
    REQUIRE(peer_endpoint.wait_until_ready(logical_endpoint_test::ready_timeout));
    endpoint.bind(device, 0, bind_ptr, bytes);
    peer_endpoint.bind(peer_device, 0, peer_bind_ptr, peer_bytes);

    cuda::fill_bytes(stream,
                     cuda::std::span<cuda::std::uint8_t>{
                       static_cast<cuda::std::uint8_t*>(bind_ptr), counter_offset + sizeof(cuda::std::uint64_t)},
                     0);
    cuda::fill_bytes(peer_stream,
                     cuda::std::span<cuda::std::uint8_t>{
                       static_cast<cuda::std::uint8_t*>(peer_bind_ptr), counter_offset + sizeof(cuda::std::uint64_t)},
                     0);
    cuda::fill_bytes(stream, status, 0);
    cuda::fill_bytes(peer_stream, peer_status, 0);

    auto config = cuda::make_config(
      cuda::make_hierarchy(cuda::grid_dims(logical_endpoint_test::ring_chunk_count), cuda::block_dims<1>()));
    cuda::launch(
      stream,
      config,
      logical_endpoint_test::fabric_ring_put_counted_kernel,
      peer_endpoint,
      counter_offset,
      static_cast<cuda::std::uint64_t>(logical_endpoint_test::ring_payload_bytes),
      cuda::std::uint32_t{0},
      counter,
      status.data());
    cuda::launch(
      peer_stream,
      config,
      logical_endpoint_test::fabric_ring_put_counted_kernel,
      endpoint,
      counter_offset,
      static_cast<cuda::std::uint64_t>(logical_endpoint_test::ring_payload_bytes),
      cuda::std::uint32_t{1},
      peer_counter,
      peer_status.data());
    stream.sync();
    peer_stream.sync();

    cuda::std::uint32_t host_status[logical_endpoint_test::ring_status_words]{};
    cuda::std::uint32_t peer_host_status[logical_endpoint_test::ring_status_words]{};
    cuda::std::uint32_t observed[logical_endpoint_test::ring_payload_words]{};
    cuda::std::uint32_t peer_observed[logical_endpoint_test::ring_payload_words]{};
    cuda::std::uint64_t observed_counter      = 0;
    cuda::std::uint64_t observed_peer_counter = 0;
    cuda::copy_bytes(
      stream, status, cuda::std::span<cuda::std::uint32_t>{host_status, logical_endpoint_test::ring_status_words});
    cuda::copy_bytes(peer_stream,
                     peer_status,
                     cuda::std::span<cuda::std::uint32_t>{peer_host_status, logical_endpoint_test::ring_status_words});
    cuda::copy_bytes(stream,
                     cuda::std::span<cuda::std::uint32_t>{
                       static_cast<cuda::std::uint32_t*>(bind_ptr), logical_endpoint_test::ring_payload_words},
                     cuda::std::span<cuda::std::uint32_t>{observed, logical_endpoint_test::ring_payload_words});
    cuda::copy_bytes(peer_stream,
                     cuda::std::span<cuda::std::uint32_t>{
                       static_cast<cuda::std::uint32_t*>(peer_bind_ptr), logical_endpoint_test::ring_payload_words},
                     cuda::std::span<cuda::std::uint32_t>{peer_observed, logical_endpoint_test::ring_payload_words});
    cuda::copy_bytes(stream,
                     cuda::std::span<cuda::std::uint64_t>{counter, 1},
                     cuda::std::span<cuda::std::uint64_t>{&observed_counter, 1});
    cuda::copy_bytes(peer_stream,
                     cuda::std::span<cuda::std::uint64_t>{peer_counter, 1},
                     cuda::std::span<cuda::std::uint64_t>{&observed_peer_counter, 1});
    stream.sync();
    peer_stream.sync();

    check_ring_statuses(host_status);
    check_ring_statuses(peer_host_status);
    check_ring_payload(observed, cuda::std::uint32_t{1});
    check_ring_payload(peer_observed, cuda::std::uint32_t{0});
    CHECK(observed_counter == logical_endpoint_test::ring_payload_bytes);
    CHECK(observed_peer_counter == logical_endpoint_test::ring_payload_bytes);

    endpoint.unbind(device, 0, bytes);
    peer_endpoint.unbind(peer_device, 0, peer_bytes);
  }
  stream.sync();
  peer_stream.sync();
}

C2H_CCCLRT_TEST("multicast logical endpoint adds device and becomes ready", "[logical_endpoint]")
{
  auto [device, peer_device] = multicast_test_devices();
  auto spec                  = multicast_spec(2);

  auto limits = validate_logical_endpoint_support(spec, device, peer_device);
  auto bytes  = smoke_size(limits);

  cuda::multicast_logical_endpoint endpoint{spec, bytes};
  CHECK(endpoint.has_value());
  CHECK(endpoint.size() == bytes);
  endpoint.add_device(device);
  endpoint.add_device(peer_device);
  REQUIRE(endpoint.wait_until_ready(logical_endpoint_test::ready_timeout));
}

C2H_CCCLRT_TEST("multicast logical endpoint honors reported maximum size", "[logical_endpoint]")
{
  auto [device, peer_device] = multicast_test_devices();
  auto spec                  = multicast_spec(2);
  auto limits                = validate_logical_endpoint_support(spec, device, peer_device);

  if (limits.max_size == 0)
  {
    SKIP("no maximum logical endpoint size is reported");
  }

  cuda::multicast_logical_endpoint at_max{spec, limits.max_size};
  at_max.add_device(device);
  at_max.add_device(peer_device);
  REQUIRE(at_max.wait_until_ready(logical_endpoint_test::ready_timeout));

#  if TEST_HAS_EXCEPTIONS()
  REQUIRE(limits.max_size < cuda::std::numeric_limits<cuda::std::uint64_t>::max());
  CHECK_THROWS_AS((cuda::multicast_logical_endpoint{spec, limits.max_size + 1}), std::invalid_argument);
#  endif // TEST_HAS_EXCEPTIONS()
}

C2H_CCCLRT_TEST("multicast logical endpoint release from explicit ID has no retained ID range", "[logical_endpoint]")
{
  auto [device, peer_device] = multicast_test_devices();
  auto spec                  = multicast_spec(2);

  auto limits = validate_logical_endpoint_support(spec, device, peer_device);
  auto bytes  = smoke_size(limits);

  cuda::logical_endpoint_id_range ids{1};
  cuda::multicast_logical_endpoint endpoint{ids[0], spec, bytes};
  CHECK(endpoint.has_value());
  CHECK(endpoint.id() == ids[0]);
  endpoint.add_device(device);
  endpoint.add_device(peer_device);
  REQUIRE(endpoint.wait_until_ready(logical_endpoint_test::ready_timeout));

  auto released = endpoint.release();
  CHECK(!endpoint.has_value());
  CHECK(released.first == ids[0]);
  CHECK(!released.second.has_value());

  destroy_released_endpoint(released.first);
}

C2H_CCCLRT_TEST("multicast logical endpoint binds memory pool allocation by address", "[logical_endpoint]")
{
  auto [device, peer_device] = multicast_test_devices();
  auto spec                  = multicast_spec(2);
  skip_if_memory_pools_are_unsupported(device);

  auto limits                 = validate_logical_endpoint_support(spec, device, peer_device);
  const auto alignment        = limits.bind_alignment;
  const auto bytes            = smoke_size(limits);
  const auto allocation_bytes = bytes + alignment;
  cuda::stream stream{device};
  auto resource = cuda::device_default_memory_pool(device);

  {
    auto allocation = cuda::make_buffer<cuda::std::uint8_t>(stream, resource, allocation_bytes, cuda::no_init);
    stream.sync();
    const auto allocation_addr = reinterpret_cast<cuda::std::uintptr_t>(allocation.data());
    const auto bind_addr       = logical_endpoint_test::align_up(allocation_addr, alignment);
    void* bind_ptr             = reinterpret_cast<void*>(bind_addr);
    cuda::multicast_logical_endpoint endpoint{spec, bytes};

    REQUIRE(bind_addr + bytes <= allocation_addr + allocation_bytes);
    endpoint.add_device(device);
    endpoint.add_device(peer_device);
    REQUIRE(endpoint.wait_until_ready(logical_endpoint_test::ready_timeout));
    cuda::multicast_logical_endpoint_ref ref = endpoint;
    ref.bind(device, 0, bind_ptr, bytes);
    ref.unbind(device, 0, bytes);
  }
  stream.sync();
}

C2H_CCCLRT_TEST("multicast logical endpoint supports device-side fabric put to all bound devices", "[logical_endpoint]")
{
  if (cuda::devices.size() < 2)
  {
    SKIP("multicast logical endpoint fabric PTX smoke requires at least two devices");
  }

  cuda::device_ref device{0};
  cuda::device_ref peer_device{1};
  auto spec = multicast_spec(2);
  skip_if_memory_pools_are_unsupported(device, peer_device);
  skip_if_fabric_ptx_smoke_is_unsupported(device, peer_device);

  auto limits                 = validate_logical_endpoint_support(spec, device, peer_device);
  const auto alignment        = limits.bind_alignment;
  const auto bytes            = smoke_size(limits);
  const auto allocation_bytes = bytes + alignment;
  cuda::stream stream{device};
  cuda::stream peer_stream{peer_device};
  auto resource      = cuda::device_default_memory_pool(device);
  auto peer_resource = cuda::device_default_memory_pool(peer_device);

  {
    auto allocation = cuda::make_buffer<cuda::std::uint8_t>(stream, resource, allocation_bytes, cuda::no_init);
    auto peer_allocation =
      cuda::make_buffer<cuda::std::uint8_t>(peer_stream, peer_resource, allocation_bytes, cuda::no_init);
    auto status = cuda::make_device_buffer<cuda::std::uint32_t>(stream, device, 1, cuda::no_init);
    stream.sync();
    peer_stream.sync();

    const auto allocation_addr      = reinterpret_cast<cuda::std::uintptr_t>(allocation.data());
    const auto bind_addr            = logical_endpoint_test::align_up(allocation_addr, alignment);
    void* bind_ptr                  = reinterpret_cast<void*>(bind_addr);
    const auto peer_allocation_addr = reinterpret_cast<cuda::std::uintptr_t>(peer_allocation.data());
    const auto peer_bind_addr       = logical_endpoint_test::align_up(peer_allocation_addr, alignment);
    void* peer_bind_ptr             = reinterpret_cast<void*>(peer_bind_addr);
    cuda::multicast_logical_endpoint endpoint{spec, bytes};

    REQUIRE(bind_addr + bytes <= allocation_addr + allocation_bytes);
    REQUIRE(peer_bind_addr + bytes <= peer_allocation_addr + allocation_bytes);
    endpoint.add_device(device);
    endpoint.add_device(peer_device);
    REQUIRE(endpoint.wait_until_ready(logical_endpoint_test::ready_timeout));
    endpoint.bind(device, 0, bind_ptr, bytes);
    endpoint.bind(peer_device, 0, peer_bind_ptr, bytes);

    cuda::fill_bytes(stream,
                     cuda::std::span<cuda::std::uint8_t>{
                       static_cast<cuda::std::uint8_t*>(bind_ptr), logical_endpoint_test::payload_bytes},
                     0);
    cuda::fill_bytes(peer_stream,
                     cuda::std::span<cuda::std::uint8_t>{
                       static_cast<cuda::std::uint8_t*>(peer_bind_ptr), logical_endpoint_test::payload_bytes},
                     0);
    cuda::fill_bytes(stream, status, 0);
    stream.sync();
    peer_stream.sync();

    auto config = cuda::make_config(cuda::make_hierarchy(cuda::grid_dims(1), cuda::block_dims<1>()));
    cuda::launch(
      stream,
      config,
      logical_endpoint_test::fabric_try_put_multimem_smoke_kernel,
      endpoint,
      cuda::std::uint64_t{0},
      status.data());
    stream.sync();

    cuda::std::uint32_t host_status = 0;
    cuda::std::uint32_t observed[logical_endpoint_test::payload_words]{};
    cuda::std::uint32_t peer_observed[logical_endpoint_test::payload_words]{};
    cuda::copy_bytes(stream, status, cuda::std::span<cuda::std::uint32_t>{&host_status, 1});
    cuda::copy_bytes(stream,
                     cuda::std::span<cuda::std::uint32_t>{
                       static_cast<cuda::std::uint32_t*>(bind_ptr), logical_endpoint_test::payload_words},
                     cuda::std::span<cuda::std::uint32_t>{observed, logical_endpoint_test::payload_words});
    cuda::copy_bytes(peer_stream,
                     cuda::std::span<cuda::std::uint32_t>{
                       static_cast<cuda::std::uint32_t*>(peer_bind_ptr), logical_endpoint_test::payload_words},
                     cuda::std::span<cuda::std::uint32_t>{peer_observed, logical_endpoint_test::payload_words});
    stream.sync();
    peer_stream.sync();

    CHECK(host_status == logical_endpoint_test::status_success);
    CHECK(observed[0] == 0x13572468u);
    CHECK(observed[1] == 0x24681357u);
    CHECK(observed[2] == 0xdeadbeefu);
    CHECK(observed[3] == 0xcafef00du);
    CHECK(peer_observed[0] == 0x13572468u);
    CHECK(peer_observed[1] == 0x24681357u);
    CHECK(peer_observed[2] == 0xdeadbeefu);
    CHECK(peer_observed[3] == 0xcafef00du);

    endpoint.unbind(device, 0, bytes);
    endpoint.unbind(peer_device, 0, bytes);
  }
  stream.sync();
  peer_stream.sync();
}

#else // ^^^ _CCCL_CTK_AT_LEAST(13, 3) ^^^ / vvv _CCCL_CTK_BELOW(13, 3)

C2H_TEST("logical endpoint tests require CTK 13.3", "[logical_endpoint]")
{
  SUCCEED("logical endpoints require CTK 13.3 headers");
}

#endif // _CCCL_CTK_BELOW(13, 3)
