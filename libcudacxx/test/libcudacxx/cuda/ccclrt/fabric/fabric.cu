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

#if _CCCL_CUDACC_AT_LEAST(13, 4) && __cccl_ptx_isa >= 940

#  include "fabric_test_helper.h"

namespace
{
using namespace fabric_test;

// Transfer the first eight words as 3..10; values straddle the initial 8 to exercise both min/max outcomes.
// Remaining payload words retain their initial values, detecting writes beyond the requested range.
constexpr cuda::std::size_t transfer_elements = transfer_bytes / sizeof(uint32_t);
constexpr uint32_t source_base_value          = 3;
constexpr uint32_t peer_initial_value         = 12;
// Both the data and flag requests in the ordering test transfer four words (16 bytes).
constexpr cuda::std::size_t data_flag_elements = 4;
// Repeat a byte mask that copies four bytes and skips four: only even-indexed words are updated.
[[maybe_unused]] constexpr uint16_t alternating_word_mask = 0x0f0f;

enum class transfer_kind
{
  put,
  counted_put,
  masked_put,
  get,
  reduce_add,
  reduce_min,
  reduce_max,
  counted_reduce
};

template <transfer_kind Kind>
struct transfer_kernel : fabric_case<uint32_t>
{
  static constexpr bool counted = Kind == transfer_kind::counted_put || Kind == transfer_kind::counted_reduce;

  TEST_DEVICE_FUNC static uint32_t expected_value(cuda::std::size_t i, int = 0)
  {
    auto old_value = initial_value(i);
    // Get only reads the endpoint; other requests leave words beyond the transfer unchanged.
    if (i >= transfer_elements || Kind == transfer_kind::get)
    {
      return old_value;
    }
    uint32_t source_value = source_base_value + i;
    if constexpr (Kind == transfer_kind::reduce_add || Kind == transfer_kind::counted_reduce)
    {
      return old_value + source_value;
    }
    else if constexpr (Kind == transfer_kind::reduce_min)
    {
      return source_value < old_value ? source_value : old_value;
    }
    else if constexpr (Kind == transfer_kind::reduce_max)
    {
      return source_value > old_value ? source_value : old_value;
    }
    else if constexpr (Kind == transfer_kind::masked_put)
    {
      // The byte mask selects even words; odd words keep the initial endpoint value.
      return i % 2 == 0 ? source_value : old_value;
    }
    else
    {
      return source_value;
    }
  }

  template <class Config>
  TEST_DEVICE_FUNC void operator()(Config, cuda::unicast_logical_endpoint_ref endpoint) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_100, {
      __shared__ alignas(16) uint32_t staging[transfer_elements];
      auto& bar = make_barrier();
      for (cuda::std::size_t i = 0; i < transfer_elements; ++i)
      {
        staging[i] = source_base_value + i;
      }
      cuda::fence_proxy_async<cuda::proxy_async_space::shared_cta>();
      if constexpr (Kind == transfer_kind::put)
      {
        // Replace the first eight endpoint words (all 8) with 3..10.
        cuda::fabric::try_put(endpoint, data_offset, staging, sizeof(staging), bar);
      }
      else if constexpr (Kind == transfer_kind::counted_put)
      {
        // Store 3..10 and increase the remote byte counter by the 32 transferred bytes.
        cuda::fabric::try_put_counted(endpoint, data_offset, counter_offset, staging, sizeof(staging), bar);
      }
      else if constexpr (Kind == transfer_kind::masked_put)
      {
        // Store 3, 5, 7, 9 in even words; odd words retain 8.
        cuda::fabric::try_put_masked(endpoint, data_offset, staging, sizeof(staging), alternating_word_mask, bar);
      }
      else if constexpr (Kind == transfer_kind::get)
      {
        // Read eight endpoint words, all 8, into staging without changing the endpoint.
        cuda::fabric::try_get(endpoint, data_offset, staging, sizeof(staging), bar);
      }
      else if constexpr (Kind == transfer_kind::reduce_add)
      {
        // Add 3..10 to the initial 8, producing 11..18.
        cuda::fabric::try_reduce_add(endpoint, data_offset, staging, sizeof(staging), bar);
      }
      else if constexpr (Kind == transfer_kind::reduce_min)
      {
        // min(8, 3..10) produces 3, 4, 5, 6, 7, 8, 8, 8.
        cuda::fabric::try_reduce_min(endpoint, data_offset, staging, sizeof(staging), bar);
      }
      else if constexpr (Kind == transfer_kind::reduce_max)
      {
        // max(8, 3..10) produces 8, 8, 8, 8, 8, 8, 9, 10.
        cuda::fabric::try_reduce_max(endpoint, data_offset, staging, sizeof(staging), bar);
      }
      else
      {
        // Produce 11..18 and increase the remote byte counter by 32.
        cuda::fabric::try_reduce_counted_add(endpoint, data_offset, counter_offset, staging, sizeof(staging), bar);
      }

      if constexpr (Kind == transfer_kind::get)
      {
        cuda::fabric::submit_restrict_fetching();
      }
      else
      {
        cuda::fabric::submit();
      }
      if constexpr (Kind == transfer_kind::put)
      {
        // Overwrite staging before remote completion; wait_reads protects its lifetime.
        cuda::fabric::wait_reads();
        for (auto& value : staging)
        {
          value = 0;
        }
      }
      wait_success(bar, Kind == transfer_kind::get ? sizeof(staging) : sizeof(staging) / 16);
      if constexpr (Kind == transfer_kind::get)
      {
        for (auto value : staging)
        {
          CCCLRT_CHECK_DEVICE(value == initial_payload_value);
        }
      }
    })
  }
};

struct data_flag_kernel : fabric_case<uint32_t>
{
  static constexpr bool signaled = true;

  TEST_DEVICE_FUNC static uint32_t expected_value(cuda::std::size_t i, int = 0)
  {
    // Only the first four payload words are written; the flag lives in a separate region.
    return i < data_flag_elements ? uint32_t(source_base_value + i) : initial_value(i);
  }

  template <class Config>
  TEST_DEVICE_FUNC void operator()(Config, cuda::unicast_logical_endpoint_ref endpoint) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_100, {
      __shared__ alignas(16) uint32_t data[data_flag_elements];
      __shared__ alignas(16) uint32_t flag[data_flag_elements];
      auto& bar = make_barrier();
      for (cuda::std::size_t i = 0; i < data_flag_elements; ++i)
      {
        data[i] = source_base_value + i;
        flag[i] = i == 0 ? 1 : 0;
      }
      cuda::fence_proxy_async<cuda::proxy_async_space::shared_cta>();
      // Replace the first four payload words with 3, 4, 5, 6.
      cuda::fabric::try_put(endpoint, data_offset, data, sizeof(data), bar);
      cuda::fabric::submit();
      wait_success(bar, 1);

      cuda::fence_proxy_generic_fabric_alias<cuda::memory_order_release>();
      // After data completion, publish the separate flag as {1, 0, 0, 0}.
      cuda::fabric::try_put(endpoint, flag_offset, flag, sizeof(flag), bar);
      cuda::fabric::submit();
      wait_success(bar, 1);
    })
  }
};

template <transfer_kind Kind>
void test_transfer()
{
  run_unicast(transfer_kernel<Kind>{});
}

enum class multicast_kind
{
  put,
  counted_put,
  masked_put,
  add,
  min,
  max,
  counted_add,
  pull_add,
  pull_min,
  pull_max
};

template <multicast_kind Kind>
struct multicast_kernel : fabric_case<uint32_t>
{
  TEST_DEVICE_FUNC static uint32_t initial_value(cuda::std::size_t, int device_index = 0)
  {
    // Distinct per-device values make pull-add/min/max distinguishable: 20, 8, and 12.
    return device_index == 0 ? initial_payload_value : peer_initial_value;
  }

  TEST_DEVICE_FUNC static uint32_t expected_value(cuda::std::size_t i, int device_index = 0)
  {
    uint32_t old_value = initial_value(i, device_index);
    // Pull reductions read both endpoints without changing either one's payload.
    if (i >= transfer_elements || Kind == multicast_kind::pull_add || Kind == multicast_kind::pull_min
        || Kind == multicast_kind::pull_max)
    {
      return old_value;
    }
    uint32_t source_value = source_base_value + i;
    if constexpr (Kind == multicast_kind::add || Kind == multicast_kind::counted_add)
    {
      return old_value + source_value;
    }
    else if constexpr (Kind == multicast_kind::min)
    {
      return source_value < old_value ? source_value : old_value;
    }
    else if constexpr (Kind == multicast_kind::max)
    {
      return source_value > old_value ? source_value : old_value;
    }
    else if constexpr (Kind == multicast_kind::masked_put)
    {
      // The same alternating-word mask is applied independently to both endpoint bindings.
      return i % 2 == 0 ? source_value : old_value;
    }
    else
    {
      return source_value;
    }
  }

  template <class Config>
  TEST_DEVICE_FUNC void operator()(Config config, cuda::multicast_logical_endpoint_ref endpoint) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_100, {
      constexpr bool pull =
        Kind == multicast_kind::pull_add || Kind == multicast_kind::pull_min || Kind == multicast_kind::pull_max;
      static_assert(cuda::gpu_thread.count(cuda::block, config) == 32);
      const auto rank = cuda::gpu_thread.rank(cuda::block, config);
      __shared__ alignas(16) uint32_t staging[transfer_elements];
      auto& bar = make_barrier(rank);
      if (rank < transfer_elements)
      {
        staging[rank] = source_base_value + rank;
      }
      __syncthreads();

      if constexpr (pull)
      {
        // pullred.sync is a full-warp collective with identical operands.
        cuda::fence_proxy_async<cuda::proxy_async_space::shared_cta>();
        if constexpr (Kind == multicast_kind::pull_add)
        {
          // Each staging word receives 8 + 12 = 20; both endpoint payloads stay unchanged.
          cuda::fabric::try_pull_reduce_add(endpoint, data_offset, staging, sizeof(staging), bar);
        }
        else if constexpr (Kind == multicast_kind::pull_min)
        {
          // Each staging word receives min(8, 12) = 8.
          cuda::fabric::try_pull_reduce_min(endpoint, data_offset, staging, sizeof(staging), bar);
        }
        else
        {
          // Each staging word receives max(8, 12) = 12.
          cuda::fabric::try_pull_reduce_max(endpoint, data_offset, staging, sizeof(staging), bar);
        }
        cuda::fabric::submit_restrict_fetching();
      }
      else if (rank == 0)
      {
        cuda::fence_proxy_async<cuda::proxy_async_space::shared_cta>();
        if constexpr (Kind == multicast_kind::put)
        {
          // Store 3..10 in the first eight words on both devices.
          cuda::fabric::try_put(endpoint, data_offset, staging, sizeof(staging), bar);
        }
        else if constexpr (Kind == multicast_kind::counted_put)
        {
          // Store 3..10 on both devices and increase each byte counter by 32.
          cuda::fabric::try_put_counted(endpoint, data_offset, counter_offset, staging, sizeof(staging), bar);
        }
        else if constexpr (Kind == multicast_kind::masked_put)
        {
          // Store 3, 5, 7, 9 in even words; odd words retain their device's initial 8 or 12.
          cuda::fabric::try_put_masked(endpoint, data_offset, staging, sizeof(staging), alternating_word_mask, bar);
        }
        else if constexpr (Kind == multicast_kind::add)
        {
          // Adding 3..10 produces 11..18 on device 0 and 15..22 on device 1.
          cuda::fabric::try_reduce_add(endpoint, data_offset, staging, sizeof(staging), bar);
        }
        else if constexpr (Kind == multicast_kind::min)
        {
          // Clamp 3..10 to each initial value: device 0 caps at 8; device 1 keeps 3..10.
          cuda::fabric::try_reduce_min(endpoint, data_offset, staging, sizeof(staging), bar);
        }
        else if constexpr (Kind == multicast_kind::max)
        {
          // Raise each initial value to at least 3..10: device 0 reaches 10; device 1 stays 12.
          cuda::fabric::try_reduce_max(endpoint, data_offset, staging, sizeof(staging), bar);
        }
        else
        {
          // Produce 11..18 and 15..22 on the two devices and increase each byte counter by 32.
          cuda::fabric::try_reduce_counted_add(endpoint, data_offset, counter_offset, staging, sizeof(staging), bar);
        }
        cuda::fabric::submit();
      }
      __syncthreads();
      if (rank == 0)
      {
        wait_success(bar, pull ? sizeof(staging) : sizeof(staging) / 16);
        if constexpr (pull)
        {
          constexpr uint32_t expected =
            Kind == multicast_kind::pull_add
              ? initial_payload_value + peer_initial_value
              : (Kind == multicast_kind::pull_min ? initial_payload_value : peer_initial_value);
          for (auto value : staging)
          {
            CCCLRT_CHECK_DEVICE(value == expected);
          }
        }
      }
      __syncthreads();
    })
  }
};

template <multicast_kind Kind>
void test_multicast()
{
  require_runtime_support();
  constexpr bool counted = Kind == multicast_kind::counted_put || Kind == multicast_kind::counted_add;
  using kernel           = multicast_kernel<Kind>;
  cuda::device_ref device{0};
  cuda::device_ref peer{1};
  auto spec = cuda::multicast_logical_endpoint_spec{
    2,
    counted ? cuda::logical_endpoint_flag::counted_ops : cuda::logical_endpoint_flag::none,
    cuda::logical_endpoint_ipc_handle_type::none};
  auto limits = require_fabric_support(spec, device, peer);

  cuda::stream stream{device};
  cuda::stream peer_stream{peer};
  endpoint_storage storage{stream, device, limits};
  endpoint_storage peer_storage{peer_stream, peer, limits};
  auto payload      = make_payload<uint32_t>(storage);
  auto peer_payload = make_payload<uint32_t>(peer_storage);
  cuda::multicast_logical_endpoint endpoint{spec, storage.bound.size()};
  endpoint.add_device(device);
  endpoint.add_device(peer);
  REQUIRE(endpoint.wait_ready_for(logical_endpoint_test::ready_timeout));
  endpoint.bind(device, 0, storage.bound.data(), storage.bound.size());
  endpoint.bind(peer, 0, peer_storage.bound.data(), peer_storage.bound.size());

  auto config = cuda::make_config(cuda::make_hierarchy(cuda::grid_dims<1>(), cuda::block_dims<1>()));
  cuda::launch(stream, config, initialize_payload<kernel>{0}, payload);
  cuda::launch(peer_stream, config, initialize_payload<kernel>{1}, peer_payload);
  stream.sync();
  peer_stream.sync();

  auto warp_config = cuda::make_config(cuda::make_hierarchy(cuda::grid_dims<1>(), cuda::block_dims<32>()));
  cuda::launch(stream, warp_config, kernel{}, endpoint);
  stream.sync();

  constexpr uint64_t expected_counter = counted ? transfer_bytes : 0;
  cuda::launch(stream, config, validate_payload<kernel>{expected_counter, false, 0}, payload);
  cuda::launch(peer_stream, config, validate_payload<kernel>{expected_counter, false, 1}, peer_payload);
  stream.sync();
  peer_stream.sync();
  endpoint.unbind(device, 0, storage.bound.size());
  endpoint.unbind(peer, 0, peer_storage.bound.size());
}
} // namespace

C2H_CCCLRT_TEST("direct fabric unicast transfers and reductions", "[fabric]")
{
  test_transfer<transfer_kind::put>();
  test_transfer<transfer_kind::counted_put>();
  test_transfer<transfer_kind::masked_put>();
  test_transfer<transfer_kind::get>();
  test_transfer<transfer_kind::reduce_add>();
  test_transfer<transfer_kind::reduce_min>();
  test_transfer<transfer_kind::reduce_max>();
  test_transfer<transfer_kind::counted_reduce>();
}

C2H_CCCLRT_TEST("direct fabric multicast transfers and warp-collective pull reductions", "[fabric]")
{
  test_multicast<multicast_kind::put>();
  test_multicast<multicast_kind::counted_put>();
  test_multicast<multicast_kind::masked_put>();
  test_multicast<multicast_kind::add>();
  test_multicast<multicast_kind::min>();
  test_multicast<multicast_kind::max>();
  test_multicast<multicast_kind::counted_add>();
  test_multicast<multicast_kind::pull_add>();
  test_multicast<multicast_kind::pull_min>();
  test_multicast<multicast_kind::pull_max>();
}

C2H_CCCLRT_TEST("direct fabric data completion then alias fence then flag", "[fabric]")
{
  run_unicast(data_flag_kernel{});
}

#else
C2H_TEST("direct fabric transfers runtime tests require CUDA 13.4", "[fabric]")
{
  SKIP("fabric operations and shared_barrier require CUDA 13.4");
}
#endif
