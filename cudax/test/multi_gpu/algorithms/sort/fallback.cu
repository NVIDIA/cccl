// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/experimental/mgmn/__nccl/device_api.h>

#include <atomic>

#include "nccl_device_headers.cuh"

#if defined(_CUDAX_HAS_NCCL_DEVICE_API)
namespace fallback_test
{
enum class capability
{
  available,
  missing_symbols,
  missing_device,
  missing_device_on_one_rank,
  query_failure,
  missing_gin,
  missing_gin_symbols
};
// Set before sort starts its rank threads, and unchanged until they all join.
inline capability mode = capability::available;
inline std::atomic<int> query_calls{0};
inline cuda::experimental::__nccl::__device_api api;

inline ncclResult_t query_properties(ncclComm_t comm, ncclCommProperties* properties)
{
  query_calls.fetch_add(1, std::memory_order_relaxed);
  if (mode == capability::query_failure)
  {
    return ncclInvalidUsage;
  }
  const auto result = cuda::experimental::__nccl::__get_device_api().__query(comm, properties);
  if (result != ncclSuccess)
  {
    return result;
  }
  int rank = 0, size = 0;
  ncclCommUserRank(comm, &rank);
  ncclCommCount(comm, &size);
  if (mode == capability::missing_device || (mode == capability::missing_device_on_one_rank && rank == size - 1))
  {
    properties->deviceApiSupport = 0;
  }
  if (mode == capability::missing_gin)
  {
    properties->ginType = NCCL_GIN_TYPE_NONE;
  }
  if (mode == capability::missing_gin_symbols)
  {
    properties->ginType = NCCL_GIN_TYPE_PROXY;
  }
  return result;
}

inline ncclTeam_t lsa_team(ncclComm_t comm)
{
  auto team = cuda::experimental::__nccl::__get_device_api().__lsa(comm);
  if (mode == capability::missing_gin || mode == capability::missing_gin_symbols)
  {
    team.nRanks = 1;
  }
  return team;
}

inline void configure(capability value)
{
  mode = value;
  api  = cuda::experimental::__nccl::__get_device_api();
  if (api.__query)
  {
    api.__query = query_properties;
  }
  if (api.__lsa)
  {
    api.__lsa = lsa_team;
  }
  if (mode == capability::missing_symbols)
  {
    api.__create = nullptr;
  }
  if (mode == capability::missing_gin_symbols)
  {
    api.__gin_barrier = nullptr;
  }
}
} // namespace fallback_test

namespace cuda::experimental::__nccl
{
inline const __device_api& __get_test_device_api()
{
  return ::fallback_test::api;
}
} // namespace cuda::experimental::__nccl

// Replace the optional API view only in sort. The loader, transfers, streams,
// and validation still use the real implementation. Probe counters and fresh
// context checks below ensure this interposition cannot silently stop working.
#  define __get_device_api __get_test_device_api
#endif
#include <cuda/experimental/mgmn/sort>
#undef __get_device_api

#include <cuda/std/functional>

#include <algorithm>
#include <memory>
#include <vector>

#include <nccl_test_common.h>
#include <testing.cuh>

#include "sort_common.cuh"
#include <c2h/catch2_test_helper.h>

MULTI_GPU_TEST("sort, collective host fallback when optional device capabilities are unavailable")
{
#if defined(_CUDAX_SORT_HAS_NCCL_DEVICE)
  auto comms = this->communicators();
  if (comms.size() < 2)
  {
    SKIP("Collective capability fallback requires at least two GPUs");
  }
  auto owned_streams = nccl_test_util::make_streams();
  std::vector<cuda::stream_ref> streams{owned_streams.begin(), owned_streams.end()};
  std::vector<std::unique_ptr<cudax::mgmn::sort_context>> contexts;
  for (cuda::std::size_t rank = 0; rank < comms.size(); ++rank)
  {
    contexts.emplace_back(new cudax::mgmn::sort_context);
  }
  using Env = decltype(contexts[0]->bind(streams[0]));
  std::vector<Env> envs;
  for (cuda::std::size_t rank = 0; rank < comms.size(); ++rank)
  {
    envs.push_back(contexts[rank]->bind(streams[rank]));
  }
  const auto& native_api = cuda::experimental::__nccl::__get_device_api();
  bool device_supported  = native_api.__supports_lsa();
  if (device_supported)
  {
    for (auto comm : comms)
    {
      _CCCL_DIAG_PUSH
      _CCCL_DIAG_SUPPRESS_GCC("-Wmissing-field-initializers")
      _CCCL_DIAG_SUPPRESS_CLANG("-Wmissing-field-initializers")
      ncclCommProperties properties = NCCL_COMM_PROPERTIES_INITIALIZER;
      _CCCL_DIAG_POP
      device_supported &=
        native_api.__query(comm.native_handle(), &properties) == ncclSuccess && properties.deviceApiSupport
        && (native_api.__lsa(comm.native_handle()).nRanks == comm.size()
            || (native_api.__supports_gin() && properties.ginType != NCCL_GIN_TYPE_NONE));
    }
  }
  for (auto mode :
       {fallback_test::capability::missing_symbols,
        fallback_test::capability::missing_device,
        fallback_test::capability::missing_device_on_one_rank,
        fallback_test::capability::available,
        fallback_test::capability::missing_device,
        fallback_test::capability::missing_device_on_one_rank,
        fallback_test::capability::query_failure,
        fallback_test::capability::available,
        fallback_test::capability::missing_gin,
        fallback_test::capability::missing_gin_symbols,
        fallback_test::capability::available})
  {
    if (mode == fallback_test::capability::missing_gin || mode == fallback_test::capability::missing_gin_symbols)
    {
      // This probe simulates a different transport. Drop any real-transport
      // association first; the other modes exercise fallback with a live cache.
      for (auto& context : contexts)
      {
        context->reset();
      }
    }
    fallback_test::configure(mode);
    CAPTURE(int(mode));
    // Reuse the same context across backend decisions, ascending/descending
    // calls, ragged counts, and an empty rank. Guards are outside the subrange.
    for (int iteration = 0; iteration < 3; ++iteration)
    {
      CAPTURE(iteration);
      constexpr int guard = 123456789;
      std::vector<std::vector<int>> host_inputs(comms.size());
      std::vector<int> expected, counts;
      for (cuda::std::size_t rank = 0; rank < comms.size(); ++rank)
      {
        const int count = iteration == 2 && rank == 0 ? 0 : 127 + int(rank) * 19;
        counts.push_back(count);
        host_inputs[rank].push_back(guard);
        for (int index = 0; index < count; ++index)
        {
          const int key = (index * 31 + int(rank) * 7) % 53 - 26;
          host_inputs[rank].push_back(key);
          expected.push_back(key);
        }
        host_inputs[rank].push_back(guard);
      }
      auto buffers = sort_test_util::make_device_inputs(comms, streams, host_inputs);
      std::vector<int*> inputs;
      for (auto& buffer : buffers)
      {
        inputs.push_back(buffer.data() + 1);
      }
      std::vector<bool> had_window;
      for (const auto& context : contexts)
      {
        had_window.push_back(bool(context->__window()));
      }
      fallback_test::query_calls.store(0, std::memory_order_relaxed);
      if (iteration == 1)
      {
        std::sort(expected.begin(), expected.end(), cuda::std::greater<>{});
        cudax::mgmn::sort(cudax::distributed, comms, envs, inputs, counts, cuda::std::greater<>{});
      }
      else
      {
        std::sort(expected.begin(), expected.end());
        cudax::mgmn::sort(cudax::distributed, comms, envs, inputs, counts);
      }
      const int expected_queries =
        native_api.__supports_lsa() && mode != fallback_test::capability::missing_symbols ? int(comms.size()) : 0;
      REQUIRE(fallback_test::query_calls.load(std::memory_order_relaxed) == expected_queries);
      for (cuda::std::size_t rank = 0; rank < comms.size(); ++rank)
      {
        if (mode == fallback_test::capability::available && device_supported)
        {
          REQUIRE(bool(contexts[rank]->__window()));
        }
        else if (!had_window[rank])
        {
          REQUIRE_FALSE(bool(contexts[rank]->__window()));
        }
      }
      const auto actual        = sort_test_util::gather_outputs(comms, buffers);
      cuda::std::size_t offset = 0, expected_offset = 0;
      for (int count : counts)
      {
        REQUIRE(actual[offset] == guard);
        REQUIRE(actual[offset + count + 1] == guard);
        REQUIRE(std::equal(
          actual.begin() + offset + 1, actual.begin() + offset + count + 1, expected.begin() + expected_offset));
        offset += count + 2;
        expected_offset += count;
      }
    }
  }
#else
  SKIP("Capability-probe regression requires NCCL device headers");
#endif
}
