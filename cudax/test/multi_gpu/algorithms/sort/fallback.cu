// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <nccl.h>
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 2) && __has_include(<nccl_device.h>)
#  include <nccl_device.h>

namespace fallback_test
{
enum class capability
{
  available,
  missing_device,
  missing_device_on_one_rank,
  query_failure,
  missing_gin
};
// Set before sort starts its rank threads, and unchanged until they all join.
inline capability mode = capability::missing_device;

inline ncclResult_t query_properties(ncclComm_t comm, ncclCommProperties* properties)
{
  if (mode == capability::query_failure)
  {
    return ncclInvalidUsage;
  }
  const auto result = ncclCommQueryProperties(comm, properties);
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
  return result;
}

inline ncclTeam_t lsa_team(ncclComm_t comm)
{
  auto team = ncclTeamLsa(comm);
  if (mode == capability::missing_gin)
  {
    team.nRanks = 1;
  }
  return team;
}
} // namespace fallback_test

// Interpose only the capability probes in sort. All collectives, transfers,
// streams, and input/output checks still use the real NCCL/CUDA implementation.
#  define ncclCommQueryProperties fallback_test::query_properties
#  define ncclTeamLsa             fallback_test::lsa_team
#endif
#include <cuda/experimental/mgmn/sort>
#undef ncclCommQueryProperties
#undef ncclTeamLsa

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
  for (auto mode :
       {fallback_test::capability::available,
        fallback_test::capability::missing_device,
        fallback_test::capability::missing_device_on_one_rank,
        fallback_test::capability::query_failure,
        fallback_test::capability::available,
        fallback_test::capability::missing_gin,
        fallback_test::capability::available})
  {
    if (mode == fallback_test::capability::missing_gin)
    {
      // This probe simulates a different transport. Drop any real-transport
      // association first; the other modes exercise fallback with a live cache.
      for (auto& context : contexts)
      {
        context->reset();
      }
    }
    fallback_test::mode = mode;
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
