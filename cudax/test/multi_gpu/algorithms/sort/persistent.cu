// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/std/functional>
#include <cuda/std/type_traits>

#include <cuda/experimental/mgmn/sort>

#include <algorithm>
#include <memory>
#include <vector>

#include <algorithm_common.h>
#include <nccl_test_common.h>
#include <testing.cuh>

#include "sort_common.cuh"
#include <c2h/catch2_test_helper.h>

#if defined(_CUDAX_SORT_HAS_NCCL_DEVICE)
static_assert(!cuda::std::is_copy_constructible_v<cudax::mgmn::sort_context>);
static_assert(!cuda::std::is_move_constructible_v<cudax::mgmn::sort_context>);
using persistent_types = c2h::type_list<int, cuda::std::int64_t>;

MULTI_GPU_TEST("sort, persistent context growth, reuse, and rank association", persistent_types)
{
  using T    = typename c2h::get<0, TestType>;
  auto comms = this->communicators();
  if (comms.size() < 2)
  {
    SKIP("Persistent device communication requires at least two GPUs");
  }
  for (auto comm : comms)
  {
    _CCCL_DIAG_PUSH
    _CCCL_DIAG_SUPPRESS_GCC("-Wmissing-field-initializers")
    _CCCL_DIAG_SUPPRESS_CLANG("-Wmissing-field-initializers")
    ncclCommProperties properties = NCCL_COMM_PROPERTIES_INITIALIZER;
    _CCCL_DIAG_POP
    REQUIRE(ncclCommQueryProperties(comm.native_handle(), &properties) == ncclSuccess);
    if (!properties.deviceApiSupport
        || (ncclTeamLsa(comm.native_handle()).nRanks != comm.size() && properties.ginType == NCCL_GIN_TYPE_NONE))
    {
      SKIP("The communicator does not support the required NCCL device transport");
    }
  }
  auto first_streams  = nccl_test_util::make_streams();
  auto second_streams = nccl_test_util::make_streams();
  std::vector<std::unique_ptr<cudax::mgmn::sort_context>> contexts;
  for (cuda::std::size_t rank = 0; rank < comms.size(); ++rank)
  {
    contexts.emplace_back(new cudax::mgmn::sort_context);
  }
  using Env = decltype(contexts[0]->bind(cuda::stream_ref(first_streams[0])));
  // Grow, shrink, skip all-empty calls, and reset a single rank. Alternating
  // streams exercises ownership independently of any one stream's lifetime.
  for (int iteration = 0; iteration < 12; ++iteration)
  {
    CAPTURE(iteration);
    if (iteration == 8)
    {
      contexts.back()->reset();
    }
    std::vector<cuda::stream_ref> streams;
    std::vector<Env> envs;
    std::vector<std::vector<T>> host_inputs(comms.size());
    std::vector<T> expected;
    std::vector<int> counts;
    constexpr T guard = T{123456789};
    for (cuda::std::size_t rank = 0; rank < comms.size(); ++rank)
    {
      streams.emplace_back(iteration % 2 == 0 ? first_streams[rank] : second_streams[rank]);
      envs.push_back(contexts[rank]->bind(streams.back()));
      REQUIRE(cuda::get_stream(envs.back()) == streams.back());
      int count = iteration == 2 ? 16381 + int(rank) * 19 : 73 + int(rank) * 17;
      if (iteration == 4 || (iteration == 6 && rank % 2 == 0))
      {
        count = 0;
      }
      counts.push_back(count);
      auto& local = host_inputs[rank];
      local.push_back(guard);
      for (int index = 0; index < count; ++index)
      {
        const T key = iteration == 7 ? T{7} : T{(index * 917 + int(rank) * 61) % 8191 - 4095};
        local.push_back(key);
        expected.push_back(key);
      }
      local.push_back(guard);
    }
    auto buffers = sort_test_util::make_device_inputs(comms, streams, host_inputs);
    std::vector<T*> inputs;
    for (auto& buffer : buffers)
    {
      inputs.push_back(buffer.data() + 1);
    }
    if (iteration % 2 == 0)
    {
      std::sort(expected.begin(), expected.end());
      cudax::mgmn::sort(cudax::distributed, comms, envs, inputs, counts);
    }
    else
    {
      std::sort(expected.begin(), expected.end(), cuda::std::greater<>{});
      cudax::mgmn::sort(cudax::distributed, comms, envs, inputs, counts, cuda::std::greater<>{});
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
    if (iteration == 10)
    {
      // Swap two rank contexts; every rank must reject before
      // any collective resource creation, without touching either context.
      envs.front() = contexts.back()->bind(streams.front());
      envs.back()  = contexts.front()->bind(streams.back());
      REQUIRE_THROWS_AS(cudax::mgmn::sort(cudax::distributed, comms, envs, inputs, counts), std::invalid_argument);
      envs.back() = contexts.back()->bind(streams.back());
      REQUIRE_THROWS_AS(cudax::mgmn::sort(cudax::distributed, comms, envs, inputs, counts), std::invalid_argument);
    }
  }
}
#endif // _CUDAX_SORT_HAS_NCCL_DEVICE
