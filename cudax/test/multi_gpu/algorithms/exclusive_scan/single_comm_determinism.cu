//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>
#include <cuda/buffer>
#include <cuda/functional>
#include <cuda/memory_resource>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda/std/span>
#include <cuda/std/type_traits>

#include <cuda/experimental/__multi_gpu/algorithm/scan/scan.h>

#include <vector>

#include <algorithm_common.h>
#include <determinism_common.h>
#include <nccl_test_common.h>
#include <scan_common.h>
#include <testing.cuh>

namespace
{
// `exclusive_scan` forwards the environment to `cub::DeviceScan::ExclusiveScan` and to the two
// `cub::DeviceReduce::Reduce` calls that build the per-rank prefix. A requirement is supported
// only where both CUB algorithms support it. Every other combination is a compile-time error, so
// it cannot appear here.
using supported_cases =
  c2h::type_list<c2h::type_list<cuda::std::int32_t, cuda::std::plus<>, cuda::execution::determinism::run_to_run_t>,
                 c2h::type_list<cuda::std::int32_t, cuda::std::plus<>, cuda::execution::determinism::gpu_to_gpu_t>,
                 c2h::type_list<cuda::std::int32_t, cuda::maximum<>, cuda::execution::determinism::run_to_run_t>,
                 c2h::type_list<cuda::std::int32_t, cuda::maximum<>, cuda::execution::determinism::gpu_to_gpu_t>,
                 // `gpu_to_gpu` does not support a floating-point type.
                 c2h::type_list<float, cuda::std::plus<>, cuda::execution::determinism::run_to_run_t>>;
} // namespace

// Each rank runs on its own thread, because the per-rank calls must rendezvous in their
// collectives. Catch2 assertions stay on the main thread after the join.
MULTI_GPU_TEST("exclusive_scan single-comm, supported determinism requirements", supported_cases)
{
  using Case        = c2h::get<0, TestType>;
  using T           = c2h::get<0, Case>;
  using Op          = c2h::get<1, Case>;
  using Determinism = c2h::get<2, Case>;

  const T init     = static_cast<T>(GENERATE(0, 1, -1, 5));
  const auto ident = get_identity<T, Op>();
  constexpr Op op{};

  auto comms = this->communicators();
  auto rng   = make_rng(C2H_SEED(2));

  std::vector<std::vector<T>> inputs_by_rank(static_cast<cuda::std::size_t>(comms.front().size()));

  const auto total_count = inputs_by_rank.size() * large_values_per_rank;
  for (auto& values : inputs_by_rank)
  {
    values = make_random_values<T>(large_values_per_rank, total_count, rng);
  }

  auto streams = nccl_test_util::make_streams();

  const auto make_env = [](cuda::stream_ref stream) {
    return ::cuda::std::execution::env{stream, ::cuda::execution::require(Determinism{})};
  };

  std::vector<cuda::device_buffer<T>> in;
  std::vector<cuda::device_buffer<T>> out;
  std::vector<decltype(make_env(streams[0]))> envs;

  in.reserve(comms.size());
  out.reserve(comms.size());
  envs.reserve(comms.size());
  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto& values = inputs_by_rank[static_cast<cuda::std::size_t>(comms[i].rank())];

    in.emplace_back(cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), values));
    out.emplace_back(cuda::make_device_buffer<T>(
      streams[i], comms[i].logical_device().underlying_device(), values.size(), cuda::no_init));
    envs.emplace_back(make_env(streams[i]));
  }

  INFO("init = " << init);
  INFO("ident = " << ident);

  std::vector<cuda::buffer<T, cuda::mr::host_accessible>> expected;

  expected.reserve(comms.size());
  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    expected.emplace_back(cuda::make_buffer<T>(
      out[i].stream(),
      cuda::mr::legacy_pinned_memory_resource{},
      scan_test_util::exclusive_expected_for_rank<T>(comms[i].rank(), inputs_by_rank, init, op)));
  }

  // Every run is checked against the same reference, so the runs must also agree with each other.
  // The repeat catches a result that changes between two runs of the same input.
  constexpr int num_runs = 4;

  for (int run = 0; run < num_runs; ++run)
  {
    INFO("run = " << run);

    run_threaded(comms.size(), [&](cuda::std::size_t i) {
      cudax::exclusive_scan(
        cudax::distributed, comms[i], envs[i], in[i].begin(), in[i].size(), out[i].begin(), init, op, ident);
    });

    for (cuda::std::size_t i = 0; i < out.size(); ++i)
    {
      INFO("device = " << i);

      REQUIRE_THAT(out[i], Equals(expected[i]));
    }
  }
}
