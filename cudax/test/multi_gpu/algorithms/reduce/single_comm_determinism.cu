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

#include <cuda/experimental/__multi_gpu/algorithm/reduce/reduce.h>

#include <numeric>
#include <vector>

#include <algorithm_common.h>
#include <determinism_common.h>
#include <nccl_test_common.h>
#include <testing.cuh>

namespace
{
// `reduce` forwards the environment to `cub::DeviceReduce::Reduce`. Only `run_to_run` and
// `not_guaranteed` are supported; `gpu_to_gpu` is a compile-time error, so the
// `*_determinism_fail.cu` test covers it.
//
// `not_guaranteed` makes CUB select an atomic-based block reduction. That path calls
// `cuda::atomic_ref<T>::fetch_add`, which exists only for integral and floating-point types.
using supported_cases =
  c2h::type_list<c2h::type_list<cuda::std::int32_t, cuda::std::plus<>, cuda::execution::determinism::run_to_run_t>,
                 c2h::type_list<cuda::std::int32_t, cuda::std::plus<>, cuda::execution::determinism::not_guaranteed_t>,
                 c2h::type_list<cuda::std::int32_t, cuda::maximum<>, cuda::execution::determinism::run_to_run_t>,
                 c2h::type_list<cuda::std::int32_t, cuda::maximum<>, cuda::execution::determinism::not_guaranteed_t>,
                 c2h::type_list<float, cuda::std::plus<>, cuda::execution::determinism::run_to_run_t>,
                 c2h::type_list<float, cuda::std::plus<>, cuda::execution::determinism::not_guaranteed_t>>;

// `make_random_values` draws from a numeric distribution, so `custom_value` cannot be an input
// here.
using arithmetic_value_types = c2h::remove<value_types, custom_value>;

template <class T, class Op>
[[nodiscard]] T expected_result(const std::vector<std::vector<T>>& inputs_by_rank, const T& init, Op op)
{
  std::vector<T> reference;

  for (const auto& values : inputs_by_rank)
  {
    reference.insert(reference.end(), values.begin(), values.end());
  }

  return std::accumulate(reference.begin(), reference.end(), init, op);
}
} // namespace

// Each rank runs on its own thread, because the per-rank calls must rendezvous in their
// collectives. Catch2 assertions stay on the main thread after the join.
MULTI_GPU_TEST("reduce single-comm, supported determinism requirements", supported_cases)
{
  using Case        = c2h::get<0, TestType>;
  using T           = c2h::get<0, Case>;
  using Op          = c2h::get<1, Case>;
  using Determinism = c2h::get<2, Case>;

  const T init     = make_value<T>(GENERATE(0, 1, 5));
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
    out.emplace_back(
      cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), 1, cuda::no_init));
    envs.emplace_back(make_env(streams[i]));
  }

  INFO("init = " << init);
  INFO("ident = " << ident);

  const T result = expected_result<T>(inputs_by_rank, init, op);

  std::vector<cuda::buffer<T, cuda::mr::host_accessible>> expected;

  expected.reserve(comms.size());
  for (const auto& buf : out)
  {
    expected.emplace_back(
      cuda::make_buffer(buf.stream(), cuda::mr::legacy_pinned_memory_resource{}, /*__size=*/1, result));
  }

  // Every run is checked against the same reference, so the runs must also agree with each other.
  // The repeat catches a result that changes between two runs of the same input.
  constexpr int num_runs = 4;

  for (int run = 0; run < num_runs; ++run)
  {
    INFO("run = " << run);

    run_threaded(comms.size(), [&](cuda::std::size_t i) {
      cudax::reduce(cudax::broadcasted, comms[i], envs[i], in[i].begin(), in[i].size(), out[i].begin(), init, op, ident);
    });

    for (cuda::std::size_t i = 0; i < out.size(); ++i)
    {
      INFO("device = " << i);

      REQUIRE_THAT(out[i], Equals(expected[i]));
    }
  }
}

// `run_to_run` is the default, so an environment that carries no requirement at all must give the
// same guarantee as one that asks for `run_to_run` explicitly.
MULTI_GPU_TEST("reduce single-comm, default determinism requirement", arithmetic_value_types, operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  const T init     = make_value<T>(GENERATE(0, 1, 5));
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
    return ::cuda::std::execution::env{stream};
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
    out.emplace_back(
      cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), 1, cuda::no_init));
    envs.emplace_back(make_env(streams[i]));
  }

  INFO("init = " << init);
  INFO("ident = " << ident);

  const T result = expected_result<T>(inputs_by_rank, init, op);

  std::vector<cuda::buffer<T, cuda::mr::host_accessible>> expected;

  expected.reserve(comms.size());
  for (const auto& buf : out)
  {
    expected.emplace_back(
      cuda::make_buffer(buf.stream(), cuda::mr::legacy_pinned_memory_resource{}, /*__size=*/1, result));
  }

  // Every run is checked against the same reference, so the runs must also agree with each other.
  // The repeat catches a result that changes between two runs of the same input.
  constexpr int num_runs = 4;

  for (int run = 0; run < num_runs; ++run)
  {
    INFO("run = " << run);

    run_threaded(comms.size(), [&](cuda::std::size_t i) {
      cudax::reduce(cudax::broadcasted, comms[i], envs[i], in[i].begin(), in[i].size(), out[i].begin(), init, op, ident);
    });

    for (cuda::std::size_t i = 0; i < out.size(); ++i)
    {
      INFO("device = " << i);

      REQUIRE_THAT(out[i], Equals(expected[i]));
    }
  }
}
