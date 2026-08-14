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
#include <cuda/std/type_traits>

#include <cuda/experimental/__multi_gpu/algorithm/reduce/reduce.h>

#include <numeric>
#include <vector>

#include <algorithm_common.h>
#include <determinism_common.h>
#include <nccl_test_common.h>
#include <testing.cuh>

namespace
{
// `gpu_to_gpu` is a compile-time error, covered by the `*_determinism_fail.cu` test.
using run_to_run_cases =
  c2h::type_list<c2h::type_list<cuda::std::int32_t, cuda::std::plus<>, cuda::execution::determinism::run_to_run_t>,
                 c2h::type_list<cuda::std::int32_t, cuda::maximum<>, cuda::execution::determinism::run_to_run_t>,
                 c2h::type_list<float, cuda::std::plus<>, cuda::execution::determinism::run_to_run_t>>;

// `make_random_values` draws from a numeric distribution, so `custom_value` cannot be an input
// here.
using arithmetic_value_types = c2h::remove<value_types, custom_value>;

using nondeterministic_cases =
  c2h::type_list<c2h::type_list<cuda::std::int32_t, cuda::std::plus<>, cuda::execution::determinism::not_guaranteed_t>,
                 c2h::type_list<cuda::std::int32_t, cuda::maximum<>, cuda::execution::determinism::not_guaranteed_t>,
                 c2h::type_list<float, cuda::std::plus<>, cuda::execution::determinism::not_guaranteed_t>>;
} // namespace

// Each rank runs on its own thread, because the per-rank calls must rendezvous in their
// collectives. Catch2 assertions stay on the main thread after the join.
MULTI_GPU_TEST("reduce single-comm, run_to_run determinism", run_to_run_cases)
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

  run_threaded(comms.size(), [&](cuda::std::size_t i) {
    cudax::mgmn::reduce(
      cudax::broadcasted, comms[i], envs[i], in[i].begin(), in[i].size(), out[i].begin(), init, op, ident);
  });

  // Keep an independent snapshot because each launch overwrites the output buffers.
  std::vector<std::vector<T>> first_results;
  for (const auto& buf : out)
  {
    first_results.push_back(::detail::to_vec(buf));
  }

  constexpr int num_runs = 4;
  for (int run = 0; run < num_runs; ++run)
  {
    INFO("run = " << run);

    run_threaded(comms.size(), [&](cuda::std::size_t i) {
      cudax::mgmn::reduce(
        cudax::broadcasted, comms[i], envs[i], in[i].begin(), in[i].size(), out[i].begin(), init, op, ident);
    });

    for (cuda::std::size_t i = 0; i < out.size(); ++i)
    {
      INFO("device = " << i);

      const auto actual = ::detail::to_vec(out[i]);
      REQUIRE(actual.size() == first_results[i].size());
      REQUIRE_THAT(actual, ::detail::BitwiseEqualsRange(first_results[i]));
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

  run_threaded(comms.size(), [&](cuda::std::size_t i) {
    cudax::mgmn::reduce(
      cudax::broadcasted, comms[i], envs[i], in[i].begin(), in[i].size(), out[i].begin(), init, op, ident);
  });

  // Keep an independent snapshot because each launch overwrites the output buffers.
  std::vector<std::vector<T>> first_results;
  for (const auto& buf : out)
  {
    first_results.push_back(::detail::to_vec(buf));
  }

  constexpr int num_runs = 4;
  for (int run = 0; run < num_runs; ++run)
  {
    INFO("run = " << run);

    run_threaded(comms.size(), [&](cuda::std::size_t i) {
      cudax::mgmn::reduce(
        cudax::broadcasted, comms[i], envs[i], in[i].begin(), in[i].size(), out[i].begin(), init, op, ident);
    });

    for (cuda::std::size_t i = 0; i < out.size(); ++i)
    {
      INFO("device = " << i);

      const auto actual = ::detail::to_vec(out[i]);
      REQUIRE(actual.size() == first_results[i].size());
      REQUIRE_THAT(actual, ::detail::BitwiseEqualsRange(first_results[i]));
    }
  }
}

MULTI_GPU_TEST("reduce single-comm, not_guaranteed correctness", nondeterministic_cases)
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

  T expected = init;
  for (const auto& values : inputs_by_rank)
  {
    expected = std::accumulate(values.begin(), values.end(), expected, op);
  }

  constexpr int num_runs = 4;

  for (int run = 0; run < num_runs; ++run)
  {
    INFO("run = " << run);

    run_threaded(comms.size(), [&](cuda::std::size_t i) {
      cudax::mgmn::reduce(
        cudax::broadcasted, comms[i], envs[i], in[i].begin(), in[i].size(), out[i].begin(), init, op, ident);
    });

    for (cuda::std::size_t i = 0; i < out.size(); ++i)
    {
      INFO("device = " << i);

      const auto actual = ::detail::to_vec(out[i]);
      REQUIRE(actual.size() == 1);
      if constexpr (cuda::std::is_floating_point_v<T>)
      {
        REQUIRE_APPROX_EQ_EPSILON(std::vector<T>{expected}, actual, 0.001);
      }
      else
      {
        REQUIRE(actual[0] == expected);
      }
    }
  }
}
