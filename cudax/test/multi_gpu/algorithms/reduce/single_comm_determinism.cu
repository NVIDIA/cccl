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
#include <cuda/memory_resource>
#include <cuda/std/cstddef>
#include <cuda/std/execution>
#include <cuda/std/functional>

#include <cuda/experimental/__multi_gpu/algorithm/reduce/reduce.h>

#include <exception>
#include <future>
#include <numeric>
#include <vector>

#include <algorithm_common.h>
#include <determinism_common.h>
#include <nccl_test_common.h>
#include <testing.cuh>

#include "reduce_common.cuh"
#include <c2h/catch2_test_helper.h>

// An explicit `not_guaranteed` requirement makes CUB select an atomic-based block reduction. That
// path calls `cuda::atomic_ref<T>::fetch_add`, which exists only for integral and floating-point
// types. So the requirement cannot be used with a class type such as `custom_value`.
using atomic_value_types = c2h::remove<value_types, custom_value>;

// `reduce` accepts an environment that requires `not_guaranteed` determinism, and an environment
// that requires nothing at all. The rejected requirements (`run_to_run` and `gpu_to_gpu`) are
// compile-time errors, so they are covered by the `*_determinism_fail.cu` tests instead.
MULTI_GPU_TEST("reduce single-comm, accepted determinism requirements", atomic_value_types, operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  const T init     = make_value<T>(GENERATE(0, 1, 5));
  const auto ident = get_identity<T, Op>();

  auto comms   = this->communicators();
  auto streams = nccl_test_util::make_streams();
  auto rng     = make_rng(C2H_SEED(2));

  std::vector<std::vector<T>> inputs_by_rank(static_cast<cuda::std::size_t>(comms.front().size()));

  const auto total_count = inputs_by_rank.size() * large_values_per_rank;
  for (auto& values : inputs_by_rank)
  {
    values = make_random_values<T>(large_values_per_rank, total_count, rng);
  }

  std::vector<cuda::device_buffer<T>> in;
  std::vector<cuda::device_buffer<T>> out;

  in.reserve(comms.size());
  out.reserve(comms.size());

  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto& values = inputs_by_rank[static_cast<cuda::std::size_t>(comms[i].rank())];

    in.emplace_back(cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), values));
    out.emplace_back(
      cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), 1, cuda::no_init));
  }

  const T expected = [&] {
    std::vector<T> reference;

    reference.reserve(inputs_by_rank.size() * large_values_per_rank);
    for (const auto& values : inputs_by_rank)
    {
      reference.insert(reference.end(), values.begin(), values.end());
    }

    return std::accumulate(reference.begin(), reference.end(), init, Op{});
  }();

  std::vector<cuda::buffer<T, cuda::mr::host_accessible>> exp;

  exp.reserve(comms.size());
  for (const auto& buf : out)
  {
    exp.emplace_back(
      cuda::make_buffer(buf.stream(), cuda::mr::legacy_pinned_memory_resource{}, /*__size=*/1, expected));
  }

  const auto run_and_check = [&](auto make_env) {
    std::vector<decltype(make_env(streams[0]))> envs;

    envs.reserve(comms.size());
    for (cuda::std::size_t i = 0; i < comms.size(); ++i)
    {
      envs.emplace_back(make_env(streams[i]));
    }

    // Every run is checked against the same reference, so the runs must also agree with each
    // other. The repeat catches a result that changes between two runs of the same input.
    constexpr int num_runs = 4;

    for (int run = 0; run < num_runs; ++run)
    {
      INFO("run = " << run);

      run_threaded(comms.size(), [&](cuda::std::size_t i) {
        cudax::reduce(
          cudax::broadcasted, comms[i], envs[i], in[i].begin(), in[i].size(), out[i].begin(), init, Op{}, ident);
      });

      for (cuda::std::size_t i = 0; i < out.size(); ++i)
      {
        INFO("device = " << i);

        REQUIRE_THAT(out[i], Equals(exp[i]));
      }
    }
  };

  SECTION("No requirements")
  {
    run_and_check([](cuda::stream_ref stream) {
      return ::cuda::std::execution::env{stream};
    });
  }

  SECTION("Explicit not_guaranteed requirement")
  {
    run_and_check([](cuda::stream_ref stream) {
      return ::cuda::std::execution::env{
        stream, ::cuda::execution::require(::cuda::execution::determinism::not_guaranteed)};
    });
  }
}
