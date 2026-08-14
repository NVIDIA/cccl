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

  // Global rank `comms[i].rank()` contributes ten copies of `rank`, exactly like the basic
  // single-communicator test. The only thing under test here is that a `not_guaranteed`
  // requirement in the environment neither fails to compile nor changes the result.
  std::vector<cuda::device_buffer<T>> in;
  std::vector<cuda::device_buffer<T>> out;

  in.reserve(comms.size());
  out.reserve(comms.size());

  constexpr auto values_per_rank = 10;
  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const std::vector<T> values(values_per_rank, make_value<T>(comms[i].rank()));

    in.emplace_back(cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), values));
    out.emplace_back(
      cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), 1, cuda::no_init));
  }

  const T expected = [&] {
    std::vector<T> reference;

    reference.reserve(comms.front().size() * values_per_rank);
    for (int r = 0; r < comms.front().size(); ++r)
    {
      reference.insert(reference.end(), values_per_rank, make_value<T>(r));
    }

    return std::accumulate(reference.begin(), reference.end(), init, Op{});
  }();

  // Builds one environment per rank from `make_env(stream)`, runs the reduction on every rank
  // concurrently, then checks every output against the host-side fold.
  const auto run_and_check = [&](auto make_env) {
    std::vector<decltype(make_env(streams[0]))> envs;

    envs.reserve(comms.size());
    for (cuda::std::size_t i = 0; i < comms.size(); ++i)
    {
      envs.emplace_back(make_env(streams[i]));
    }

    run_threaded(comms.size(), [&](cuda::std::size_t i) {
      cudax::reduce(
        cudax::broadcasted, comms[i], envs[i], in[i].begin(), in[i].size(), out[i].begin(), init, Op{}, ident);
    });

    for (const auto& buf : out)
    {
      const auto exp =
        cuda::make_buffer(buf.stream(), cuda::mr::legacy_pinned_memory_resource{}, /*__size=*/1, expected);

      REQUIRE_THAT(buf, Equals(exp));
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
