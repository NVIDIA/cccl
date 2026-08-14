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

#include <numeric>
#include <vector>

#include <algorithm_common.h>
#include <nccl_test_common.h>
#include <testing.cuh>

namespace
{
// `exclusive_scan` forwards the environment to `cub::DeviceScan::ExclusiveScan` and to the two
// `cub::DeviceReduce::Reduce` calls that build the per-rank prefix. So a determinism requirement
// is supported only where both CUB algorithms support it:
//
// - `run_to_run` needs an integral type with a known CUDA binary operator, or a floating-point
//   type with `cuda::std::plus`.
// - `gpu_to_gpu` needs an integral type with a known CUDA binary operator.
//
// `cuda::std::plus` and `cuda::maximum` are both known CUDA binary operators. Every other
// combination is rejected at compile time, so it cannot appear here.
using supported_cases =
  c2h::type_list<c2h::type_list<cuda::std::int32_t, cuda::std::plus<>, cuda::execution::determinism::run_to_run_t>,
                 c2h::type_list<cuda::std::int32_t, cuda::std::plus<>, cuda::execution::determinism::gpu_to_gpu_t>,
                 c2h::type_list<cuda::std::int32_t, cuda::maximum<>, cuda::execution::determinism::run_to_run_t>,
                 c2h::type_list<cuda::std::int32_t, cuda::maximum<>, cuda::execution::determinism::gpu_to_gpu_t>,
                 // A floating-point type needs the stable, fixed reduction order that only `plus` under
                 // `run_to_run` engages. `gpu_to_gpu` never supports a floating-point type.
                 c2h::type_list<float, cuda::std::plus<>, cuda::execution::determinism::run_to_run_t>>;

template <class T, class Op>
[[nodiscard]] T get_identity()
{
  if constexpr (cuda::std::is_same_v<Op, cuda::std::plus<>>)
  {
    return T{};
  }
  else
  {
    static_assert(cuda::std::is_same_v<Op, cuda::maximum<>>, "Add handling");
    return cuda::std::numeric_limits<T>::lowest();
  }
}

template <class T, class Op>
[[nodiscard]] std::vector<T>
expected_for_rank(int rank, const std::vector<std::vector<T>>& inputs_by_rank, const T& init, Op op)
{
  std::vector<T> reference;

  for (const auto& values : inputs_by_rank)
  {
    reference.insert(reference.end(), values.begin(), values.end());
  }

  std::vector<T> scan(reference.size());
  std::exclusive_scan(reference.begin(), reference.end(), scan.begin(), init, op);

  cuda::std::size_t offset = 0;
  for (int r = 0; r < rank; ++r)
  {
    offset += inputs_by_rank[static_cast<cuda::std::size_t>(r)].size();
  }

  const auto count = inputs_by_rank[static_cast<cuda::std::size_t>(rank)].size();
  return {scan.begin() + offset, scan.begin() + offset + count};
}
} // namespace

// The scan must accept the determinism requirement and still produce the same result as the
// host-side scan. Each rank runs on its own thread, since the per-rank calls must rendezvous in
// their collectives. Catch2 assertions stay on the main thread after the join.
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

  constexpr auto values_per_rank = 10;
  std::vector<std::vector<T>> inputs_by_rank(static_cast<cuda::std::size_t>(comms.front().size()));
  for (int r = 0; r < comms.front().size(); ++r)
  {
    inputs_by_rank[static_cast<cuda::std::size_t>(r)] = std::vector<T>(values_per_rank, static_cast<T>(r));
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

  run_threaded(comms.size(), [&](cuda::std::size_t i) {
    cudax::exclusive_scan(
      cudax::distributed, comms[i], envs[i], in[i].begin(), in[i].size(), out[i].begin(), init, op, ident);
  });

  for (cuda::std::size_t i = 0; i < out.size(); ++i)
  {
    INFO("device = " << i);

    const auto expected_values = expected_for_rank<T>(comms[i].rank(), inputs_by_rank, init, op);
    const auto expected =
      cuda::make_buffer<T>(out[i].stream(), cuda::mr::legacy_pinned_memory_resource{}, expected_values);

    REQUIRE_THAT(out[i], Equals(expected));
  }
}
