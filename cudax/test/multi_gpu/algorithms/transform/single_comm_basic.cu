//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/buffer>
#include <cuda/memory_resource>
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/std/span>

#include <cuda/experimental/__multi_gpu/algorithm/transform/transform.h>

#include <string>
#include <vector>

#include <algorithm_common.h>
#include <nccl_test_common.h>
#include <testing.cuh>

#include "transform_common.cuh"

namespace
{
// Drive the transform through the single-communicator overload, one thread per local rank. Unlike
// `inclusive_scan`, `transform` posts no collective, so the per-rank calls need not rendezvous.
// Running them concurrently anyway shows that this overload needs no peer to make progress.
// Catch2 assertions remain on the main thread after all worker threads have joined.
template <class T, class Op>
void run_case(cuda::std::span<cudax::nccl_communicator_ref> comms,
              const std::vector<std::vector<T>>& inputs_by_rank,
              Op op)
{
  auto streams = nccl_test_util::make_streams();

  std::vector<cuda::device_buffer<T>> in;
  std::vector<cuda::device_buffer<T>> out;
  std::vector<decltype(::cuda::std::execution::env{::cuda::stream_ref{streams[0]}})> envs;

  in.reserve(comms.size());
  out.reserve(comms.size());
  envs.reserve(comms.size());
  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto& values = inputs_by_rank[static_cast<cuda::std::size_t>(comms[i].rank())];
    const auto device  = comms[i].logical_device().underlying_device();

    in.emplace_back(cuda::make_device_buffer<T>(streams[i], device, values));
    out.emplace_back(cuda::make_device_buffer<T>(streams[i], device, values.size(), cuda::no_init));
    envs.emplace_back(::cuda::std::execution::env{::cuda::stream_ref{streams[i]}});
  }

  const auto in_copy = in;

  run_threaded(comms.size(), [&](cuda::std::size_t i) {
    cudax::transform(cudax::distributed, comms[i], envs[i], in[i].begin(), in[i].size(), out[i].begin(), op);
  });

  // `transform` writes only to the output range, so the input must come back unchanged.
  REQUIRE(in.size() == in_copy.size());
  for (cuda::std::size_t i = 0; i < in.size(); ++i)
  {
    INFO("device = " << i);
    REQUIRE_THAT(in[i], Equals(in_copy[i]));

    const auto expected_values =
      transform_test_util::expected_for_rank(inputs_by_rank[static_cast<cuda::std::size_t>(comms[i].rank())], op);
    const auto expected =
      cuda::make_buffer<T>(out[i].stream(), cuda::mr::legacy_pinned_memory_resource{}, expected_values);

    REQUIRE_THAT(out[i], Equals(expected));
  }
}
} // namespace

MULTI_GPU_TEST("transform single-comm documentation example", c2h::type_list<int>)
{
  auto comms = this->communicators();

  if (comms.size() < 2)
  {
    SKIP("The transform documentation example requires at least two local GPUs");
  }

  auto streams_owned = nccl_test_util::make_streams();
  auto streams       = std::vector<cuda::stream_ref>{streams_owned.begin(), streams_owned.end()};

  // Must be pre-allocated since it is written to by threads
  std::vector<std::string> failed(comms.front().size());

  // `transform` posts no collective, but running every rank concurrently shows that this overload
  // needs no peer to make progress.
  run_threaded(comms.size(), [&](cuda::std::size_t i) {
    auto& communicator = comms[i];
    auto environment   = streams[i];
    const auto device  = communicator.logical_device().underlying_device();

    //! [transform_single_range]
    constexpr cuda::std::array input_values{1, 2};

    auto input  = cuda::make_device_buffer<int>(environment, device, input_values);
    auto output = cuda::make_device_buffer<int>(environment, device, input_values.size(), cuda::no_init);

    cudax::transform(
      cudax::distributed, communicator, environment, input.begin(), input.size(), output.begin(), cuda::std::negate<>{});

    // The operator is applied element by element and no rank sees another rank's elements, so every
    // rank negates its own {1, 2}.
    const auto expected = cuda::make_buffer<int>(output.stream(), cuda::mr::legacy_pinned_memory_resource{}, {-1, -2});
    //! [transform_single_range]

    // catch2 isn't thread safe by default, so we can't use the usual requires expression. So
    // we roll a hacky version of it ourselves
    if (const auto matcher = Equals(expected); !matcher.match(output))
    {
      failed[communicator.rank()] = matcher.describe();
    }
  });

  for (cuda::std::size_t i = 0; i < failed.size(); ++i)
  {
    if (const auto& err_str = failed[i]; !err_str.empty())
    {
      INFO("rank: " << i);
      REQUIRE(err_str == ""); // Should print the full error string
    }
  }
}

MULTI_GPU_TEST("transform single-comm, one element per rank", value_types, transform_test_util::operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  auto comms = this->communicators();

  std::vector<std::vector<T>> inputs_by_rank;

  inputs_by_rank.reserve(static_cast<cuda::std::size_t>(comms.front().size()));
  for (int r = 0; r < comms.front().size(); ++r)
  {
    const auto v = {make_value<T>(r)};

    inputs_by_rank.emplace_back(v);
  }

  run_case(comms, inputs_by_rank, Op{});
}

MULTI_GPU_TEST("transform single-comm, multiple elements per rank", value_types, transform_test_util::operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  auto comms = this->communicators();

  constexpr auto values_per_rank = 10;
  std::vector<std::vector<T>> inputs_by_rank(static_cast<cuda::std::size_t>(comms.front().size()));
  for (int r = 0; r < comms.front().size(); ++r)
  {
    const auto value                                  = make_value<T>(r);
    inputs_by_rank[static_cast<cuda::std::size_t>(r)] = std::vector<T>(values_per_rank, value);
  }

  run_case(comms, inputs_by_rank, Op{});
}

// Uneven sizes make sure a rank's element count comes from its own size argument and not from a
// neighbour's.
MULTI_GPU_TEST("transform single-comm, uneven rank sizes", value_types, transform_test_util::operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  auto comms = this->communicators();

  std::vector<std::vector<T>> inputs_by_rank(static_cast<cuda::std::size_t>(comms.front().size()));
  for (int r = 0; r < comms.front().size(); ++r)
  {
    const auto count = static_cast<cuda::std::size_t>(r) * 100 + 1;

    inputs_by_rank[static_cast<cuda::std::size_t>(r)] = std::vector<T>(count, make_value<T>(r));
  }

  run_case(comms, inputs_by_rank, Op{});
}

MULTI_GPU_TEST("transform single-comm, some ranks empty", value_types, transform_test_util::operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  auto comms = this->communicators();

  constexpr auto values_per_rank = 10;
  std::vector<std::vector<T>> inputs_by_rank(static_cast<cuda::std::size_t>(comms.front().size()));
  for (int r = 0; r < comms.front().size(); ++r)
  {
    if (r % 2 == 0)
    {
      inputs_by_rank[static_cast<cuda::std::size_t>(r)] = std::vector<T>(values_per_rank, make_value<T>(r));
    }
  }

  run_case(comms, inputs_by_rank, Op{});
}

MULTI_GPU_TEST("transform single-comm, all ranks empty", value_types, transform_test_util::operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  auto comms = this->communicators();

  const std::vector<std::vector<T>> inputs_by_rank(static_cast<cuda::std::size_t>(comms.front().size()));

  run_case(comms, inputs_by_rank, Op{});
}
