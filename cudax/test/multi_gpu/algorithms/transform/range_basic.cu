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
#include <cuda/functional>
#include <cuda/memory_resource>
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/std/ranges>
#include <cuda/std/span>

#include <cuda/experimental/__multi_gpu/algorithm/transform/transform.h>

#include <vector>

#include <nccl_test_common.h>
#include <testing.cuh>

#include "transform_common.cuh"

namespace
{
using transform_test_util::expected_for_rank;
using transform_test_util::make_value;
using transform_test_util::operators;
using transform_test_util::value_types;

// Run the whole world's transform through the range overload and check every rank against its own
// reference. This boilerplate is identical for every test regardless of how the inputs are shaped.
template <class T, class Op>
void run_case(cuda::std::span<cudax::nccl_communicator_ref> comms,
              const std::vector<std::vector<T>>& inputs_by_rank,
              Op op)
{
  auto streams_owned = nccl_test_util::make_streams();
  // Convert to stream_ref directly, cuda::stream on their own cant be passed directly to CUB
  auto streams = std::vector<cuda::stream_ref>{streams_owned.begin(), streams_owned.end()};

  std::vector<cuda::device_buffer<T>> in;
  std::vector<cuda::device_buffer<T>> out;

  in.reserve(comms.size());
  out.reserve(comms.size());
  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto& values = inputs_by_rank[static_cast<cuda::std::size_t>(comms[i].rank())];
    const auto device  = comms[i].logical_device().underlying_device();

    in.emplace_back(cuda::make_device_buffer<T>(streams[i], device, values));
    out.emplace_back(cuda::make_device_buffer<T>(streams[i], device, values.size(), cuda::no_init));
  }

  const auto in_copy = in;

  cudax::transform(
    cudax::distributed,
    comms,
    streams,
    in | cuda::std::views::transform(cuda::std::ranges::begin),
    in | cuda::std::views::transform(cuda::std::ranges::size),
    out | cuda::std::views::transform(cuda::std::ranges::begin),
    op);

  // `transform` writes only to the output range, so the input must come back unchanged.
  REQUIRE(in.size() == in_copy.size());
  for (cuda::std::size_t i = 0; i < in.size(); ++i)
  {
    INFO("device = " << i);
    REQUIRE_THAT(in[i], Equals(in_copy[i]));

    const auto expected_values = expected_for_rank(inputs_by_rank[static_cast<cuda::std::size_t>(comms[i].rank())], op);
    const auto expected =
      cuda::make_buffer<T>(out[i].stream(), cuda::mr::legacy_pinned_memory_resource{}, expected_values);

    REQUIRE_THAT(out[i], Equals(expected));
  }
}
} // namespace

MULTI_GPU_TEST("transform documentation example", c2h::type_list<int>)
{
  auto comms = this->communicators();

  if (comms.size() < 2)
  {
    SKIP("The transform documentation example requires at least two local GPUs");
  }

  auto streams_owned = nccl_test_util::make_streams();
  // Convert to stream_ref directly, cuda::stream on their own cant be passed directly to CUB
  auto streams = std::vector<cuda::stream_ref>{streams_owned.begin(), streams_owned.end()};

  //! [transform]
  constexpr cuda::std::array input_values{1, 2};
  std::vector<cuda::device_buffer<int>> inputs;
  std::vector<cuda::device_buffer<int>> outputs;

  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto device = comms[i].logical_device().underlying_device();

    inputs.emplace_back(cuda::make_device_buffer<int>(streams[i], device, input_values));
    outputs.emplace_back(cuda::make_device_buffer<int>(streams[i], device, input_values.size(), cuda::no_init));
  }

  cudax::transform(
    cudax::distributed,
    comms,
    // Passing streams as the environment directly
    streams,
    inputs | cuda::std::views::transform(cuda::std::ranges::begin),
    inputs | cuda::std::views::transform(cuda::std::ranges::size),
    outputs | cuda::std::views::transform(cuda::std::ranges::begin),
    cuda::std::negate<>{});

  // The operator is applied element by element and no rank sees another rank's elements, so every
  // rank negates its own {1, 2}.
  constexpr cuda::std::array expected_values{-1, -2};
  const auto expected_0 =
    cuda::make_buffer<int>(outputs[0].stream(), cuda::mr::legacy_pinned_memory_resource{}, expected_values);
  const auto expected_1 =
    cuda::make_buffer<int>(outputs[1].stream(), cuda::mr::legacy_pinned_memory_resource{}, expected_values);
  REQUIRE_THAT(outputs[0], Equals(expected_0));
  REQUIRE_THAT(outputs[1], Equals(expected_1));
  //! [transform]
}

MULTI_GPU_TEST("transform, one element per rank", value_types, operators)
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

MULTI_GPU_TEST("transform, multiple elements per rank", value_types, operators)
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

// Uneven sizes make sure a rank's element count comes from its own size entry and not from a
// neighbour's.
MULTI_GPU_TEST("transform, uneven rank sizes", value_types, operators)
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

MULTI_GPU_TEST("transform, some ranks empty", value_types, operators)
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

MULTI_GPU_TEST("transform, all ranks empty", value_types, operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  auto comms = this->communicators();

  const std::vector<std::vector<T>> inputs_by_rank(static_cast<cuda::std::size_t>(comms.front().size()));

  run_case(comms, inputs_by_rank, Op{});
}

// The environment is an `env` carrying a stream rather than a bare `stream_ref`, which is the
// other shape `transform` must accept for the `get_stream` query.
MULTI_GPU_TEST("transform, env environments", value_types, operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  auto comms   = this->communicators();
  auto streams = nccl_test_util::make_streams();

  constexpr auto values_per_rank = 10;
  std::vector<std::vector<T>> inputs_by_rank(static_cast<cuda::std::size_t>(comms.front().size()));
  for (int r = 0; r < comms.front().size(); ++r)
  {
    inputs_by_rank[static_cast<cuda::std::size_t>(r)] = std::vector<T>(values_per_rank, make_value<T>(r));
  }

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

  cudax::transform(
    cudax::distributed,
    comms,
    envs,
    in | cuda::std::views::transform(cuda::std::ranges::begin),
    in | cuda::std::views::transform(cuda::std::ranges::size),
    out | cuda::std::views::transform(cuda::std::ranges::begin),
    Op{});

  for (cuda::std::size_t i = 0; i < out.size(); ++i)
  {
    INFO("device = " << i);
    const auto expected_values =
      expected_for_rank(inputs_by_rank[static_cast<cuda::std::size_t>(comms[i].rank())], Op{});
    const auto expected =
      cuda::make_buffer<T>(out[i].stream(), cuda::mr::legacy_pinned_memory_resource{}, expected_values);

    REQUIRE_THAT(out[i], Equals(expected));
  }
}

// The output value type does not have to match the input value type.
MULTI_GPU_TEST("transform, differing input and output types", )
{
  using in_t  = cuda::std::int32_t;
  using out_t = double;

  auto comms         = this->communicators();
  auto streams_owned = nccl_test_util::make_streams();
  auto streams       = std::vector<cuda::stream_ref>{streams_owned.begin(), streams_owned.end()};

  constexpr auto values_per_rank = 10;
  std::vector<std::vector<in_t>> inputs_by_rank(static_cast<cuda::std::size_t>(comms.front().size()));
  for (int r = 0; r < comms.front().size(); ++r)
  {
    inputs_by_rank[static_cast<cuda::std::size_t>(r)] = std::vector<in_t>(values_per_rank, static_cast<in_t>(r));
  }

  std::vector<cuda::device_buffer<in_t>> in;
  std::vector<cuda::device_buffer<out_t>> out;

  in.reserve(comms.size());
  out.reserve(comms.size());
  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto& values = inputs_by_rank[static_cast<cuda::std::size_t>(comms[i].rank())];
    const auto device  = comms[i].logical_device().underlying_device();

    in.emplace_back(cuda::make_device_buffer<in_t>(streams[i], device, values));
    out.emplace_back(cuda::make_device_buffer<out_t>(streams[i], device, values.size(), cuda::no_init));
  }

  cudax::transform(
    cudax::distributed,
    comms,
    streams,
    in | cuda::std::views::transform(cuda::std::ranges::begin),
    in | cuda::std::views::transform(cuda::std::ranges::size),
    out | cuda::std::views::transform(cuda::std::ranges::begin),
    cuda::proclaim_return_type<out_t>([] _CCCL_HOST_DEVICE_API(in_t value) {
      return static_cast<out_t>(value) / 2.0;
    }));

  for (cuda::std::size_t i = 0; i < out.size(); ++i)
  {
    INFO("device = " << i);

    const auto& values = inputs_by_rank[static_cast<cuda::std::size_t>(comms[i].rank())];
    std::vector<out_t> expected_values(values.size());
    for (cuda::std::size_t item = 0; item < values.size(); ++item)
    {
      expected_values[item] = static_cast<out_t>(values[item]) / 2.0;
    }

    const auto expected =
      cuda::make_buffer<out_t>(out[i].stream(), cuda::mr::legacy_pinned_memory_resource{}, expected_values);

    REQUIRE_THAT(out[i], Equals(expected));
  }
}

// In place is a legal use: the output iterator may alias the input iterator.
MULTI_GPU_TEST("transform, in place", value_types, operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  auto comms         = this->communicators();
  auto streams_owned = nccl_test_util::make_streams();
  auto streams       = std::vector<cuda::stream_ref>{streams_owned.begin(), streams_owned.end()};

  constexpr auto values_per_rank = 10;
  std::vector<std::vector<T>> inputs_by_rank(static_cast<cuda::std::size_t>(comms.front().size()));
  for (int r = 0; r < comms.front().size(); ++r)
  {
    inputs_by_rank[static_cast<cuda::std::size_t>(r)] = std::vector<T>(values_per_rank, make_value<T>(r));
  }

  std::vector<cuda::device_buffer<T>> buffers;

  buffers.reserve(comms.size());
  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto& values = inputs_by_rank[static_cast<cuda::std::size_t>(comms[i].rank())];

    buffers.emplace_back(
      cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), values));
  }

  cudax::transform(
    cudax::distributed,
    comms,
    streams,
    buffers | cuda::std::views::transform(cuda::std::ranges::begin),
    buffers | cuda::std::views::transform(cuda::std::ranges::size),
    buffers | cuda::std::views::transform(cuda::std::ranges::begin),
    Op{});

  for (cuda::std::size_t i = 0; i < buffers.size(); ++i)
  {
    INFO("device = " << i);
    const auto expected_values =
      expected_for_rank(inputs_by_rank[static_cast<cuda::std::size_t>(comms[i].rank())], Op{});
    const auto expected =
      cuda::make_buffer<T>(buffers[i].stream(), cuda::mr::legacy_pinned_memory_resource{}, expected_values);

    REQUIRE_THAT(buffers[i], Equals(expected));
  }
}
