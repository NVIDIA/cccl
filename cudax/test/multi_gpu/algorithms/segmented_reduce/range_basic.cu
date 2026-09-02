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
#include <cuda/std/execution>
#include <cuda/std/ranges>
#include <cuda/std/span>

#include <cuda/experimental/__multi_gpu/algorithm/reduce/segmented_reduce.h>

#include <numeric>
#include <vector>

#include <algorithm_common.h>
#include <nccl_test_common.h>
#include <testing.cuh>

#include "segmented_reduce_common.cuh"

namespace
{
template <class Env, class T, class Op>
void do_segmented_reduce(
  cuda::std::span<cudax::nccl_communicator_ref> comms,
  const std::vector<Env>& envs,
  std::vector<cuda::device_buffer<T>>& in,
  cuda::std::size_t num_segments,
  std::vector<cuda::device_buffer<offset_type>>& offsets,
  std::vector<cuda::device_buffer<T>>& out,
  const T& init,
  const T& ident,
  Op op)
{
  const auto envs_size    = envs.size();
  const auto in_copy      = in;
  const auto offsets_copy = offsets;

  INFO("init = " << init);
  INFO("ident = " << ident);

  cudax::segmented_reduce(
    cudax::broadcasted,
    comms,
    envs,
    in | cuda::std::views::transform(cuda::std::ranges::begin),
    num_segments,
    offsets | cuda::std::views::transform(cuda::std::ranges::begin),
    offsets | cuda::std::views::transform([](auto& buf) {
      return cuda::std::ranges::begin(buf) + 1;
    }),
    out | cuda::std::views::transform(cuda::std::ranges::begin),
    init,
    op,
    ident);

  REQUIRE(envs.size() == envs_size);
  REQUIRE(in.size() == in_copy.size());
  REQUIRE(offsets.size() == offsets_copy.size());
  for (cuda::std::size_t i = 0; i < in.size(); ++i)
  {
    INFO("device = " << i);
    REQUIRE_THAT(in[i], Equals(in_copy[i]));
    REQUIRE_THAT(offsets[i], Equals(offsets_copy[i]));
  }
}
} // namespace

MULTI_GPU_TEST("segmented_reduce documentation example", c2h::type_list<int>)
{
  auto comms = this->communicators();

  if (comms.size() != 2)
  {
    SKIP("The segmented_reduce documentation example requires exactly two local GPUs");
  }

  auto streams_owned = nccl_test_util::make_streams();
  // Convert to stream_ref directly, cuda::stream on their own cant be passed directly to CUB
  auto streams = std::vector<cuda::stream_ref>{streams_owned.begin(), streams_owned.end()};

  //! [segmented_reduce]
  // Two segments per rank: {1, 2} and {3, 4, 5}.
  constexpr cuda::std::array input_values{1, 2, 3, 4, 5};
  constexpr cuda::std::array offset_values{0, 2, 5};
  constexpr cuda::std::size_t num_segments = offset_values.size() - 1;

  std::vector<cuda::device_buffer<int>> inputs;
  std::vector<cuda::device_buffer<int>> offsets;
  std::vector<cuda::device_buffer<int>> outputs;

  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto device = comms[i].logical_device().underlying_device();

    inputs.emplace_back(cuda::make_device_buffer<int>(streams[i], device, input_values));
    offsets.emplace_back(cuda::make_device_buffer<int>(streams[i], device, offset_values));
    outputs.emplace_back(cuda::make_device_buffer<int>(streams[i], device, num_segments, cuda::no_init));
  }

  cudax::segmented_reduce(
    cudax::broadcasted,
    comms,
    // Passing streams as the environment directly
    streams,
    inputs | cuda::std::views::transform(cuda::std::ranges::begin),
    num_segments,
    // Segment `s` covers [begin_offsets[s], end_offsets[s]), so the end offsets are just the
    // begin offsets shifted by one.
    offsets | cuda::std::views::transform(cuda::std::ranges::begin),
    offsets | cuda::std::views::transform([](auto& buf) {
      return cuda::std::ranges::begin(buf) + 1;
    }),
    outputs | cuda::std::views::transform(cuda::std::ranges::begin),
    /*__init=*/0);

  // Every rank contributes the same two segments, so segment 0 sums to 3 * nranks and segment 1
  // to 12 * nranks. `segmented_reduce` broadcasts the result, so both local outputs hold the
  // same values.
  const auto nranks = comms.front().size();
  const std::vector<int> expected_values{3 * nranks, 12 * nranks};
  const auto expected_0 =
    cuda::make_buffer<int>(outputs[0].stream(), cuda::mr::legacy_pinned_memory_resource{}, expected_values);
  const auto expected_1 =
    cuda::make_buffer<int>(outputs[1].stream(), cuda::mr::legacy_pinned_memory_resource{}, expected_values);
  REQUIRE_THAT(outputs[0], Equals(expected_0));
  REQUIRE_THAT(outputs[1], Equals(expected_1));
  //! [segmented_reduce]
}

MULTI_GPU_TEST("segmented_reduce, one segment of one element per rank", value_types, operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  const T init     = make_value<T>(GENERATE(0, 1, -1, 5));
  const auto ident = get_identity<T, Op>();

  auto comms   = this->communicators();
  auto streams = nccl_test_util::make_streams();
  auto rng     = make_rng(C2H_SEED(2));

  constexpr cuda::std::array local_offsets = {1};
  constexpr cuda::std::size_t num_segments = local_offsets.size();
  const auto values_by_rank                = uniform_values<T>(comms.front().size(), local_offsets, rng);
  const auto offsets_by_rank               = uniform_offsets(comms.front().size(), local_offsets);

  std::vector<cuda::device_buffer<T>> in;
  std::vector<cuda::device_buffer<offset_type>> offsets;
  std::vector<cuda::device_buffer<T>> out;
  std::vector<decltype(::cuda::std::execution::env{::cuda::stream_ref{streams[0]}})> envs;

  in.reserve(comms.size());
  offsets.reserve(comms.size());
  out.reserve(comms.size());
  envs.reserve(comms.size());
  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto device = comms[i].logical_device().underlying_device();
    const auto rank   = static_cast<cuda::std::size_t>(comms[i].rank());

    in.emplace_back(cuda::make_device_buffer<T>(streams[i], device, values_by_rank[rank]));
    offsets.emplace_back(cuda::make_device_buffer<offset_type>(streams[i], device, offsets_by_rank[rank]));
    out.emplace_back(cuda::make_device_buffer<T>(streams[i], device, num_segments, cuda::no_init));
    envs.emplace_back(::cuda::std::execution::env{::cuda::stream_ref{streams[i]}});
  }

  do_segmented_reduce(comms, envs, in, num_segments, offsets, out, init, ident, Op{});
  check_outputs(out, values_by_rank, offsets_by_rank, num_segments, init, Op{});
}

MULTI_GPU_TEST("segmented_reduce, multiple equal-sized segments per rank", value_types, operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  const T init     = make_value<T>(GENERATE(0, 1, -1, 5));
  const auto ident = get_identity<T, Op>();

  auto comms   = this->communicators();
  auto streams = nccl_test_util::make_streams();
  auto rng     = make_rng(C2H_SEED(2));

  constexpr cuda::std::array local_offsets = {10, 10, 10, 10};
  constexpr cuda::std::size_t num_segments = local_offsets.size();
  const auto values_by_rank                = uniform_values<T>(comms.front().size(), local_offsets, rng);
  const auto offsets_by_rank               = uniform_offsets(comms.front().size(), local_offsets);

  std::vector<cuda::device_buffer<T>> in;
  std::vector<cuda::device_buffer<offset_type>> offsets;
  std::vector<cuda::device_buffer<T>> out;
  std::vector<decltype(::cuda::std::execution::env{::cuda::stream_ref{streams[0]}})> envs;

  in.reserve(comms.size());
  offsets.reserve(comms.size());
  out.reserve(comms.size());
  envs.reserve(comms.size());
  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto device = comms[i].logical_device().underlying_device();
    const auto rank   = static_cast<cuda::std::size_t>(comms[i].rank());

    in.emplace_back(cuda::make_device_buffer<T>(streams[i], device, values_by_rank[rank]));
    offsets.emplace_back(cuda::make_device_buffer<offset_type>(streams[i], device, offsets_by_rank[rank]));
    out.emplace_back(cuda::make_device_buffer<T>(streams[i], device, num_segments, cuda::no_init));
    envs.emplace_back(::cuda::std::execution::env{::cuda::stream_ref{streams[i]}});
  }

  do_segmented_reduce(comms, envs, in, num_segments, offsets, out, init, ident, Op{});
  check_outputs(out, values_by_rank, offsets_by_rank, num_segments, init, Op{});
}

MULTI_GPU_TEST("segmented_reduce, ragged segments per rank", value_types, operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  const T init     = make_value<T>(GENERATE(0, 1, -1, 5));
  const auto ident = get_identity<T, Op>();

  auto comms   = this->communicators();
  auto streams = nccl_test_util::make_streams();
  auto rng     = make_rng(C2H_SEED(2));

  constexpr cuda::std::array local_offsets = {1, 7, 3, 128, 12};
  constexpr cuda::std::size_t num_segments = local_offsets.size();
  const auto values_by_rank                = uniform_values<T>(comms.front().size(), local_offsets, rng);
  const auto offsets_by_rank               = uniform_offsets(comms.front().size(), local_offsets);

  std::vector<cuda::device_buffer<T>> in;
  std::vector<cuda::device_buffer<offset_type>> offsets;
  std::vector<cuda::device_buffer<T>> out;
  std::vector<decltype(::cuda::std::execution::env{::cuda::stream_ref{streams[0]}})> envs;

  in.reserve(comms.size());
  offsets.reserve(comms.size());
  out.reserve(comms.size());
  envs.reserve(comms.size());
  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto device = comms[i].logical_device().underlying_device();
    const auto rank   = static_cast<cuda::std::size_t>(comms[i].rank());

    in.emplace_back(cuda::make_device_buffer<T>(streams[i], device, values_by_rank[rank]));
    offsets.emplace_back(cuda::make_device_buffer<offset_type>(streams[i], device, offsets_by_rank[rank]));
    out.emplace_back(cuda::make_device_buffer<T>(streams[i], device, num_segments, cuda::no_init));
    envs.emplace_back(::cuda::std::execution::env{::cuda::stream_ref{streams[i]}});
  }

  do_segmented_reduce(comms, envs, in, num_segments, offsets, out, init, ident, Op{});
  check_outputs(out, values_by_rank, offsets_by_rank, num_segments, init, Op{});
}

MULTI_GPU_TEST("segmented_reduce, segment lengths differ across ranks", value_types, operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  const T init     = make_value<T>(GENERATE(0, 1, -1, 5));
  const auto ident = get_identity<T, Op>();

  auto comms   = this->communicators();
  auto streams = nccl_test_util::make_streams();
  auto rng     = make_rng(C2H_SEED(2));

  constexpr cuda::std::size_t num_segments = 3;
  std::vector<std::vector<T>> values_by_rank;
  std::vector<std::vector<offset_type>> offsets_by_rank;

  values_by_rank.reserve(static_cast<cuda::std::size_t>(comms.front().size()));
  offsets_by_rank.reserve(static_cast<cuda::std::size_t>(comms.front().size()));
  for (int r = 0; r < comms.front().size(); ++r)
  {
    const std::vector<offset_type> segment_sizes{
      static_cast<offset_type>(r + 1), static_cast<offset_type>(2 * r + 3), static_cast<offset_type>(r + 5)};

    fill_random(
      values_by_rank.emplace_back(),
      static_cast<cuda::std::size_t>(std::accumulate(segment_sizes.begin(), segment_sizes.end(), offset_type{0})),
      rng);
    offsets_by_rank.push_back(make_offsets(segment_sizes));
  }

  std::vector<cuda::device_buffer<T>> in;
  std::vector<cuda::device_buffer<offset_type>> offsets;
  std::vector<cuda::device_buffer<T>> out;
  std::vector<decltype(::cuda::std::execution::env{::cuda::stream_ref{streams[0]}})> envs;

  in.reserve(comms.size());
  offsets.reserve(comms.size());
  out.reserve(comms.size());
  envs.reserve(comms.size());
  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto device = comms[i].logical_device().underlying_device();
    const auto rank   = static_cast<cuda::std::size_t>(comms[i].rank());

    in.emplace_back(cuda::make_device_buffer<T>(streams[i], device, values_by_rank[rank]));
    offsets.emplace_back(cuda::make_device_buffer<offset_type>(streams[i], device, offsets_by_rank[rank]));
    out.emplace_back(cuda::make_device_buffer<T>(streams[i], device, num_segments, cuda::no_init));
    envs.emplace_back(::cuda::std::execution::env{::cuda::stream_ref{streams[i]}});
  }

  do_segmented_reduce(comms, envs, in, num_segments, offsets, out, init, ident, Op{});
  check_outputs(out, values_by_rank, offsets_by_rank, num_segments, init, Op{});
}

MULTI_GPU_TEST("segmented_reduce, some segments empty", value_types, operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  const T init     = make_value<T>(GENERATE(0, 1, -1, 5));
  const auto ident = get_identity<T, Op>();

  auto comms   = this->communicators();
  auto streams = nccl_test_util::make_streams();
  auto rng     = make_rng(C2H_SEED(2));

  constexpr cuda::std::array local_offsets = {6, 0, 4, 0};
  constexpr cuda::std::size_t num_segments = local_offsets.size();
  const auto values_by_rank                = uniform_values<T>(comms.front().size(), local_offsets, rng);
  const auto offsets_by_rank               = uniform_offsets(comms.front().size(), local_offsets);

  std::vector<cuda::device_buffer<T>> in;
  std::vector<cuda::device_buffer<offset_type>> offsets;
  std::vector<cuda::device_buffer<T>> out;
  std::vector<decltype(::cuda::std::execution::env{::cuda::stream_ref{streams[0]}})> envs;

  in.reserve(comms.size());
  offsets.reserve(comms.size());
  out.reserve(comms.size());
  envs.reserve(comms.size());
  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto device = comms[i].logical_device().underlying_device();
    const auto rank   = static_cast<cuda::std::size_t>(comms[i].rank());

    in.emplace_back(cuda::make_device_buffer<T>(streams[i], device, values_by_rank[rank]));
    offsets.emplace_back(cuda::make_device_buffer<offset_type>(streams[i], device, offsets_by_rank[rank]));
    out.emplace_back(cuda::make_device_buffer<T>(streams[i], device, num_segments, cuda::no_init));
    envs.emplace_back(::cuda::std::execution::env{::cuda::stream_ref{streams[i]}});
  }

  do_segmented_reduce(comms, envs, in, num_segments, offsets, out, init, ident, Op{});
  check_outputs(out, values_by_rank, offsets_by_rank, num_segments, init, Op{});
}

MULTI_GPU_TEST("segmented_reduce, some ranks empty", value_types, operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  const T init     = make_value<T>(GENERATE(0, 1, -1, 5));
  const auto ident = get_identity<T, Op>();

  auto comms   = this->communicators();
  auto streams = nccl_test_util::make_streams();
  auto rng     = make_rng(C2H_SEED(2));

  constexpr cuda::std::size_t num_segments = 3;
  std::vector<std::vector<T>> values_by_rank;
  std::vector<std::vector<offset_type>> offsets_by_rank;

  values_by_rank.reserve(static_cast<cuda::std::size_t>(comms.front().size()));
  offsets_by_rank.reserve(static_cast<cuda::std::size_t>(comms.front().size()));
  for (int r = 0; r < comms.front().size(); ++r)
  {
    const std::vector<offset_type> segment_sizes =
      (r % 2 == 0) ? std::vector<offset_type>{5, 8, 2} : std::vector<offset_type>{0, 0, 0};

    values_by_rank.emplace_back();
    fill_random(
      values_by_rank.back(),
      static_cast<cuda::std::size_t>(std::accumulate(segment_sizes.begin(), segment_sizes.end(), offset_type{0})),
      rng);
    offsets_by_rank.push_back(make_offsets(segment_sizes));
  }

  std::vector<cuda::device_buffer<T>> in;
  std::vector<cuda::device_buffer<offset_type>> offsets;
  std::vector<cuda::device_buffer<T>> out;
  std::vector<decltype(::cuda::std::execution::env{::cuda::stream_ref{streams[0]}})> envs;

  in.reserve(comms.size());
  offsets.reserve(comms.size());
  out.reserve(comms.size());
  envs.reserve(comms.size());
  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto device = comms[i].logical_device().underlying_device();
    const auto rank   = static_cast<cuda::std::size_t>(comms[i].rank());

    in.emplace_back(cuda::make_device_buffer<T>(streams[i], device, values_by_rank[rank]));
    offsets.emplace_back(cuda::make_device_buffer<offset_type>(streams[i], device, offsets_by_rank[rank]));
    out.emplace_back(cuda::make_device_buffer<T>(streams[i], device, num_segments, cuda::no_init));
    envs.emplace_back(::cuda::std::execution::env{::cuda::stream_ref{streams[i]}});
  }

  do_segmented_reduce(comms, envs, in, num_segments, offsets, out, init, ident, Op{});
  check_outputs(out, values_by_rank, offsets_by_rank, num_segments, init, Op{});
}

MULTI_GPU_TEST("segmented_reduce, all ranks empty", value_types, operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  const T init     = make_value<T>(GENERATE(0, 1, -1, 5));
  const auto ident = get_identity<T, Op>();

  auto comms   = this->communicators();
  auto streams = nccl_test_util::make_streams();
  auto rng     = make_rng(C2H_SEED(2));

  constexpr cuda::std::array local_offsets = {0, 0, 0};
  constexpr cuda::std::size_t num_segments = local_offsets.size();
  const auto values_by_rank                = uniform_values<T>(comms.front().size(), local_offsets, rng);
  const auto offsets_by_rank               = uniform_offsets(comms.front().size(), local_offsets);

  std::vector<cuda::device_buffer<T>> in;
  std::vector<cuda::device_buffer<offset_type>> offsets;
  std::vector<cuda::device_buffer<T>> out;
  std::vector<decltype(::cuda::std::execution::env{::cuda::stream_ref{streams[0]}})> envs;

  in.reserve(comms.size());
  offsets.reserve(comms.size());
  out.reserve(comms.size());
  envs.reserve(comms.size());
  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto device = comms[i].logical_device().underlying_device();
    const auto rank   = static_cast<cuda::std::size_t>(comms[i].rank());

    in.emplace_back(cuda::make_device_buffer<T>(streams[i], device, values_by_rank[rank]));
    offsets.emplace_back(cuda::make_device_buffer<offset_type>(streams[i], device, offsets_by_rank[rank]));
    out.emplace_back(cuda::make_device_buffer<T>(streams[i], device, num_segments, cuda::no_init));
    envs.emplace_back(::cuda::std::execution::env{::cuda::stream_ref{streams[i]}});
  }

  do_segmented_reduce(comms, envs, in, num_segments, offsets, out, init, ident, Op{});
  check_outputs(out, values_by_rank, offsets_by_rank, num_segments, init, Op{});
}

// Row-wise reduction of a row-major matrix that is split by column across the ranks. Each rank
// owns `cols_per_rank` columns of every row and treats one row as one segment, so the local
// reduction folds each rank's slice of a row along axis 1 and the cross-rank reduction then
// combines the per-rank partials of that row. Every rank receives the full row totals.
//
// The diagram shows all 1's for legibility only; the test uses random values.
//
//   input              local red.     global red.
//   GPU 0   GPU 1     GPU 0   GPU 1  GPU 0   GPU 1
//   axis 1 ->
// a 1 1 1 | 1 1 1         3 | 3          6 | 6
// x 1 1 1 | 1 1 1         3 | 3          6 | 6
// i 1 1 1 | 1 1 1         3 | 3          6 | 6
// s 1 1 1 | 1 1 1   =>    3 | 3   =>     6 | 6
// 0 1 1 1 | 1 1 1         3 | 3          6 | 6
// | 1 1 1 | 1 1 1         3 | 3          6 | 6
// V 1 1 1 | 1 1 1         3 | 3          6 | 6
//   1 1 1 | 1 1 1         3 | 3          6 | 6
MULTI_GPU_TEST("segmented_reduce, row-wise reduction of a column-split matrix", value_types, operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  const T init     = make_value<T>(GENERATE(0, 1, -1, 5));
  const auto ident = get_identity<T, Op>();

  auto comms   = this->communicators();
  auto streams = nccl_test_util::make_streams();
  auto rng     = make_rng(C2H_SEED(2));

  constexpr cuda::std::size_t num_rows     = 8;
  constexpr offset_type cols_per_rank      = 3;
  constexpr cuda::std::size_t num_segments = num_rows;
  const std::vector<offset_type> row_sizes(num_rows, cols_per_rank);

  // Each rank's slice is row-major, so row `r` occupies `[r * cols_per_rank, (r + 1) *
  // cols_per_rank)` and the offsets are the same on every rank.
  const auto offsets_by_rank = uniform_offsets(comms.front().size(), row_sizes);
  std::vector<std::vector<T>> values_by_rank(static_cast<cuda::std::size_t>(comms.front().size()));

  for (auto& values : values_by_rank)
  {
    fill_random(values, num_rows * static_cast<cuda::std::size_t>(cols_per_rank), rng);
  }

  std::vector<cuda::device_buffer<T>> in;
  std::vector<cuda::device_buffer<offset_type>> offsets;
  std::vector<cuda::device_buffer<T>> out;
  std::vector<decltype(::cuda::std::execution::env{::cuda::stream_ref{streams[0]}})> envs;

  in.reserve(comms.size());
  offsets.reserve(comms.size());
  out.reserve(comms.size());
  envs.reserve(comms.size());
  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto device = comms[i].logical_device().underlying_device();
    const auto rank   = static_cast<cuda::std::size_t>(comms[i].rank());

    in.emplace_back(cuda::make_device_buffer<T>(streams[i], device, values_by_rank[rank]));
    offsets.emplace_back(cuda::make_device_buffer<offset_type>(streams[i], device, offsets_by_rank[rank]));
    out.emplace_back(cuda::make_device_buffer<T>(streams[i], device, num_segments, cuda::no_init));
    envs.emplace_back(::cuda::std::execution::env{::cuda::stream_ref{streams[i]}});
  }

  do_segmented_reduce(comms, envs, in, num_segments, offsets, out, init, ident, Op{});
  check_outputs(out, values_by_rank, offsets_by_rank, num_segments, init, Op{});
}

MULTI_GPU_TEST("segmented_reduce, zero segments", value_types, operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  const T init     = make_value<T>(GENERATE(0, 1, -1, 5));
  const auto ident = get_identity<T, Op>();

  auto comms   = this->communicators();
  auto streams = nccl_test_util::make_streams();
  auto rng     = make_rng(C2H_SEED(2));

  constexpr cuda::std::array<offset_type, 0> local_offsets = {};
  constexpr cuda::std::size_t num_segments                 = local_offsets.size();
  const auto values_by_rank  = uniform_values<T>(comms.front().size(), local_offsets, rng);
  const auto offsets_by_rank = uniform_offsets(comms.front().size(), local_offsets);

  // With no segments there is no output to compare against, so comparing `out` against an
  // empty expected buffer asserts nothing. Instead give `out` one element holding a sentinel
  // and require that `segmented_reduce` leaves it untouched. This catches a write past the
  // end of a zero-length output, which is the failure a zero-segment run can actually have.
  const T sentinel = make_value<T>(4242);
  const std::vector<T> sentinel_values(1, sentinel);

  std::vector<cuda::device_buffer<T>> in;
  std::vector<cuda::device_buffer<offset_type>> offsets;
  std::vector<cuda::device_buffer<T>> out;
  std::vector<decltype(::cuda::std::execution::env{::cuda::stream_ref{streams[0]}})> envs;

  in.reserve(comms.size());
  offsets.reserve(comms.size());
  out.reserve(comms.size());
  envs.reserve(comms.size());
  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto device = comms[i].logical_device().underlying_device();
    const auto rank   = static_cast<cuda::std::size_t>(comms[i].rank());

    in.emplace_back(cuda::make_device_buffer<T>(streams[i], device, values_by_rank[rank]));
    offsets.emplace_back(cuda::make_device_buffer<offset_type>(streams[i], device, offsets_by_rank[rank]));
    out.emplace_back(cuda::make_device_buffer<T>(streams[i], device, sentinel_values));
    envs.emplace_back(::cuda::std::execution::env{::cuda::stream_ref{streams[i]}});
  }

  do_segmented_reduce(comms, envs, in, num_segments, offsets, out, init, ident, Op{});

  for (cuda::std::size_t i = 0; i < out.size(); ++i)
  {
    INFO("device = " << i);

    const auto expected =
      cuda::make_buffer<T>(out[i].stream(), cuda::mr::legacy_pinned_memory_resource{}, sentinel_values);

    REQUIRE_THAT(out[i], Equals(expected));
  }
}
