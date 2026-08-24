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
#include <cuda/std/span>

#include <cuda/experimental/__multi_gpu/algorithm/reduce/segmented_reduce.h>

#include <exception>
#include <future>
#include <numeric>
#include <string>
#include <vector>

#include <algorithm_common.h>
#include <nccl_test_common.h>
#include <testing.cuh>

#include "segmented_reduce_common.cuh"

namespace
{
// Drive the reduction through the single-communicator overload of `segmented_reduce`, one thread
// per rank. That overload opens its own NCCL group on a single communicator, so issuing the
// per-rank calls serially on one thread would deadlock at `ncclGroupEnd`. Running each rank on
// its own thread lets the per-thread groups rendezvous across ranks. Only the `segmented_reduce`
// call happens on the worker threads; every Catch2 assertion runs on the main thread after the
// join, since the assertion macros are not safe to fire concurrently.
template <class Env, class T, class Op>
void do_segmented_reduce_threaded(
  cuda::std::span<cudax::nccl_communicator_ref> comms,
  std::vector<Env>& envs,
  std::vector<cuda::device_buffer<T>>& in,
  cuda::std::size_t num_segments,
  std::vector<cuda::device_buffer<offset_type>>& offsets,
  std::vector<cuda::device_buffer<T>>& out,
  const T& init,
  const T& ident,
  Op op)
{
  const auto in_copy      = in;
  const auto offsets_copy = offsets;

  INFO("init = " << init);
  INFO("ident = " << ident);

  run_threaded(comms.size(), [&](cuda::std::size_t i) {
    // Segment `s` covers [begin_offsets[s], end_offsets[s]), so the end offsets are just the
    // begin offsets shifted by one.
    cudax::segmented_reduce(
      cudax::broadcasted,
      comms[i],
      envs[i],
      in[i].begin(),
      num_segments,
      offsets[i].begin(),
      offsets[i].begin() + 1,
      out[i].begin(),
      init,
      op,
      ident);
  });

  // The reduction must not modify the inputs in any way.
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

MULTI_GPU_TEST("segmented_reduce single-comm documentation example", c2h::type_list<int>)
{
  auto comms = this->communicators();

  if (comms.size() != 2)
  {
    SKIP("The segmented_reduce documentation example requires exactly two local GPUs");
  }

  auto streams_owned = nccl_test_util::make_streams();
  auto streams       = std::vector<cuda::stream_ref>{streams_owned.begin(), streams_owned.end()};

  // Must be pre-allocated since it is written to by threads
  std::vector<std::string> failed(comms.front().size());

  // Every communicator rank must invoke the collective concurrently.
  run_threaded(comms.size(), [&](cuda::std::size_t i) {
    auto& communicator = comms[i];
    auto& stream       = streams[i];
    // Rename the stream to env for the example
    auto& env = stream;

    //! [segmented_reduce_single_range]
    // Two segments per rank: {1, 2} and {3, 4, 5}.
    constexpr cuda::std::array input_values{1, 2, 3, 4, 5};
    constexpr cuda::std::array offset_values{0, 2, 5};
    constexpr cuda::std::size_t num_segments = offset_values.size() - 1;

    const auto device = communicator.logical_device().underlying_device();

    auto input   = cuda::make_device_buffer<int>(stream, device, input_values);
    auto offsets = cuda::make_device_buffer<int>(stream, device, offset_values);
    auto output  = cuda::make_device_buffer<int>(stream, device, num_segments, cuda::no_init);

    cudax::segmented_reduce(
      cudax::broadcasted,
      communicator,
      env,
      input.begin(),
      num_segments,
      // Segment `s` covers [begin_offsets[s], end_offsets[s]), so the end offsets are just the
      // begin offsets shifted by one.
      offsets.begin(),
      offsets.begin() + 1,
      output.begin(),
      /*__init=*/0);

    // Every rank contributes the same two segments, so segment 0 sums to 3 * nranks and segment
    // 1 to 12 * nranks. `segmented_reduce` broadcasts the result, so every rank sees the same
    // values.
    const auto nranks = communicator.size();
    const std::vector<int> expected_values{3 * nranks, 12 * nranks};
    const auto expected =
      cuda::make_buffer<int>(output.stream(), cuda::mr::legacy_pinned_memory_resource{}, expected_values);
    //! [segmented_reduce_single_range]

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

MULTI_GPU_TEST("segmented_reduce single-comm, one segment of one element per rank", value_types, operators)
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

  do_segmented_reduce_threaded(comms, envs, in, num_segments, offsets, out, init, ident, Op{});
  check_outputs(out, values_by_rank, offsets_by_rank, num_segments, init, Op{});
}

MULTI_GPU_TEST("segmented_reduce single-comm, multiple equal-sized segments per rank", value_types, operators)
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

  do_segmented_reduce_threaded(comms, envs, in, num_segments, offsets, out, init, ident, Op{});
  check_outputs(out, values_by_rank, offsets_by_rank, num_segments, init, Op{});
}

MULTI_GPU_TEST("segmented_reduce single-comm, ragged segments per rank", value_types, operators)
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

  do_segmented_reduce_threaded(comms, envs, in, num_segments, offsets, out, init, ident, Op{});
  check_outputs(out, values_by_rank, offsets_by_rank, num_segments, init, Op{});
}

MULTI_GPU_TEST("segmented_reduce single-comm, segment lengths differ across ranks", value_types, operators)
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

  do_segmented_reduce_threaded(comms, envs, in, num_segments, offsets, out, init, ident, Op{});
  check_outputs(out, values_by_rank, offsets_by_rank, num_segments, init, Op{});
}

MULTI_GPU_TEST("segmented_reduce single-comm, some segments empty", value_types, operators)
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

  do_segmented_reduce_threaded(comms, envs, in, num_segments, offsets, out, init, ident, Op{});
  check_outputs(out, values_by_rank, offsets_by_rank, num_segments, init, Op{});
}

MULTI_GPU_TEST("segmented_reduce single-comm, some ranks empty", value_types, operators)
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

  do_segmented_reduce_threaded(comms, envs, in, num_segments, offsets, out, init, ident, Op{});
  check_outputs(out, values_by_rank, offsets_by_rank, num_segments, init, Op{});
}

MULTI_GPU_TEST("segmented_reduce single-comm, all ranks empty", value_types, operators)
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

  do_segmented_reduce_threaded(comms, envs, in, num_segments, offsets, out, init, ident, Op{});
  check_outputs(out, values_by_rank, offsets_by_rank, num_segments, init, Op{});
}

MULTI_GPU_TEST("segmented_reduce single-comm, zero segments", value_types, operators)
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

  do_segmented_reduce_threaded(comms, envs, in, num_segments, offsets, out, init, ident, Op{});

  for (cuda::std::size_t i = 0; i < out.size(); ++i)
  {
    INFO("device = " << i);

    const auto expected =
      cuda::make_buffer<T>(out[i].stream(), cuda::mr::legacy_pinned_memory_resource{}, sentinel_values);

    REQUIRE_THAT(out[i], Equals(expected));
  }
}
