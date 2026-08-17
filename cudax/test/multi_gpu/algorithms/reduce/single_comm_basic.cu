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

#include <cuda/experimental/__multi_gpu/algorithm/reduce/reduce.h>

#include <exception>
#include <future>
#include <numeric>
#include <string>
#include <vector>

#include <algorithm_common.h>
#include <nccl_test_common.h>
#include <testing.cuh>

#include "reduce_common.cuh"
#include <c2h/catch2_test_helper.h>

namespace
{
// Drive the reduction through the single-communicator overload of `reduce`, one thread per
// rank. That overload opens its own NCCL group on a single communicator, so issuing the
// per-rank calls serially on one thread would deadlock at `ncclGroupEnd`. Running each rank on
// its own thread lets the per-thread groups rendezvous across ranks. Only the `reduce` call
// happens on the worker threads; every Catch2 assertion runs on the main thread after the
// join, since the assertion macros are not safe to fire concurrently.
template <class Env, class T, class Op>
void do_reduce_threaded(
  cuda::std::span<cudax::nccl_communicator_ref> comms,
  std::vector<Env>& envs,
  std::vector<cuda::device_buffer<T>>& in,
  std::vector<cuda::device_buffer<T>>& out,
  const T& init,
  const T& ident,
  Op op)
{
  const auto in_copy = in;

  INFO("init = " << init);
  INFO("ident = " << ident);

  run_threaded(comms.size(), [&](cuda::std::size_t i) {
    cudax::reduce(cudax::broadcasted, comms[i], envs[i], in[i].begin(), in[i].size(), out[i].begin(), init, op, ident);
  });

  // Reduction call should not modify the inputs in any ways
  REQUIRE(in.size() == in_copy.size());
  for (cuda::std::size_t i = 0; i < in.size(); ++i)
  {
    INFO("device = " << i);
    REQUIRE_THAT(in[i], Equals(in_copy[i]));
  }
}
} // namespace

MULTI_GPU_TEST("reduce single-comm documentation example", c2h::type_list<int>)
{
  auto comms = this->communicators();

  if (comms.size() != 2)
  {
    SKIP("The reduce documentation example requires exactly two local GPUs");
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

    //! [reduce_single_range]
    constexpr cuda::std::array input_values{1, 2};
    const auto device = communicator.logical_device().underlying_device();

    auto input  = cuda::make_device_buffer<int>(stream, device, input_values);
    auto output = cuda::make_device_buffer<int>(stream, device, 1, cuda::no_init);

    cudax::reduce(cudax::broadcasted, communicator, env, input.begin(), input.size(), output.begin(), /*__init=*/0);

    // Every rank contributes {1, 2}, so the reduction over all ranks is 3 * nranks. `reduce`
    // broadcasts the result, so every rank sees the same value.
    const auto expected =
      cuda::make_buffer<int>(output.stream(), cuda::mr::legacy_pinned_memory_resource{}, 1, 3 * communicator.size());
    //! [reduce_single_range]

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

MULTI_GPU_TEST("reduce single-comm, one element per rank", value_types, operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  // Seed each reduction with a few hardcoded initializers. The init participates in the fold the
  // same way on host and device, so any value works for every operator under test.
  const T init     = make_value<T>(GENERATE(0, 1, -1, 5));
  const auto ident = get_identity<T, Op>();

  auto comms   = this->communicators();
  auto streams = nccl_test_util::make_streams();

  // Global rank `comms[i].rank()` contributes the single value `rank`. Each local rank also gets a
  // one-element output buffer and an environment carrying its stream, so the reduction is
  // stream-ordered on the correct device. `reference` mirrors the contributions of every global
  // rank so we can fold them on the host exactly like `reduce` does on the device.
  std::vector<cuda::device_buffer<T>> in;
  std::vector<cuda::device_buffer<T>> out;
  std::vector<decltype(::cuda::std::execution::env{::cuda::stream_ref{streams[0]}})> envs;

  in.reserve(comms.size());
  out.reserve(comms.size());
  envs.reserve(comms.size());
  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto values = {make_value<T>(comms[i].rank())};
    in.emplace_back(cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), values));
    out.emplace_back(
      cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), 1, cuda::no_init));
    envs.emplace_back(::cuda::std::execution::env{::cuda::stream_ref{streams[i]}});
  }

  do_reduce_threaded(comms, envs, in, out, init, ident, Op{});

  const T expected = [&] {
    std::vector<T> reference;

    reference.reserve(comms.front().size());
    for (int r = 0; r < comms.front().size(); ++r)
    {
      reference.push_back(make_value<T>(r));
    }

    return std::accumulate(reference.begin(), reference.end(), init, Op{});
  }();

  for (const auto& buf : out)
  {
    const auto exp = cuda::make_buffer(buf.stream(), cuda::mr::legacy_pinned_memory_resource{}, 1, expected);

    REQUIRE_THAT(buf, Equals(exp));
  }
}

MULTI_GPU_TEST("reduce single-comm, multiple elements per rank", value_types, operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  // Seed each reduction with a few hardcoded initializers. The init participates in the fold the
  // same way on host and device, so any value works for every operator under test.
  const T init     = make_value<T>(GENERATE(0, 1, -1, 5));
  const auto ident = get_identity<T, Op>();

  auto comms   = this->communicators();
  auto streams = nccl_test_util::make_streams();

  // Global rank `comms[i].rank()` contributes ten copies of `rank`. `reduce` first does a local
  // CUB reduction of each rank's range, then combines the partials across ranks. Each local rank
  // also gets a one-element output buffer and an environment carrying its stream. `reference`
  // mirrors every global rank's ten contributions for the host-side fold.
  std::vector<cuda::device_buffer<T>> in;
  std::vector<cuda::device_buffer<T>> out;
  std::vector<decltype(::cuda::std::execution::env{::cuda::stream_ref{streams[0]}})> envs;

  in.reserve(comms.size());
  out.reserve(comms.size());
  envs.reserve(comms.size());

  constexpr auto values_per_rank = 10;
  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto v = make_value<T>(comms[i].rank());
    const std::vector<T> values(values_per_rank, v);

    in.emplace_back(cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), values));
    out.emplace_back(
      cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), 1, cuda::no_init));
    envs.emplace_back(::cuda::std::execution::env{::cuda::stream_ref{streams[i]}});
  }

  do_reduce_threaded(comms, envs, in, out, init, ident, Op{});

  const T expected = [&] {
    std::vector<T> reference;

    reference.reserve(comms.front().size() * values_per_rank);
    for (int r = 0; r < comms.front().size(); ++r)
    {
      const auto v = make_value<T>(r);

      reference.insert(reference.end(), values_per_rank, v);
    }

    return std::accumulate(reference.begin(), reference.end(), init, Op{});
  }();

  for (const auto& buf : out)
  {
    const auto exp = cuda::make_buffer(buf.stream(), cuda::mr::legacy_pinned_memory_resource{}, 1, expected);

    REQUIRE_THAT(buf, Equals(exp));
  }
}

MULTI_GPU_TEST("reduce single-comm, some ranks empty", value_types, operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  const T init     = make_value<T>(GENERATE(0, 1, -1, 5));
  const auto ident = get_identity<T, Op>();

  auto comms   = this->communicators();
  auto streams = nccl_test_util::make_streams();

  // Even global ranks contribute ten copies of `rank`; odd global ranks contribute an empty input
  // range. Rank 0 (the reduction root) is always non-empty. `reduce` must treat an empty rank as
  // contributing nothing, exactly like `std::accumulate` over the surviving elements. `reference`
  // mirrors that for the host-side fold.
  std::vector<cuda::device_buffer<T>> in;
  std::vector<cuda::device_buffer<T>> out;
  std::vector<decltype(::cuda::std::execution::env{::cuda::stream_ref{streams[0]}})> envs;

  in.reserve(comms.size());
  out.reserve(comms.size());
  envs.reserve(comms.size());

  constexpr auto values_per_rank = 10;
  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto rank = comms[i].rank();
    if (rank % 2 == 0)
    {
      const std::vector<T> values(values_per_rank, make_value<T>(rank));
      in.emplace_back(cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), values));
    }
    else
    {
      in.emplace_back(cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device()));
    }
    out.emplace_back(
      cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), 1, cuda::no_init));
    envs.emplace_back(::cuda::std::execution::env{::cuda::stream_ref{streams[i]}});
  }

  do_reduce_threaded(comms, envs, in, out, init, ident, Op{});

  const T expected = [&] {
    std::vector<T> reference;

    reference.reserve(comms.front().size() * values_per_rank);
    for (int r = 0; r < comms.front().size(); ++r)
    {
      if (r % 2 == 0)
      {
        reference.insert(reference.end(), values_per_rank, make_value<T>(r));
      }
    }

    return std::accumulate(reference.begin(), reference.end(), init, Op{});
  }();

  for (const auto& buf : out)
  {
    const auto exp = cuda::make_buffer(buf.stream(), cuda::mr::legacy_pinned_memory_resource{}, 1, expected);

    REQUIRE_THAT(buf, Equals(exp));
  }
}

MULTI_GPU_TEST("reduce single-comm, all ranks empty", value_types, operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  const auto init  = make_value<T>(GENERATE(0, 1, -1, 5));
  const auto ident = get_identity<T, Op>();

  auto comms   = this->communicators();
  auto streams = nccl_test_util::make_streams();

  // No rank contributes any element. Reducing nothing seeded by `init` is just `init`, so every
  // output must equal `init` regardless of the operator.
  std::vector<cuda::device_buffer<T>> in;
  std::vector<cuda::device_buffer<T>> out;
  std::vector<decltype(::cuda::std::execution::env{::cuda::stream_ref{streams[0]}})> envs;

  in.reserve(comms.size());
  out.reserve(comms.size());
  envs.reserve(comms.size());
  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    in.emplace_back(cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device()));
    out.emplace_back(
      cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), 1, cuda::no_init));
    envs.emplace_back(::cuda::std::execution::env{::cuda::stream_ref{streams[i]}});
  }

  do_reduce_threaded(comms, envs, in, out, init, ident, Op{});

  // Reducing nothing seeded by `init` yields `init`, exactly like `std::accumulate` over an empty
  // range.
  const T expected = init;

  for (const auto& buf : out)
  {
    const auto exp = cuda::make_buffer(buf.stream(), cuda::mr::legacy_pinned_memory_resource{}, 1, expected);

    REQUIRE_THAT(buf, Equals(exp));
  }
}
