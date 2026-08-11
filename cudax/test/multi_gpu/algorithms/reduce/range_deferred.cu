//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/argument>
#include <cuda/buffer>
#include <cuda/memory_resource>
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/std/ranges>

#include <cuda/experimental/__multi_gpu/algorithm/reduce/reduce.h>

#include <algorithm>
#include <numeric>
#include <vector>

#include <algorithm_common.h>
#include <nccl_test_common.h>
#include <testing.cuh>

#include "reduce_common.cuh"
#include <c2h/catch2_test_helper.h>

namespace
{
using count_type = cuda::std::int32_t;

// Run the full reduction, wait for it to finish, and check that `reduce` left its argument ranges
// untouched. This boilerplate is identical for every test regardless of how the inputs are shaped.
template <class Env, class T, class Op>
void do_reduce_deferred(
  cuda::std::span<cudax::nccl_communicator_ref> comms,
  const std::vector<Env>& envs,
  std::vector<cuda::device_buffer<T>>& in,
  std::vector<cuda::device_buffer<count_type>>& num_items,
  std::vector<cuda::device_buffer<T>>& out,
  const T& init,
  const T& ident,
  Op op)
{
  const auto envs_size = envs.size();
  const auto in_copy   = in;

  INFO("init = " << init);
  INFO("ident = " << ident);

  cudax::reduce(
    cudax::broadcasted,
    comms,
    envs,
    in | cuda::std::views::transform(cuda::std::ranges::begin),
    num_items | cuda::std::views::transform([](auto& buf) {
      return cuda::args::deferred{buf.begin()};
    }),
    out | cuda::std::views::transform(cuda::std::ranges::begin),
    init,
    op,
    ident);

  // cuda::std::execution::env has no operator==, so we can only compare the sizes.
  REQUIRE(envs.size() == envs_size);
  // Reduction call should not modify the inputs in any ways
  REQUIRE(in.size() == in_copy.size());
  for (cuda::std::size_t i = 0; i < in.size(); ++i)
  {
    INFO("device = " << i);
    REQUIRE_THAT(in[i], Equals(in_copy[i]));
  }
}
} // namespace

MULTI_GPU_TEST("reduce deferred documentation example", c2h::type_list<int>)
{
  auto comms = this->communicators();

  if (comms.size() != 2)
  {
    SKIP("The reduce documentation example requires exactly two local GPUs");
  }

  auto streams_owned = nccl_test_util::make_streams();
  // Convert to stream_ref directly, cuda::stream on their own cant be passed directly to CUB
  auto streams = std::vector<cuda::stream_ref>{streams_owned.begin(), streams_owned.end()};

  //! [reduce_deferred]
  constexpr cuda::std::array input_values{1, 2, 3, 4};
  // Only the first two elements take part in the reduction. A real caller would have a
  // preceding device-side step write this count.
  constexpr cuda::std::array count_values{2};

  std::vector<cuda::device_buffer<int>> inputs;
  std::vector<cuda::device_buffer<int>> counts;
  std::vector<cuda::device_buffer<int>> outputs;

  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto device = comms[i].logical_device().underlying_device();

    inputs.emplace_back(cuda::make_device_buffer<int>(streams[i], device, input_values));
    counts.emplace_back(cuda::make_device_buffer<int>(streams[i], device, count_values));
    outputs.emplace_back(cuda::make_device_buffer<int>(streams[i], device, 1, cuda::no_init));
  }

  cudax::reduce(
    cudax::broadcasted,
    comms,
    streams,
    inputs | cuda::std::views::transform(cuda::std::ranges::begin),
    // The count is read on the device in stream order, so it need not be known on the host
    // when `reduce` is called.
    counts | cuda::std::views::transform([](auto& buf) {
      return cuda::args::deferred{buf.begin()};
    }),
    outputs | cuda::std::views::transform(cuda::std::ranges::begin),
    /*__init=*/0);

  // Every rank contributes the first two of its four values, so every output holds
  // `(1 + 2) * nranks`.
  const auto expected_value = 3 * comms.front().size();
  //! [reduce_deferred]

  const auto expected_0 =
    cuda::make_buffer<int>(outputs[0].stream(), cuda::mr::legacy_pinned_memory_resource{}, 1, expected_value);
  const auto expected_1 =
    cuda::make_buffer<int>(outputs[1].stream(), cuda::mr::legacy_pinned_memory_resource{}, 1, expected_value);
  REQUIRE_THAT(outputs[0], Equals(expected_0));
  REQUIRE_THAT(outputs[1], Equals(expected_1));
}

MULTI_GPU_TEST("reduce with deferred counts, one element per rank", value_types, operators)
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
  std::vector<cuda::device_buffer<count_type>> num_items;
  std::vector<cuda::device_buffer<T>> out;
  std::vector<decltype(::cuda::std::execution::env{::cuda::stream_ref{streams[0]}})> envs;

  in.reserve(comms.size());
  num_items.reserve(comms.size());
  out.reserve(comms.size());
  envs.reserve(comms.size());
  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto values = {make_value<T>(comms[i].rank())};
    const auto counts = {count_type{1}};

    in.emplace_back(cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), values));
    num_items.emplace_back(
      cuda::make_device_buffer<count_type>(streams[i], comms[i].logical_device().underlying_device(), counts));
    out.emplace_back(
      cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), 1, cuda::no_init));
    envs.emplace_back(::cuda::std::execution::env{::cuda::stream_ref{streams[i]}});
  }

  do_reduce_deferred(comms, envs, in, num_items, out, init, ident, Op{});

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

MULTI_GPU_TEST("reduce with deferred counts, multiple elements per rank", value_types, operators)
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
  std::vector<cuda::device_buffer<count_type>> num_items;
  std::vector<cuda::device_buffer<T>> out;
  std::vector<decltype(::cuda::std::execution::env{::cuda::stream_ref{streams[0]}})> envs;

  in.reserve(comms.size());
  num_items.reserve(comms.size());
  out.reserve(comms.size());
  envs.reserve(comms.size());

  constexpr auto values_per_rank = 10;
  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto v = make_value<T>(comms[i].rank());
    const std::vector<T> values(values_per_rank, v);
    const auto counts = {count_type{values_per_rank}};

    in.emplace_back(cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), values));
    num_items.emplace_back(
      cuda::make_device_buffer<count_type>(streams[i], comms[i].logical_device().underlying_device(), counts));
    out.emplace_back(
      cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), 1, cuda::no_init));
    envs.emplace_back(::cuda::std::execution::env{::cuda::stream_ref{streams[i]}});
  }

  do_reduce_deferred(comms, envs, in, num_items, out, init, ident, Op{});

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

MULTI_GPU_TEST("reduce with deferred counts, some ranks empty", value_types, operators)
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
  std::vector<cuda::device_buffer<count_type>> num_items;
  std::vector<cuda::device_buffer<T>> out;
  std::vector<decltype(::cuda::std::execution::env{::cuda::stream_ref{streams[0]}})> envs;

  in.reserve(comms.size());
  num_items.reserve(comms.size());
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
    const auto counts = {rank % 2 == 0 ? count_type{values_per_rank} : count_type{0}};

    num_items.emplace_back(
      cuda::make_device_buffer<count_type>(streams[i], comms[i].logical_device().underlying_device(), counts));
    out.emplace_back(
      cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), 1, cuda::no_init));
    envs.emplace_back(::cuda::std::execution::env{::cuda::stream_ref{streams[i]}});
  }

  do_reduce_deferred(comms, envs, in, num_items, out, init, ident, Op{});

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

MULTI_GPU_TEST("reduce with deferred counts, all ranks empty", value_types, operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  const auto init  = make_value<T>(GENERATE(0, 1, -1, 5));
  const auto ident = get_identity<T, Op>();

  auto comms   = this->communicators();
  auto streams = nccl_test_util::make_streams();

  // No rank contributes any element, so every deferred count is zero. Reducing nothing seeded by
  // `init` is just `init`, so every output must equal `init` regardless of the operator.
  std::vector<cuda::device_buffer<T>> in;
  std::vector<cuda::device_buffer<count_type>> num_items;
  std::vector<cuda::device_buffer<T>> out;
  std::vector<decltype(::cuda::std::execution::env{::cuda::stream_ref{streams[0]}})> envs;

  in.reserve(comms.size());
  num_items.reserve(comms.size());
  out.reserve(comms.size());
  envs.reserve(comms.size());
  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto counts = {count_type{0}};

    in.emplace_back(cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device()));
    num_items.emplace_back(
      cuda::make_device_buffer<count_type>(streams[i], comms[i].logical_device().underlying_device(), counts));
    out.emplace_back(
      cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), 1, cuda::no_init));
    envs.emplace_back(::cuda::std::execution::env{::cuda::stream_ref{streams[i]}});
  }

  do_reduce_deferred(comms, envs, in, num_items, out, init, ident, Op{});

  // Reducing nothing seeded by `init` yields `init`, exactly like `std::accumulate` over an empty
  // range.
  const T expected = init;

  for (const auto& buf : out)
  {
    const auto exp = cuda::make_buffer(buf.stream(), cuda::mr::legacy_pinned_memory_resource{}, 1, expected);

    REQUIRE_THAT(buf, Equals(exp));
  }
}

MULTI_GPU_TEST("reduce with deferred counts smaller than the input range", value_types, operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  const T init     = make_value<T>(GENERATE(0, 1, -1, 5));
  const auto ident = get_identity<T, Op>();

  auto comms   = this->communicators();
  auto streams = nccl_test_util::make_streams();

  // The tail past `counted_per_rank` holds poison values that would change the result of every
  // operator under test, so a reduction that used the buffer size instead of the deferred count
  // cannot produce the expected value.
  std::vector<cuda::device_buffer<T>> in;
  std::vector<cuda::device_buffer<count_type>> num_items;
  std::vector<cuda::device_buffer<T>> out;
  std::vector<decltype(::cuda::std::execution::env{::cuda::stream_ref{streams[0]}})> envs;

  in.reserve(comms.size());
  num_items.reserve(comms.size());
  out.reserve(comms.size());
  envs.reserve(comms.size());

  constexpr auto values_per_rank  = 10;
  constexpr auto counted_per_rank = 4;
  constexpr auto poison           = 1000;
  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto rank = comms[i].rank();
    std::vector<T> values(values_per_rank, make_value<T>(poison));

    std::fill(values.begin(), values.begin() + counted_per_rank, make_value<T>(rank));

    const auto counts = {count_type{counted_per_rank}};

    in.emplace_back(cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), values));
    num_items.emplace_back(
      cuda::make_device_buffer<count_type>(streams[i], comms[i].logical_device().underlying_device(), counts));
    out.emplace_back(
      cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), 1, cuda::no_init));
    envs.emplace_back(::cuda::std::execution::env{::cuda::stream_ref{streams[i]}});
  }

  do_reduce_deferred(comms, envs, in, num_items, out, init, ident, Op{});

  const T expected = [&] {
    std::vector<T> reference;

    reference.reserve(comms.front().size() * counted_per_rank);
    for (int r = 0; r < comms.front().size(); ++r)
    {
      reference.insert(reference.end(), counted_per_rank, make_value<T>(r));
    }

    return std::accumulate(reference.begin(), reference.end(), init, Op{});
  }();

  for (const auto& buf : out)
  {
    const auto exp = cuda::make_buffer(buf.stream(), cuda::mr::legacy_pinned_memory_resource{}, 1, expected);

    REQUIRE_THAT(buf, Equals(exp));
  }
}
