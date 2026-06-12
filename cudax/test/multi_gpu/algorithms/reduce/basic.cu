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
#include <cuda/std/cstdint>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/std/type_traits>

#include <cuda/experimental/__multi_gpu/algorithm/reduce/reduce.h>

#include <numeric>
#include <vector>

#include <testing.cuh>

#include "../../communicators/nccl/nccl_test_helpers.cuh"

namespace
{
using custom_value = c2h::custom_type_t<c2h::accumulateable_t, c2h::less_comparable_t, c2h::equal_comparable_t>;
static_assert(cudax::nccl_transportable<custom_value>);

template <typename T>
T make_value(int i)
{
  return static_cast<T>(i);
}

template <>
custom_value make_value<>(int i)
{
  return {static_cast<cuda::std::size_t>(i), static_cast<cuda::std::size_t>(i)};
};

// A custom reduction operator equivalent to `cuda::std::plus<>`.
struct custom_plus
{
  template <class T>
  _CCCL_HOST_DEVICE constexpr T operator()(const T& lhs, const T& rhs) const
  {
    return lhs + rhs;
  }
};

using value_types = c2h::type_list<cuda::std::int32_t, float, custom_value>;
using operators   = c2h::type_list<::cuda::std::plus<>, ::cuda::maximum<>, custom_plus>;

// One output iterator per local output buffer. Collected after `out` is fully built so the
// iterators do not dangle across reallocations.
template <class OutBuffers>
[[nodiscard]] auto make_output_iterators(OutBuffers& out)
{
  std::vector<typename cuda::std::remove_cvref_t<decltype(out.front())>::iterator> outputs;

  outputs.reserve(out.size());
  for (auto& buf : out)
  {
    outputs.push_back(buf.begin());
  }
  return outputs;
}

// Run the full reduction, wait for it to finish, and check that `reduce` left its argument ranges
// untouched. This boilerplate is identical for every test regardless of how the inputs are shaped.
template <class Comms, class Envs, class In, class Outputs, class T, class Op, class Streams>
void do_reduce(Comms& comms, Envs& envs, In& in, Outputs& outputs, const T& init, Op op, Streams& streams)
{
  using value_type = typename cuda::std::remove_cvref_t<decltype(in.front())>::value_type;

  // cuda::std::execution::env has no operator==, so we can only compare the sizes.
  const auto envs_size = envs.size();

  auto pool = cuda::mr::legacy_pinned_memory_resource{};
  std::vector<cuda::host_buffer<value_type>> in_copy;
  in_copy.reserve(in.size());
  for (const auto& buf : in)
  {
    in_copy.emplace_back(cuda::make_buffer(buf.stream(), pool, buf));
  }

  const auto outputs_copy = outputs;

  cudax::reduce(comms, envs, in, outputs, init, op);

  for (auto& stream : streams)
  {
    stream.sync();
  }

  REQUIRE(envs.size() == envs_size);
  for (cuda::std::size_t i = 0; i < in.size(); ++i)
  {
    const auto actual = cuda::make_buffer(in[i].stream(), pool, in[i]);
    REQUIRE_THAT(actual, Equals(in_copy[i]));
  }
  REQUIRE(outputs == outputs_copy);
}
} // namespace

MGMN_TEST("reduce, one element per rank", value_types, operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  // Seed each reduction with a few hardcoded initializers. The init participates in the fold the
  // same way on host and device, so any value works for every operator under test.
  const T init = make_value<T>(GENERATE(0, 1, -1, 5));

  auto comms   = this->communicators();
  auto streams = nccl_test_util::make_streams();

  // Global rank `comms[i].rank()` contributes the single value `rank`. Each local rank also gets a
  // one-element output buffer and an environment carrying its stream, so the reduction is
  // stream-ordered on the correct device. `reference` mirrors the contributions of every global
  // rank so we can fold them on the host exactly like `reduce` does on the device.
  std::vector<cuda::device_buffer<T>> in;
  std::vector<cuda::device_buffer<T>> out;
  std::vector<decltype(::cuda::std::execution::env{::cuda::stream_ref{streams[0]}})> envs;
  std::vector<T> reference;

  in.reserve(comms.size());
  out.reserve(comms.size());
  envs.reserve(comms.size());
  for (int i = 0; i < comms.size(); ++i)
  {
    const auto values = {make_value<T>(comms[i].rank())};
    in.emplace_back(cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), values));
    out.emplace_back(
      cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), 1, cuda::no_init));
    envs.emplace_back(::cuda::std::execution::env{::cuda::stream_ref{streams[i]}});
  }
  for (int r = 0; r < comms.front().size(); ++r)
  {
    reference.push_back(make_value<T>(r));
  }

  auto outputs = make_output_iterators(out);

  do_reduce(comms, envs, in, outputs, init, Op{}, streams);

  const T expected = std::accumulate(reference.begin(), reference.end(), init, Op{});

  for (const auto& buf : out)
  {
    auto pool         = cuda::mr::legacy_pinned_memory_resource{};
    const auto actual = cuda::make_buffer(buf.stream(), pool, buf);

    REQUIRE(actual.size() == 1);
    REQUIRE(actual.front() == expected);
  }
}

MGMN_TEST("reduce, multiple elements per rank", value_types, operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  // Seed each reduction with a few hardcoded initializers. The init participates in the fold the
  // same way on host and device, so any value works for every operator under test.
  const T init = make_value<T>(GENERATE(0, 1, -1, 5));

  auto comms   = this->communicators();
  auto streams = nccl_test_util::make_streams();

  // Global rank `comms[i].rank()` contributes `{rank, rank, rank}`. `reduce` first does a local CUB
  // reduction of each rank's range, then combines the partials across ranks. Each local rank also
  // gets a one-element output buffer and an environment carrying its stream. `reference` mirrors
  // every global rank's three contributions for the host-side fold.
  std::vector<cuda::device_buffer<T>> in;
  std::vector<cuda::device_buffer<T>> out;
  std::vector<decltype(::cuda::std::execution::env{::cuda::stream_ref{streams[0]}})> envs;
  std::vector<T> reference;

  in.reserve(comms.size());
  out.reserve(comms.size());
  envs.reserve(comms.size());
  for (int i = 0; i < comms.size(); ++i)
  {
    const auto v      = make_value<T>(comms[i].rank());
    const auto values = {v, v, v};
    in.emplace_back(cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), values));
    out.emplace_back(
      cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), 1, cuda::no_init));
    envs.emplace_back(::cuda::std::execution::env{::cuda::stream_ref{streams[i]}});
  }
  for (int r = 0; r < comms.front().size(); ++r)
  {
    const auto v = make_value<T>(r);
    reference.insert(reference.end(), {v, v, v});
  }

  auto outputs = make_output_iterators(out);

  do_reduce(comms, envs, in, outputs, init, Op{}, streams);

  const T expected = std::accumulate(reference.begin(), reference.end(), init, Op{});

  for (const auto& buf : out)
  {
    auto pool         = cuda::mr::legacy_pinned_memory_resource{};
    const auto actual = cuda::make_buffer(buf.stream(), pool, buf);

    REQUIRE(actual.size() == 1);
    REQUIRE(actual.front() == expected);
  }
}

MGMN_TEST("reduce, some ranks empty", value_types, operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  const T init = make_value<T>(GENERATE(0, 1, -1, 5));

  auto comms   = this->communicators();
  auto streams = nccl_test_util::make_streams();

  // Even global ranks contribute two copies of `rank`; odd global ranks contribute an empty input
  // range. Rank 0 (the reduction root) is always non-empty. `reduce` must treat an empty rank as
  // contributing nothing, exactly like `std::accumulate` over the surviving elements. `reference`
  // mirrors that for the host-side fold.
  std::vector<cuda::device_buffer<T>> in;
  std::vector<cuda::device_buffer<T>> out;
  std::vector<decltype(::cuda::std::execution::env{::cuda::stream_ref{streams[0]}})> envs;
  std::vector<T> reference;

  in.reserve(comms.size());
  out.reserve(comms.size());
  envs.reserve(comms.size());
  for (int i = 0; i < comms.size(); ++i)
  {
    const auto rank = comms[i].rank();
    if (rank % 2 == 0)
    {
      const auto values = {make_value<T>(rank), make_value<T>(rank)};
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

  for (int r = 0; r < comms.front().size(); r += 2)
  {
    reference.push_back(make_value<T>(r));
    reference.push_back(make_value<T>(r));
  }

  auto outputs = make_output_iterators(out);

  do_reduce(comms, envs, in, outputs, init, Op{}, streams);

  const T expected = std::accumulate(reference.begin(), reference.end(), init, Op{});

  for (const auto& buf : out)
  {
    auto pool         = cuda::mr::legacy_pinned_memory_resource{};
    const auto actual = cuda::make_buffer(buf.stream(), pool, buf);

    REQUIRE(actual.size() == 1);
    REQUIRE(actual.front() == expected);
  }
}

MGMN_TEST("reduce, all ranks empty", value_types, operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  const T init = make_value<T>(GENERATE(0, 1, -1, 5));

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
  for (int i = 0; i < comms.size(); ++i)
  {
    in.emplace_back(cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device()));
    out.emplace_back(
      cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), 1, cuda::no_init));
    envs.emplace_back(::cuda::std::execution::env{::cuda::stream_ref{streams[i]}});
  }

  auto outputs = make_output_iterators(out);

  do_reduce(comms, envs, in, outputs, init, Op{}, streams);

  // Reducing nothing seeded by `init` yields `init`, exactly like `std::accumulate` over an empty
  // range.
  const T expected = init;

  for (const auto& buf : out)
  {
    auto pool         = cuda::mr::legacy_pinned_memory_resource{};
    const auto actual = cuda::make_buffer(buf.stream(), pool, buf);

    REQUIRE(actual.size() == 1);
    REQUIRE(actual.front() == expected);
  }
}
