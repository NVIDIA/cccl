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

#include <algorithm_common.h>
#include <nccl_test_common.h>
#include <testing.cuh>

namespace
{
struct custom_plus
{
  template <class T>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr T operator()(const T& lhs, const T& rhs) const
  {
    return lhs + rhs;
  }
};

using custom_value = c2h::custom_type_t<c2h::accumulateable_t, c2h::less_comparable_t, c2h::equal_comparable_t>;
using value_types  = c2h::type_list<cuda::std::int32_t, float, custom_value>;
using operators    = c2h::type_list<::cuda::std::plus<>, ::cuda::maximum<>, custom_plus>;

static_assert(cudax::nccl_transportable<custom_value>);

template <typename T>
T make_value(int i)
{
  return static_cast<T>(i);
}

template <>
custom_value make_value<>(int i)
{
  custom_value ret{};

  ret.key = static_cast<std::size_t>(i);
  ret.val = static_cast<std::size_t>(i);
  return ret;
};

template <class T, class Op>
[[nodiscard]] T get_identity()
{
  if constexpr (cuda::std::is_same_v<Op, cuda::std::plus<>> || cuda::std::is_same_v<Op, custom_plus>)
  {
    return make_value<T>(0);
  }
  else if constexpr (cuda::std::is_same_v<Op, cuda::maximum<>>)
  {
    return cuda::std::numeric_limits<T>::lowest();
  }
  else
  {
    static_assert(cuda::std::__always_false_v<T, Op>, "Add handling");
  }
}

// Run the full reduction, wait for it to finish, and check that `reduce` left its argument ranges
// untouched. This boilerplate is identical for every test regardless of how the inputs are shaped.
template <class Env, class T, class Op>
void do_reduce(cuda::std::span<cudax::nccl_communicator_ref> comms,
               const std::vector<Env>& envs,
               std::vector<cuda::device_buffer<T>>& in,
               std::vector<typename cuda::device_buffer<T>::iterator>& outputs,
               const T& init,
               const T& ident,
               Op op)
{
  const auto envs_size    = envs.size();
  const auto in_copy      = in;
  const auto outputs_copy = outputs;

  INFO("init = " << init);
  INFO("ident = " << ident);

  cudax::reduce(comms, envs, in, outputs, init, op, ident);

  // cuda::std::execution::env has no operator==, so we can only compare the sizes.
  REQUIRE(envs.size() == envs_size);
  // Reduction call should not modify the inputs in any ways
  REQUIRE(in.size() == in_copy.size());
  for (cuda::std::size_t i = 0; i < in.size(); ++i)
  {
    INFO("device = " << i);
    REQUIRE_THAT(in[i], Equals(in_copy[i]));
  }
  REQUIRE_THAT(outputs, Catch::Matchers::Equals(outputs_copy));
}
} // namespace

MULTI_GPU_TEST("reduce, one element per rank", value_types, operators)
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

  auto outputs = make_output_iterators(out);

  do_reduce(comms, envs, in, outputs, init, ident, Op{});

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

MULTI_GPU_TEST("reduce, multiple elements per rank", value_types, operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  // Seed each reduction with a few hardcoded initializers. The init participates in the fold the
  // same way on host and device, so any value works for every operator under test.
  const T init     = make_value<T>(GENERATE(0, 1, -1, 5));
  const auto ident = get_identity<T, Op>();

  auto comms   = this->communicators();
  auto streams = nccl_test_util::make_streams();

  // Global rank `comms[i].rank()` contributes `{rank, rank, rank}`. `reduce` first does a local CUB
  // reduction of each rank's range, then combines the partials across ranks. Each local rank also
  // gets a one-element output buffer and an environment carrying its stream. `reference` mirrors
  // every global rank's three contributions for the host-side fold.
  std::vector<cuda::device_buffer<T>> in;
  std::vector<cuda::device_buffer<T>> out;
  std::vector<decltype(::cuda::std::execution::env{::cuda::stream_ref{streams[0]}})> envs;

  in.reserve(comms.size());
  out.reserve(comms.size());
  envs.reserve(comms.size());
  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto v      = make_value<T>(comms[i].rank());
    const auto values = {v, v, v};

    in.emplace_back(cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), values));
    out.emplace_back(
      cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), 1, cuda::no_init));
    envs.emplace_back(::cuda::std::execution::env{::cuda::stream_ref{streams[i]}});
  }

  auto outputs = make_output_iterators(out);

  do_reduce(comms, envs, in, outputs, init, ident, Op{});

  const T expected = [&] {
    std::vector<T> reference;

    reference.reserve(comms.front().size());
    for (int r = 0; r < comms.front().size(); ++r)
    {
      const auto v = make_value<T>(r);

      reference.insert(reference.end(), {v, v, v});
    }

    return std::accumulate(reference.begin(), reference.end(), init, Op{});
  }();

  for (const auto& buf : out)
  {
    const auto exp = cuda::make_buffer(buf.stream(), cuda::mr::legacy_pinned_memory_resource{}, 1, expected);

    REQUIRE_THAT(buf, Equals(exp));
  }
}

MULTI_GPU_TEST("reduce, some ranks empty", value_types, operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  const T init     = make_value<T>(GENERATE(0, 1, -1, 5));
  const auto ident = get_identity<T, Op>();

  auto comms   = this->communicators();
  auto streams = nccl_test_util::make_streams();

  // Even global ranks contribute two copies of `rank`; odd global ranks contribute an empty input
  // range. Rank 0 (the reduction root) is always non-empty. `reduce` must treat an empty rank as
  // contributing nothing, exactly like `std::accumulate` over the surviving elements. `reference`
  // mirrors that for the host-side fold.
  std::vector<cuda::device_buffer<T>> in;
  std::vector<cuda::device_buffer<T>> out;
  std::vector<decltype(::cuda::std::execution::env{::cuda::stream_ref{streams[0]}})> envs;

  in.reserve(comms.size());
  out.reserve(comms.size());
  envs.reserve(comms.size());
  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
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

  auto outputs = make_output_iterators(out);

  do_reduce(comms, envs, in, outputs, init, ident, Op{});

  const T expected = [&] {
    std::vector<T> reference;

    reference.reserve(comms.front().size());
    for (int r = 0; r < comms.front().size(); ++r)
    {
      if (r % 2 == 0)
      {
        reference.push_back(make_value<T>(r));
        reference.push_back(make_value<T>(r));
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

MULTI_GPU_TEST("reduce, all ranks empty", value_types, operators)
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

  auto outputs = make_output_iterators(out);

  do_reduce(comms, envs, in, outputs, init, ident, Op{});

  // Reducing nothing seeded by `init` yields `init`, exactly like `std::accumulate` over an empty
  // range.
  const T expected = init;

  for (const auto& buf : out)
  {
    const auto exp = cuda::make_buffer(buf.stream(), cuda::mr::legacy_pinned_memory_resource{}, 1, expected);

    REQUIRE_THAT(buf, Equals(exp));
  }
}
