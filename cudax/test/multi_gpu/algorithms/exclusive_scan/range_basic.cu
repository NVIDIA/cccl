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

// Run the full scan, wait for it to finish, and check that `exclusive_scan` left its argument
// ranges untouched. This boilerplate is identical for every test regardless of how the inputs are
// shaped.
template <class Env, class T, class Op>
void do_exclusive_scan(
  cuda::std::span<cudax::nccl_communicator_ref> comms,
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

  cudax::exclusive_scan(comms, envs, in, outputs, init, op, ident);

  // cuda::std::execution::env has no operator==, so we can only compare the sizes.
  REQUIRE(envs.size() == envs_size);
  // Scan call should not modify the inputs in any ways
  REQUIRE(in.size() == in_copy.size());
  for (cuda::std::size_t i = 0; i < in.size(); ++i)
  {
    INFO("device = " << i);
    REQUIRE_THAT(in[i], Equals(in_copy[i]));
  }
  REQUIRE_THAT(outputs, Catch::Matchers::Equals(outputs_copy));
}
} // namespace

MULTI_GPU_TEST("exclusive_scan documentation example", c2h::type_list<int>)
{
  auto comms = this->communicators();

  if (comms.size() < 2)
  {
    SKIP("The exclusive_scan documentation example requires at least two local GPUs");
  }

  auto streams_owned = nccl_test_util::make_streams();
  // Convert to stream_ref directly, cuda::stream on their own cant be passed directly to CUB
  auto streams = std::vector<cuda::stream_ref>{streams_owned.begin(), streams_owned.end()};

  //! [exclusive_scan]
  constexpr cuda::std::array input_values{1, 2};
  std::vector<cuda::device_buffer<int>> inputs;
  std::vector<cuda::device_buffer<int>> outputs;

  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto device = comms[i].logical_device().underlying_device();

    inputs.emplace_back(cuda::make_device_buffer<int>(streams[i], device, input_values));
    outputs.emplace_back(cuda::make_device_buffer<int>(streams[i], device, input_values.size(), cuda::no_init));
  }

  std::vector<typename cuda::device_buffer<int>::iterator> output_iterators = make_output_iterators(outputs);

  cudax::exclusive_scan(
    comms,
    // Passing streams as the environment directly
    streams,
    inputs,
    output_iterators,
    /*__init=*/0);

  constexpr cuda::std::array expected_rank_0{0, 1};
  constexpr cuda::std::array expected_rank_1{3, 4};
  const auto expected_0 =
    cuda::make_buffer<int>(outputs[0].stream(), cuda::mr::legacy_pinned_memory_resource{}, expected_rank_0);
  const auto expected_1 =
    cuda::make_buffer<int>(outputs[1].stream(), cuda::mr::legacy_pinned_memory_resource{}, expected_rank_1);
  REQUIRE_THAT(outputs[0], Equals(expected_0));
  REQUIRE_THAT(outputs[1], Equals(expected_1));
  //! [exclusive_scan]
}

MULTI_GPU_TEST("exclusive_scan, one element per rank", value_types, operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  // Seed each scan with a few hardcoded initializers. The init participates in the fold the same
  // way on host and device, so any value works for every operator under test.
  const T init     = make_value<T>(GENERATE(0, 1, -1, 5));
  const auto ident = get_identity<T, Op>();

  auto comms   = this->communicators();
  auto streams = nccl_test_util::make_streams();

  // Global rank `comms[i].rank()` contributes the single value `rank`. Each local rank also gets a
  // one-element output buffer and an environment carrying its stream, so the scan is stream-ordered
  // on the correct device. `reference` mirrors the contributions of every global rank so we can
  // compute the host-side scan exactly like `exclusive_scan` does on the device.
  std::vector<cuda::device_buffer<T>> in;
  std::vector<cuda::device_buffer<T>> out;
  std::vector<decltype(::cuda::std::execution::env{::cuda::stream_ref{streams[0]}})> envs;
  std::vector<std::vector<T>> inputs_by_rank(static_cast<cuda::std::size_t>(comms.front().size()));

  in.reserve(comms.size());
  out.reserve(comms.size());
  envs.reserve(comms.size());
  for (int r = 0; r < comms.front().size(); ++r)
  {
    inputs_by_rank[static_cast<cuda::std::size_t>(r)] = {make_value<T>(r)};
  }
  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto& values = inputs_by_rank[static_cast<cuda::std::size_t>(comms[i].rank())];
    in.emplace_back(cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), values));
    out.emplace_back(cuda::make_device_buffer<T>(
      streams[i], comms[i].logical_device().underlying_device(), values.size(), cuda::no_init));
    envs.emplace_back(::cuda::std::execution::env{::cuda::stream_ref{streams[i]}});
  }

  auto outputs = make_output_iterators(out);

  do_exclusive_scan(comms, envs, in, outputs, init, ident, Op{});

  for (cuda::std::size_t i = 0; i < out.size(); ++i)
  {
    const auto expected_values = expected_for_rank<T>(comms[i].rank(), inputs_by_rank, init, Op{});
    const auto exp = cuda::make_buffer<T>(out[i].stream(), cuda::mr::legacy_pinned_memory_resource{}, expected_values);

    REQUIRE_THAT(out[i], Equals(exp));
  }
}

MULTI_GPU_TEST("exclusive_scan, multiple elements per rank", value_types, operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  // Seed each scan with a few hardcoded initializers. The init participates in the fold the same
  // way on host and device, so any value works for every operator under test.
  const T init     = make_value<T>(GENERATE(0, 1, -1, 5));
  const auto ident = get_identity<T, Op>();

  auto comms   = this->communicators();
  auto streams = nccl_test_util::make_streams();

  // Global rank `comms[i].rank()` contributes `{rank, rank, rank}`. `exclusive_scan` first
  // computes a local prefix for each rank seeded by the prefix of all previous ranks. Each local
  // rank also gets an output buffer and an environment carrying its stream. `reference` mirrors
  // every global rank's three contributions for the host-side scan.
  std::vector<cuda::device_buffer<T>> in;
  std::vector<cuda::device_buffer<T>> out;
  std::vector<decltype(::cuda::std::execution::env{::cuda::stream_ref{streams[0]}})> envs;
  std::vector<std::vector<T>> inputs_by_rank(static_cast<cuda::std::size_t>(comms.front().size()));

  in.reserve(comms.size());
  out.reserve(comms.size());
  envs.reserve(comms.size());
  for (int r = 0; r < comms.front().size(); ++r)
  {
    const auto v                                      = make_value<T>(r);
    inputs_by_rank[static_cast<cuda::std::size_t>(r)] = {v, v, v};
  }
  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto& values = inputs_by_rank[static_cast<cuda::std::size_t>(comms[i].rank())];
    in.emplace_back(cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), values));
    out.emplace_back(cuda::make_device_buffer<T>(
      streams[i], comms[i].logical_device().underlying_device(), values.size(), cuda::no_init));
    envs.emplace_back(::cuda::std::execution::env{::cuda::stream_ref{streams[i]}});
  }

  auto outputs = make_output_iterators(out);

  do_exclusive_scan(comms, envs, in, outputs, init, ident, Op{});

  for (cuda::std::size_t i = 0; i < out.size(); ++i)
  {
    const auto expected_values = expected_for_rank<T>(comms[i].rank(), inputs_by_rank, init, Op{});
    const auto exp = cuda::make_buffer<T>(out[i].stream(), cuda::mr::legacy_pinned_memory_resource{}, expected_values);

    REQUIRE_THAT(out[i], Equals(exp));
  }
}

MULTI_GPU_TEST("exclusive_scan, some ranks empty", value_types, operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  const T init     = make_value<T>(GENERATE(0, 1, -1, 5));
  const auto ident = get_identity<T, Op>();

  auto comms   = this->communicators();
  auto streams = nccl_test_util::make_streams();

  // Even global ranks contribute two copies of `rank`; odd global ranks contribute an empty input
  // range. Rank 0 is always non-empty. `exclusive_scan` must treat an empty rank as contributing
  // nothing, exactly like `std::exclusive_scan` over the surviving elements. `reference` mirrors
  // that for the host-side scan.
  std::vector<cuda::device_buffer<T>> in;
  std::vector<cuda::device_buffer<T>> out;
  std::vector<decltype(::cuda::std::execution::env{::cuda::stream_ref{streams[0]}})> envs;
  std::vector<std::vector<T>> inputs_by_rank(static_cast<cuda::std::size_t>(comms.front().size()));

  in.reserve(comms.size());
  out.reserve(comms.size());
  envs.reserve(comms.size());
  for (int r = 0; r < comms.front().size(); ++r)
  {
    if (r % 2 == 0)
    {
      inputs_by_rank[static_cast<cuda::std::size_t>(r)] = {make_value<T>(r), make_value<T>(r)};
    }
  }
  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto& values = inputs_by_rank[static_cast<cuda::std::size_t>(comms[i].rank())];
    in.emplace_back(cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), values));
    out.emplace_back(cuda::make_device_buffer<T>(
      streams[i], comms[i].logical_device().underlying_device(), values.size(), cuda::no_init));
    envs.emplace_back(::cuda::std::execution::env{::cuda::stream_ref{streams[i]}});
  }

  auto outputs = make_output_iterators(out);

  do_exclusive_scan(comms, envs, in, outputs, init, ident, Op{});

  for (cuda::std::size_t i = 0; i < out.size(); ++i)
  {
    const auto expected_values = expected_for_rank<T>(comms[i].rank(), inputs_by_rank, init, Op{});
    const auto exp = cuda::make_buffer<T>(out[i].stream(), cuda::mr::legacy_pinned_memory_resource{}, expected_values);

    REQUIRE_THAT(out[i], Equals(exp));
  }
}

MULTI_GPU_TEST("exclusive_scan, all ranks empty", value_types, operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  const auto init  = make_value<T>(GENERATE(0, 1, -1, 5));
  const auto ident = get_identity<T, Op>();

  auto comms   = this->communicators();
  auto streams = nccl_test_util::make_streams();

  // No rank contributes any element. Scanning nothing produces no output values, exactly like
  // `std::exclusive_scan` over an empty range.
  std::vector<cuda::device_buffer<T>> in;
  std::vector<cuda::device_buffer<T>> out;
  std::vector<decltype(::cuda::std::execution::env{::cuda::stream_ref{streams[0]}})> envs;
  std::vector<std::vector<T>> inputs_by_rank(static_cast<cuda::std::size_t>(comms.front().size()));

  in.reserve(comms.size());
  out.reserve(comms.size());
  envs.reserve(comms.size());
  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto& values = inputs_by_rank[static_cast<cuda::std::size_t>(comms[i].rank())];
    in.emplace_back(cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), values));
    out.emplace_back(cuda::make_device_buffer<T>(
      streams[i], comms[i].logical_device().underlying_device(), values.size(), cuda::no_init));
    envs.emplace_back(::cuda::std::execution::env{::cuda::stream_ref{streams[i]}});
  }

  auto outputs = make_output_iterators(out);

  do_exclusive_scan(comms, envs, in, outputs, init, ident, Op{});

  for (cuda::std::size_t i = 0; i < out.size(); ++i)
  {
    const auto expected_values = expected_for_rank<T>(comms[i].rank(), inputs_by_rank, init, Op{});
    const auto exp = cuda::make_buffer<T>(out[i].stream(), cuda::mr::legacy_pinned_memory_resource{}, expected_values);

    REQUIRE_THAT(out[i], Equals(exp));
  }
}
