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

// Drive the scan through the single-communicator overload, one thread per local rank. The
// per-rank calls must rendezvous in their collectives, so issuing them serially would deadlock.
// Catch2 assertions remain on the main thread after all worker threads have joined.
template <class T, class Op>
void run_case(cuda::std::span<cudax::nccl_communicator_ref> comms,
              const std::vector<std::vector<T>>& inputs_by_rank,
              const T& init,
              const T& ident,
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
    in.emplace_back(cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), values));
    out.emplace_back(cuda::make_device_buffer<T>(
      streams[i], comms[i].logical_device().underlying_device(), values.size(), cuda::no_init));
    envs.emplace_back(::cuda::std::execution::env{::cuda::stream_ref{streams[i]}});
  }

  const auto in_copy      = in;
  auto outputs            = make_output_iterators(out);
  const auto outputs_copy = outputs;

  INFO("init = " << init);
  INFO("ident = " << ident);

  run_threaded(comms.size(), [&](cuda::std::size_t i) {
    cudax::exclusive_scan(comms[i], envs[i], in[i], outputs[i], init, op, ident);
  });

  REQUIRE(in.size() == in_copy.size());
  for (cuda::std::size_t i = 0; i < in.size(); ++i)
  {
    INFO("device = " << i);
    REQUIRE_THAT(in[i], Equals(in_copy[i]));
    const auto expected_values = expected_for_rank<T>(comms[i].rank(), inputs_by_rank, init, op);
    const auto expected =
      cuda::make_buffer<T>(out[i].stream(), cuda::mr::legacy_pinned_memory_resource{}, expected_values);

    REQUIRE_THAT(out[i], Equals(expected));
  }
  REQUIRE_THAT(outputs, Catch::Matchers::Equals(outputs_copy));
}
} // namespace

MULTI_GPU_TEST("exclusive_scan single-comm documentation example", c2h::type_list<int>)
{
  auto comms = this->communicators();

  if (comms.size() < 2)
  {
    SKIP("The exclusive_scan documentation example requires at least two local GPUs");
  }

  auto streams_owned = nccl_test_util::make_streams();
  auto streams       = std::vector<cuda::stream_ref>{streams_owned.begin(), streams_owned.end()};

  // Must be pre-allocated since it is written to by threads
  std::vector<std::string> failed(comms.front().size());

  // Every communicator rank must invoke the collective concurrently.
  run_threaded(comms.size(), [&](cuda::std::size_t i) {
    auto& communicator = comms[i];
    auto environment   = streams[i];
    const auto device  = communicator.logical_device().underlying_device();

    //! [exclusive_scan_single_range]
    constexpr cuda::std::array input_values{1, 2};

    auto input  = cuda::make_device_buffer<int>(environment, device, input_values);
    auto output = cuda::make_device_buffer<int>(environment, device, input_values.size(), cuda::no_init);

    cudax::exclusive_scan(communicator, environment, input, output.begin(), /*__init=*/0);

    // Every rank contributes {1, 2}, so rank r starts with a prefix of 3 * r.
    const auto rank = communicator.rank();
    const auto expected =
      cuda::make_buffer<int>(output.stream(), cuda::mr::legacy_pinned_memory_resource{}, {3 * rank, 3 * rank + 1});

    //! [exclusive_scan_single_range]

    // catch2 isn't thread safe by default, so we can't use the usual requires expression. So
    // we roll a hacky version of it ourselves
    if (const auto matcher = Equals(expected); !matcher.match(output))
    {
      failed[rank] = matcher.describe();
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

MULTI_GPU_TEST("exclusive_scan single-comm, one element per rank", value_types, operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  const T init     = make_value<T>(GENERATE(0, 1, -1, 5));
  const auto ident = get_identity<T, Op>();
  auto comms       = this->communicators();

  std::vector<std::vector<T>> inputs_by_rank;

  inputs_by_rank.reserve(static_cast<cuda::std::size_t>(comms.front().size()));
  for (int r = 0; r < comms.front().size(); ++r)
  {
    const auto v = {make_value<T>(r)};

    inputs_by_rank.emplace_back(v);
  }

  run_case(comms, inputs_by_rank, init, ident, Op{});
}

MULTI_GPU_TEST("exclusive_scan single-comm, multiple elements per rank", value_types, operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  const T init     = make_value<T>(GENERATE(0, 1, -1, 5));
  const auto ident = get_identity<T, Op>();
  auto comms       = this->communicators();

  std::vector<std::vector<T>> inputs_by_rank(static_cast<cuda::std::size_t>(comms.front().size()));
  for (int r = 0; r < comms.front().size(); ++r)
  {
    const auto value                                  = make_value<T>(r);
    inputs_by_rank[static_cast<cuda::std::size_t>(r)] = {value, value, value};
  }

  run_case(comms, inputs_by_rank, init, ident, Op{});
}

MULTI_GPU_TEST("exclusive_scan single-comm, some ranks empty", value_types, operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  const T init     = make_value<T>(GENERATE(0, 1, -1, 5));
  const auto ident = get_identity<T, Op>();
  auto comms       = this->communicators();

  std::vector<std::vector<T>> inputs_by_rank(static_cast<cuda::std::size_t>(comms.front().size()));
  for (int r = 0; r < comms.front().size(); ++r)
  {
    if (r % 2 == 0)
    {
      inputs_by_rank[static_cast<cuda::std::size_t>(r)] = {make_value<T>(r), make_value<T>(r)};
    }
  }

  run_case(comms, inputs_by_rank, init, ident, Op{});
}

MULTI_GPU_TEST("exclusive_scan single-comm, all ranks empty", value_types, operators)
{
  using T  = c2h::get<0, TestType>;
  using Op = c2h::get<1, TestType>;

  const T init     = make_value<T>(GENERATE(0, 1, -1, 5));
  const auto ident = get_identity<T, Op>();
  auto comms       = this->communicators();

  const std::vector<std::vector<T>> inputs_by_rank(static_cast<cuda::std::size_t>(comms.front().size()));

  run_case(comms, inputs_by_rank, init, ident, Op{});
}
