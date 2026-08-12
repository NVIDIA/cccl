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
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/span>

#include <cuda/experimental/__multi_gpu/algorithm/sort/sort.h>

#include <algorithm>
#include <exception>
#include <future>
#include <string>
#include <vector>

#include <algorithm_common.h>
#include <nccl_test_common.h>
#include <testing.cuh>

#include "sort_common.cuh"
#include <c2h/catch2_test_helper.h>

namespace
{
using sort_test_util::abs_less;
using sort_test_util::make_value;
using sort_test_util::sort_types;

// Drive the sort through the single-communicator overload, one thread per local rank. That
// overload opens its own NCCL group on a single communicator, so issuing the per-rank calls
// serially on one thread would deadlock at `ncclGroupEnd`. Only the `sort` call happens on the
// worker threads; every Catch2 assertion runs on the main thread after the join, since the
// assertion macros are not safe to fire concurrently.
template <class T, class Compare>
void check_sort_case(
  cuda::std::span<cudax::nccl_communicator_ref> comms, const std::vector<std::vector<T>>& host_inputs, Compare cmp)
{
  REQUIRE(host_inputs.size() == comms.size());

  const auto expected = sort_test_util::sorted_reference(host_inputs, cmp);
  auto streams        = nccl_test_util::make_streams();
  auto environments   = std::vector<cuda::stream_ref>{streams.begin(), streams.end()};
  auto device_vec     = sort_test_util::make_device_inputs(comms, environments, host_inputs);

  run_threaded(comms.size(), [&](cuda::std::size_t i) {
    cudax::sort(cudax::distributed, comms[i], environments[i], device_vec[i].begin(), device_vec[i].size(), cmp);
  });

  sort_test_util::check_rank_sizes(comms, device_vec, host_inputs);

  const auto output = sort_test_util::gather_outputs(comms, device_vec);

  REQUIRE(std::is_sorted(output.begin(), output.end(), cmp));
  sort_test_util::check_matches(streams.front(), output, expected);
}

template <class T>
void check_sort_case_sections(cuda::std::span<cudax::nccl_communicator_ref> comms,
                              const std::vector<std::vector<T>>& host_inputs)
{
  SECTION("ascending comparator")
  {
    check_sort_case(comms, host_inputs, cuda::std::less<>{});
  }

  SECTION("descending comparator")
  {
    check_sort_case(comms, host_inputs, cuda::std::greater<>{});
  }
}
} // namespace

MULTI_GPU_TEST("sort single-comm documentation example", c2h::type_list<int>)
{
  auto comms = this->communicators();

  if (comms.size() < 2)
  {
    SKIP("The sort documentation example requires at least two local GPUs");
  }

  auto streams_owned = nccl_test_util::make_streams();
  auto streams       = std::vector<cuda::stream_ref>{streams_owned.begin(), streams_owned.end()};

  // Must be pre-allocated since it is written to by threads
  std::vector<std::string> failed(comms.front().size());

  // Every communicator rank must invoke the collective concurrently.
  run_threaded(comms.size(), [&](cuda::std::size_t i) {
    auto& communicator = comms[i];
    auto environment   = streams[i];

    REQUIRE_CUDART(cudaSetDevice(communicator.logical_device().underlying_device().get()));

    //! [sort_single_range]
    // Rank r contributes the descending pair {2 * (size - r), 2 * (size - r) - 1}, so the ranks
    // together hold the values 1 through 2 * size in reverse rank order. The input range keeps
    // its original size: the sort re-partitions the keys across the ranks and restores the
    // per-rank sizes before it writes the results back.
    const auto rank   = communicator.rank();
    const auto high   = 2 * (communicator.size() - rank);
    const auto device = communicator.logical_device().underlying_device();

    auto input = cuda::make_device_buffer<int>(environment, device, {high, high - 1});

    cudax::sort(cudax::distributed, communicator, environment, input.begin(), input.size());

    // The sort is in place and each rank keeps its original element count, so rank r ends up with
    // its two-element slice of the globally sorted sequence.
    const auto expected =
      cuda::make_buffer<int>(environment, cuda::mr::legacy_pinned_memory_resource{}, {2 * rank + 1, 2 * rank + 2});
    //! [sort_single_range]

    environment.sync();

    // catch2 isn't thread safe by default, so we can't use the usual requires expression. So
    // we roll a hacky version of it ourselves
    if (const auto matcher = Equals(expected); !matcher.match(input))
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

MULTI_GPU_TEST("sort single-comm, random inputs", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = this->communicators();
  auto rng   = sort_test_util::make_rng(C2H_SEED(2));

  std::vector<std::vector<T>> input(comms.size());
  for (auto& local : input)
  {
    sort_test_util::fill_random(local, 100, rng);
  }

  check_sort_case_sections(comms, input);
}

MULTI_GPU_TEST("sort single-comm, uneven rank sizes", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = this->communicators();
  auto rng   = sort_test_util::make_rng(C2H_SEED(2));

  std::vector<std::vector<T>> input(comms.size());
  for (cuda::std::size_t rank = 0; rank < input.size(); ++rank)
  {
    sort_test_util::fill_random(input[rank], (rank * 10) + 1, rng);
  }

  check_sort_case_sections(comms, input);
}

MULTI_GPU_TEST("sort single-comm, inputs with some empty ranks", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = this->communicators();
  auto rng   = sort_test_util::make_rng(C2H_SEED(2));

  std::vector<std::vector<T>> input(comms.size());
  for (cuda::std::size_t rank = 1; rank < input.size(); rank += 2)
  {
    sort_test_util::fill_random(input[rank], 1'000, rng);
  }

  check_sort_case_sections(comms, input);
}

MULTI_GPU_TEST("sort single-comm, all ranks empty", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = this->communicators();
  std::vector<std::vector<T>> input(comms.size());

  check_sort_case_sections(comms, input);
}

MULTI_GPU_TEST("sort single-comm, a single global item", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = this->communicators();
  std::vector<std::vector<T>> input(comms.size());

  if (!input.empty())
  {
    input[0].push_back(make_value<T>(1, 1));
  }

  check_sort_case_sections(comms, input);
}

MULTI_GPU_TEST("sort single-comm, one item per rank", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = this->communicators();
  std::vector<std::vector<T>> input(comms.size());

  for (cuda::std::size_t rank = 0; rank < input.size(); ++rank)
  {
    const auto key = static_cast<cuda::std::int64_t>(input.size() - rank);
    input[rank].push_back(make_value<T>(key, key));
  }

  check_sort_case_sections(comms, input);
}

MULTI_GPU_TEST("sort single-comm, all equal inputs", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = this->communicators();
  std::vector<std::vector<T>> input(comms.size());

  for (auto& local : input)
  {
    local.assign(100, make_value<T>(1, 1));
  }

  check_sort_case_sections(comms, input);
}

MULTI_GPU_TEST("sort single-comm, inputs with many equal keys", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = this->communicators();
  std::vector<std::vector<T>> input(comms.size());

  for (cuda::std::size_t rank = 0; rank < input.size(); ++rank)
  {
    auto& local = input[rank];
    local.resize(1'000);

    for (cuda::std::size_t item = 0; item < local.size(); ++item)
    {
      const auto key = static_cast<cuda::std::int64_t>(item % 2);
      local[item]    = make_value<T>(key, static_cast<cuda::std::int64_t>(rank * local.size() + item));
    }
  }

  check_sort_case_sections(comms, input);
}

MULTI_GPU_TEST("sort single-comm, presorted inputs", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = this->communicators();
  std::vector<std::vector<T>> input(comms.size());

  for (cuda::std::size_t rank = 0; rank < input.size(); ++rank)
  {
    auto& local = input[rank];
    local.resize(1'000);

    for (cuda::std::size_t item = 0; item < local.size(); ++item)
    {
      const auto key = static_cast<cuda::std::int64_t>(rank * local.size() + item);
      local[item]    = make_value<T>(key, key);
    }
  }

  check_sort_case_sections(comms, input);
}

MULTI_GPU_TEST("sort single-comm, reverse-sorted inputs", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = this->communicators();
  std::vector<std::vector<T>> input(comms.size());

  for (cuda::std::size_t rank = 0; rank < input.size(); ++rank)
  {
    auto& local = input[rank];
    local.resize(1'000);

    for (cuda::std::size_t item = 0; item < local.size(); ++item)
    {
      const auto key = static_cast<cuda::std::int64_t>(input.size() * local.size() - (rank * local.size() + item));
      local[item]    = make_value<T>(key, key);
    }
  }

  check_sort_case_sections(comms, input);
}

MULTI_GPU_TEST("sort single-comm, skewed rank sizes", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = this->communicators();
  auto rng   = sort_test_util::make_rng(C2H_SEED(2));

  std::vector<std::vector<T>> input(comms.size());
  for (cuda::std::size_t rank = 0; rank < input.size(); ++rank)
  {
    sort_test_util::fill_random(input[rank], rank == 0 ? 1'000 : 1, rng);
  }

  check_sort_case_sections(comms, input);
}

MULTI_GPU_TEST("sort single-comm, nonstandard comparator", )
{
  auto comms = this->communicators();
  std::vector<std::vector<int>> input(comms.size());

  for (cuda::std::size_t rank = 0; rank < input.size(); ++rank)
  {
    auto& local = input[rank];
    local.resize(1'000);

    for (cuda::std::size_t item = 0; item < local.size(); ++item)
    {
      const auto magnitude = static_cast<int>((rank + item) % 5);
      local[item]          = item % 2 == 0 ? magnitude : -magnitude;
    }
  }

  check_sort_case(comms, input, abs_less<int>{});
}
