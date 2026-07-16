//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__algorithm/sort.h>
#include <cuda/std/algorithm>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/numeric>
#include <cuda/std/random>
#include <cuda/std/span>

#include <cuda/experimental/__multi_gpu/algorithm/sort/sort.h>

#include <vector>

#include <nccl_test_common.h>
#include <testing.cuh>

#include <c2h/catch2_test_helper.h>
#include <c2h/vector.h>

namespace
{
using custom_key_t =
  c2h::custom_type_t<c2h::equal_comparable_t,
                     c2h::lexicographical_less_comparable_t,
                     c2h::lexicographical_greater_comparable_t>;
using sort_types = c2h::type_list<int, cuda::std::int64_t, double, custom_key_t>;

template <class T>
[[nodiscard]] T make_value(const cuda::std::int64_t key, const cuda::std::int64_t)
{
  return static_cast<T>(key);
}

template <>
[[nodiscard]] custom_key_t make_value<custom_key_t>(const cuda::std::int64_t key, const cuda::std::int64_t value)
{
  custom_key_t result{};

  result.key = static_cast<cuda::std::size_t>(key);
  result.val = static_cast<cuda::std::size_t>(value);
  return result;
}

[[nodiscard]] cuda::std::minstd_rand make_rng(const c2h::seed_t& seed)
{
  return cuda::std::minstd_rand(static_cast<cuda::std::minstd_rand::result_type>(seed.get()));
}

template <class T, class RNG>
void fill_random(c2h::host_vector<T>& local, cuda::std::size_t count, RNG& rng)
{
  cuda::std::uniform_int_distribution<cuda::std::int64_t> dist{0, 1000};

  local.resize(count);
  for (cuda::std::size_t item = 0; item < local.size(); ++item)
  {
    const auto key = dist(rng);

    local[item] = make_value<T>(key, key + static_cast<cuda::std::int64_t>(item));
  }
}

template <class T>
[[nodiscard]] std::vector<c2h::device_vector<T>>
make_device_inputs(cuda::std::span<cudax::nccl_communicator_ref> comms, const std::vector<c2h::host_vector<T>>& inputs)
{
  std::vector<c2h::device_vector<T>> ret;

  ret.reserve(comms.size());
  for (cuda::std::size_t rank = 0; rank < comms.size(); ++rank)
  {
    REQUIRE_CUDART(cudaSetDevice(comms[rank].logical_device().underlying_device().get()));
    ret.emplace_back(inputs[rank]);
  }
  return ret;
}

template <class T>
[[nodiscard]] c2h::host_vector<T>
gather_outputs(cuda::std::span<cudax::nccl_communicator_ref> comms, const std::vector<c2h::device_vector<T>>& inputs)
{
  c2h::host_vector<T> ret;

  ret.reserve(cuda::std::accumulate(
    inputs.begin(), inputs.end(), cuda::std::size_t{}, [](cuda::std::size_t ret, const auto& vec) {
      return ret + vec.size();
    }));
  for (cuda::std::size_t rank = 0; rank < comms.size(); ++rank)
  {
    REQUIRE_CUDART(cudaSetDevice(comms[rank].logical_device().underlying_device().get()));

    const auto& local = inputs[rank];
    ret.insert(ret.end(), local.begin(), local.end());
  }
  return ret;
}

template <class T, class Compare>
[[nodiscard]] c2h::host_vector<T> sorted_reference(const std::vector<c2h::host_vector<T>>& inputs, Compare cmp)
{
  c2h::host_vector<T> ret;

  ret.reserve(cuda::std::accumulate(
    inputs.begin(), inputs.end(), cuda::std::size_t{}, [](cuda::std::size_t ret, const auto& vec) {
      return ret + vec.size();
    }));
  for (const auto& local : inputs)
  {
    ret.insert(ret.end(), local.begin(), local.end());
  }

  cuda::std::sort(ret.begin(), ret.end(), cmp);
  return ret;
}

template <class T, class Compare>
void check_sort_case(
  cuda::std::span<cudax::nccl_communicator_ref> comms, const std::vector<c2h::host_vector<T>>& host_inputs, Compare cmp)
{
  REQUIRE(host_inputs.size() == comms.size());

  const auto expected = sorted_reference(host_inputs, cmp);
  auto streams        = nccl_test_util::make_streams();
  auto environments   = std::vector<cuda::stream_ref>{streams.begin(), streams.end()};
  auto device_vec     = make_device_inputs(comms, host_inputs);

  cudax::sort(cudax::distributed, comms, environments, device_vec, cmp);

  // Since we are using c2h vectors instead of cuda buffers (which remember what stream they
  // are on), we need to sync here before doing the checks because the internal copy stream
  // wont know to wait on the data.
  for (auto& stream : streams)
  {
    stream.sync();
  }

  REQUIRE(device_vec.size() == host_inputs.size());
  for (cuda::std::size_t rank = 0; rank < comms.size(); ++rank)
  {
    CAPTURE(rank);
    REQUIRE(device_vec[rank].size() == host_inputs[rank].size());
  }

  const auto output = gather_outputs(comms, device_vec);

  REQUIRE(cuda::std::is_sorted(output.begin(), output.end(), cmp));
  REQUIRE_THAT(output, Equals(expected));
}

template <class T>
void check_sort_case_sections(cuda::std::span<cudax::nccl_communicator_ref> comms,
                              const std::vector<c2h::host_vector<T>>& host_inputs)
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

template <class T>
struct abs_less
{
  [[nodiscard]] static _CCCL_API constexpr T abs(const T& value)
  {
    return value < T{} ? -value : value;
  }

  [[nodiscard]] _CCCL_API constexpr bool operator()(const T& lhs, const T& rhs) const
  {
    return abs(lhs) == abs(rhs) ? lhs < rhs : abs(lhs) < abs(rhs);
  }
};
} // namespace

MULTI_GPU_TEST("random inputs", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = this->communicators();
  auto rng   = make_rng(C2H_SEED(2));

  std::vector<c2h::host_vector<T>> input(comms.size());
  for (auto& local : input)
  {
    fill_random(local, 100, rng);
  }

  check_sort_case_sections(comms, input);
}

MULTI_GPU_TEST("uneven rank sizes", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = this->communicators();
  auto rng   = make_rng(C2H_SEED(2));

  std::vector<c2h::host_vector<T>> input(comms.size());
  for (cuda::std::size_t rank = 0; rank < input.size(); ++rank)
  {
    fill_random(input[rank], (rank * 10) + 1, rng);
  }

  check_sort_case_sections(comms, input);
}

MULTI_GPU_TEST("inputs with some empty ranks", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = this->communicators();
  auto rng   = make_rng(C2H_SEED(2));

  std::vector<c2h::host_vector<T>> input(comms.size());
  for (cuda::std::size_t rank = 1; rank < input.size(); rank += 2)
  {
    fill_random(input[rank], 100, rng);
  }

  check_sort_case_sections(comms, input);
}

MULTI_GPU_TEST("no communicators", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  const auto comms = cuda::std::span<cudax::nccl_communicator_ref>{};
  std::vector<c2h::host_vector<T>> input(comms.size());

  check_sort_case_sections(comms, input);
}

MULTI_GPU_TEST("all ranks empty", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = this->communicators();
  std::vector<c2h::host_vector<T>> input(comms.size());

  check_sort_case_sections(comms, input);
}

MULTI_GPU_TEST("a single global item", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = this->communicators();
  std::vector<c2h::host_vector<T>> input(comms.size());

  if (!input.empty())
  {
    input[0].push_back(make_value<T>(1, 1));
  }

  check_sort_case_sections(comms, input);
}

MULTI_GPU_TEST("one item per rank", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = this->communicators();
  std::vector<c2h::host_vector<T>> input(comms.size());

  for (cuda::std::size_t rank = 0; rank < input.size(); ++rank)
  {
    const auto key = static_cast<cuda::std::int64_t>(input.size() - rank);
    input[rank].push_back(make_value<T>(key, key));
  }

  check_sort_case_sections(comms, input);
}

MULTI_GPU_TEST("all equal inputs", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = this->communicators();
  std::vector<c2h::host_vector<T>> input(comms.size());

  for (auto& local : input)
  {
    local.assign(100, make_value<T>(1, 1));
  }

  check_sort_case_sections(comms, input);
}

MULTI_GPU_TEST("inputs with many equal keys", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = this->communicators();
  std::vector<c2h::host_vector<T>> input(comms.size());

  for (cuda::std::size_t rank = 0; rank < input.size(); ++rank)
  {
    auto& local = input[rank];
    local.resize(100);

    for (cuda::std::size_t item = 0; item < local.size(); ++item)
    {
      const auto key = static_cast<cuda::std::int64_t>(item % 2);
      local[item]    = make_value<T>(key, static_cast<cuda::std::int64_t>(rank * local.size() + item));
    }
  }

  check_sort_case_sections(comms, input);
}

MULTI_GPU_TEST("presorted inputs", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = this->communicators();
  std::vector<c2h::host_vector<T>> input(comms.size());

  for (cuda::std::size_t rank = 0; rank < input.size(); ++rank)
  {
    auto& local = input[rank];
    local.resize(100);

    for (cuda::std::size_t item = 0; item < local.size(); ++item)
    {
      const auto key = static_cast<cuda::std::int64_t>(rank * local.size() + item);
      local[item]    = make_value<T>(key, key);
    }
  }

  check_sort_case_sections(comms, input);
}

MULTI_GPU_TEST("reverse-sorted inputs", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = this->communicators();
  std::vector<c2h::host_vector<T>> input(comms.size());

  for (cuda::std::size_t rank = 0; rank < input.size(); ++rank)
  {
    auto& local = input[rank];
    local.resize(100);

    for (cuda::std::size_t item = 0; item < local.size(); ++item)
    {
      const auto key = static_cast<cuda::std::int64_t>(input.size() * local.size() - (rank * local.size() + item));
      local[item]    = make_value<T>(key, key);
    }
  }

  check_sort_case_sections(comms, input);
}

MULTI_GPU_TEST("skewed rank sizes", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = this->communicators();
  auto rng   = make_rng(C2H_SEED(2));

  std::vector<c2h::host_vector<T>> input(comms.size());
  for (cuda::std::size_t rank = 0; rank < input.size(); ++rank)
  {
    fill_random(input[rank], rank == 0 ? 200 : 1, rng);
  }

  check_sort_case_sections(comms, input);
}

MULTI_GPU_TEST("nonstandard comparator", )
{
  auto comms = this->communicators();
  std::vector<c2h::host_vector<int>> input(comms.size());

  for (cuda::std::size_t rank = 0; rank < input.size(); ++rank)
  {
    auto& local = input[rank];
    local.resize(100);

    for (cuda::std::size_t item = 0; item < local.size(); ++item)
    {
      const auto magnitude = static_cast<int>((rank + item) % 5);
      local[item]          = item % 2 == 0 ? magnitude : -magnitude;
    }
  }

  check_sort_case(comms, input, abs_less<int>{});
}
