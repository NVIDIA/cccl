//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/memory_pool>
#include <cuda/std/__algorithm/sort.h>
#include <cuda/std/algorithm>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/numeric>
#include <cuda/std/random>
#include <cuda/stream>

#include <cuda/experimental/__multi_gpu/sort.h>
#include <cuda/experimental/execution.cuh>

#include <stack>
#include <vector>

#include <nccl.h>
#include <testing.cuh>

#include <c2h/vector.h>

#define REQUIRE_NCCL(...) REQUIRE((__VA_ARGS__) == ::ncclSuccess)

namespace
{
using env_t = cudax::env_t<cuda::mr::device_accessible>;
using custom_key_t =
  c2h::custom_type_t<c2h::equal_comparable_t,
                     c2h::lexicographical_less_comparable_t,
                     c2h::lexicographical_greater_comparable_t>;
using sort_types = c2h::type_list<int, cuda::std::int64_t, double, custom_key_t>;

struct SavedDriverStack
{
  SavedDriverStack()
  {
    while (::cuda::__driver::__ctxGetCurrent() != nullptr)
    {
      contexts.push(cuda::__driver::__ctxPop());
    }
  }

  ~SavedDriverStack()
  {
    ::test::empty_driver_stack();
    while (!contexts.empty())
    {
      ::cuda::__driver::__ctxPush(contexts.top());
      contexts.pop();
    }
  }

  SavedDriverStack(const SavedDriverStack&) = delete;
  void operator=(const SavedDriverStack&)   = delete;

  std::stack<::CUcontext> contexts;
};

struct CommsManager
{
  CommsManager()
  {
    std::vector<int> devlist;

    for (auto&& device : cuda::devices)
    {
      device.init();
      devlist.push_back(device.get());
    }

    const auto _ = SavedDriverStack{};

    comms.resize(devlist.size());
    REQUIRE_NCCL(ncclCommInitAll(comms.data(), devlist.size(), devlist.data()));
    // ncclCommInitAll() will leave one or more contexts on the stack, so we need to restore
    // what we had before the call to it. See https://github.com/NVIDIA/nccl/issues/2231
  }

  ~CommsManager()
  {
    for (auto& comm : comms)
    {
      if (comm)
      {
        static_cast<void>(ncclCommDestroy(comm));
        comm = nullptr;
      }
    }
  }

  std::vector<ncclComm_t> comms;
};

[[nodiscard]] std::vector<cudax::communicator> get_comms()
{
  static const CommsManager mgr;

  std::vector<cudax::communicator> comms;

  comms.reserve(mgr.comms.size());
  for (std::size_t i = 0; i < mgr.comms.size(); ++i)
  {
    comms.emplace_back(mgr.comms[i], cudax::logical_device{cuda::devices[i]});
  }

  return comms;
}

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
make_device_inputs(const std::vector<cudax::communicator>& comms, const std::vector<c2h::host_vector<T>>& inputs)
{
  const auto _ = SavedDriverStack{};
  std::vector<c2h::device_vector<T>> ret;

  ret.reserve(comms.size());
  for (cuda::std::size_t rank = 0; rank < comms.size(); ++rank)
  {
    REQUIRE_CUDART(cudaSetDevice(comms[rank].device().underlying_device().get()));
    ret.emplace_back(inputs[rank]);
  }
  return ret;
}

[[nodiscard]] std::vector<cuda::stream> make_streams(const std::vector<cudax::communicator>& comms)
{
  std::vector<cuda::stream> streams;

  streams.reserve(comms.size());
  for (const auto& comm : comms)
  {
    streams.emplace_back(comm.device().underlying_device());
  }

  return streams;
}

[[nodiscard]] std::vector<env_t>
make_envs(const std::vector<cudax::communicator>& comms, const std::vector<cuda::stream>& streams)
{
  std::vector<env_t> envs;

  envs.reserve(comms.size());
  for (cuda::std::size_t rank = 0; rank < comms.size(); ++rank)
  {
    const auto device = comms[rank].device().underlying_device();

    envs.emplace_back(cuda::device_default_memory_pool(device), streams[rank]);
  }

  return envs;
}

template <class T>
[[nodiscard]] c2h::host_vector<T>
gather_outputs(const std::vector<cudax::communicator>& comms, const std::vector<c2h::device_vector<T>>& inputs)
{
  const auto _ = SavedDriverStack{};
  c2h::host_vector<T> ret;

  ret.reserve(cuda::std::accumulate(
    inputs.begin(), inputs.end(), cuda::std::size_t{}, [](cuda::std::size_t ret, const auto& vec) {
      return ret + vec.size();
    }));
  for (cuda::std::size_t rank = 0; rank < comms.size(); ++rank)
  {
    REQUIRE_CUDART(cudaSetDevice(comms[rank].device().underlying_device().get()));

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
  const std::vector<cudax::communicator>& comms, const std::vector<c2h::host_vector<T>>& host_inputs, Compare cmp)
{
  REQUIRE(host_inputs.size() == comms.size());

  const auto expected = sorted_reference(host_inputs, cmp);
  const auto streams  = make_streams(comms);
  const auto envs     = make_envs(comms, streams);
  auto device_vec     = make_device_inputs(comms, host_inputs);

  REQUIRE(test::count_driver_stack() == 0);
  cudax::sort(comms, envs, device_vec, cmp);
  // REQUIRE(test::count_driver_stack() == 0);
  for (auto& stream : streams)
  {
    stream.sync();
  }
  // REQUIRE(test::count_driver_stack() == 0);

  REQUIRE(device_vec.size() == host_inputs.size());
  for (cuda::std::size_t rank = 0; rank < comms.size(); ++rank)
  {
    CAPTURE(rank);
    REQUIRE(device_vec[rank].size() == host_inputs[rank].size());
  }
  // REQUIRE(test::count_driver_stack() == 0);

  const auto output = gather_outputs(comms, device_vec);
  // REQUIRE(test::count_driver_stack() == 0);

  REQUIRE(cuda::std::is_sorted(output.begin(), output.end(), cmp));
  REQUIRE_THAT(output, Equals(expected));
  // REQUIRE(test::count_driver_stack() == 0);
}

struct abs_less
{
  [[nodiscard]] static _CCCL_API constexpr int abs_int(const int value)
  {
    return value < 0 ? -value : value;
  }

  [[nodiscard]] _CCCL_API constexpr bool operator()(const int lhs, const int rhs) const
  {
    return abs_int(lhs) == abs_int(rhs) ? lhs < rhs : abs_int(lhs) < abs_int(rhs);
  }
};
} // namespace

C2H_CCCLRT_TEST("random inputs", "[multi_gpu][sort]", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = get_comms();
  auto rng   = make_rng(C2H_SEED(2));

  std::vector<c2h::host_vector<T>> input(comms.size());
  for (auto& local : input)
  {
    fill_random(local, 10, rng);
  }

  SECTION("ascending comparator")
  {
    check_sort_case(comms, input, cuda::std::less<>{});
  }

  SECTION("descending comparator")
  {
    check_sort_case(comms, input, cuda::std::greater<>{});
  }
}

C2H_CCCLRT_TEST("uneven rank sizes", "[multi_gpu][sort]", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = get_comms();
  auto rng   = make_rng(C2H_SEED(2));

  std::vector<c2h::host_vector<T>> input(comms.size());
  for (cuda::std::size_t rank = 0; rank < input.size(); ++rank)
  {
    fill_random(input[rank], rank + 1, rng);
  }

  SECTION("ascending comparator")
  {
    check_sort_case(comms, input, cuda::std::less<>{});
  }

  SECTION("descending comparator")
  {
    check_sort_case(comms, input, cuda::std::greater<>{});
  }
}

C2H_CCCLRT_TEST("inputs with some empty ranks", "[multi_gpu][sort]", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = get_comms();
  auto rng   = make_rng(C2H_SEED(2));

  std::vector<c2h::host_vector<T>> input(comms.size());
  for (cuda::std::size_t rank = 1; rank < input.size(); rank += 2)
  {
    fill_random(input[rank], 10, rng);
  }

  SECTION("ascending comparator")
  {
    check_sort_case(comms, input, cuda::std::less<>{});
  }

  SECTION("descending comparator")
  {
    check_sort_case(comms, input, cuda::std::greater<>{});
  }
}

C2H_CCCLRT_TEST("no communicators", "[multi_gpu][sort]", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  std::vector<cudax::communicator> comms;
  std::vector<c2h::host_vector<T>> input(comms.size());

  SECTION("ascending comparator")
  {
    check_sort_case(comms, input, cuda::std::less<>{});
  }

  SECTION("descending comparator")
  {
    check_sort_case(comms, input, cuda::std::greater<>{});
  }
}

C2H_CCCLRT_TEST("all ranks empty", "[multi_gpu][sort]", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = get_comms();
  std::vector<c2h::host_vector<T>> input(comms.size());

  SECTION("ascending comparator")
  {
    check_sort_case(comms, input, cuda::std::less<>{});
  }

  SECTION("descending comparator")
  {
    check_sort_case(comms, input, cuda::std::greater<>{});
  }
}

C2H_CCCLRT_TEST("a single global item", "[multi_gpu][sort]", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = get_comms();
  std::vector<c2h::host_vector<T>> input(comms.size());

  if (!input.empty())
  {
    input[0].push_back(make_value<T>(1, 1));
  }

  SECTION("ascending comparator")
  {
    check_sort_case(comms, input, cuda::std::less<>{});
  }

  SECTION("descending comparator")
  {
    check_sort_case(comms, input, cuda::std::greater<>{});
  }
}

C2H_CCCLRT_TEST("one item per rank", "[multi_gpu][sort]", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = get_comms();
  std::vector<c2h::host_vector<T>> input(comms.size());

  for (cuda::std::size_t rank = 0; rank < input.size(); ++rank)
  {
    const auto key = static_cast<cuda::std::int64_t>(input.size() - rank);
    input[rank].push_back(make_value<T>(key, key));
  }

  SECTION("ascending comparator")
  {
    check_sort_case(comms, input, cuda::std::less<>{});
  }

  SECTION("descending comparator")
  {
    check_sort_case(comms, input, cuda::std::greater<>{});
  }
}

C2H_CCCLRT_TEST("all equal inputs", "[multi_gpu][sort]", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = get_comms();
  std::vector<c2h::host_vector<T>> input(comms.size());

  for (auto& local : input)
  {
    local.assign(10, make_value<T>(1, 1));
  }

  SECTION("ascending comparator")
  {
    check_sort_case(comms, input, cuda::std::less<>{});
  }

  SECTION("descending comparator")
  {
    check_sort_case(comms, input, cuda::std::greater<>{});
  }
}

C2H_CCCLRT_TEST("inputs with many equal keys", "[multi_gpu][sort]", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = get_comms();
  std::vector<c2h::host_vector<T>> input(comms.size());

  for (cuda::std::size_t rank = 0; rank < input.size(); ++rank)
  {
    auto& local = input[rank];
    local.resize(10);

    for (cuda::std::size_t item = 0; item < local.size(); ++item)
    {
      const auto key = static_cast<cuda::std::int64_t>(item % 2);
      local[item]    = make_value<T>(key, static_cast<cuda::std::int64_t>(rank * local.size() + item));
    }
  }

  SECTION("ascending comparator")
  {
    check_sort_case(comms, input, cuda::std::less<>{});
  }

  SECTION("descending comparator")
  {
    check_sort_case(comms, input, cuda::std::greater<>{});
  }
}

C2H_CCCLRT_TEST("presorted inputs", "[multi_gpu][sort]", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = get_comms();
  std::vector<c2h::host_vector<T>> input(comms.size());

  for (cuda::std::size_t rank = 0; rank < input.size(); ++rank)
  {
    auto& local = input[rank];
    local.resize(10);

    for (cuda::std::size_t item = 0; item < local.size(); ++item)
    {
      const auto key = static_cast<cuda::std::int64_t>(rank * local.size() + item);
      local[item]    = make_value<T>(key, key);
    }
  }

  SECTION("ascending comparator")
  {
    check_sort_case(comms, input, cuda::std::less<>{});
  }

  SECTION("descending comparator")
  {
    check_sort_case(comms, input, cuda::std::greater<>{});
  }
}

C2H_CCCLRT_TEST("reverse-sorted inputs", "[multi_gpu][sort]", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = get_comms();
  std::vector<c2h::host_vector<T>> input(comms.size());

  for (cuda::std::size_t rank = 0; rank < input.size(); ++rank)
  {
    auto& local = input[rank];
    local.resize(10);

    for (cuda::std::size_t item = 0; item < local.size(); ++item)
    {
      const auto key = static_cast<cuda::std::int64_t>(input.size() * local.size() - (rank * local.size() + item));
      local[item]    = make_value<T>(key, key);
    }
  }

  SECTION("ascending comparator")
  {
    check_sort_case(comms, input, cuda::std::less<>{});
  }

  SECTION("descending comparator")
  {
    check_sort_case(comms, input, cuda::std::greater<>{});
  }
}

C2H_CCCLRT_TEST("skewed rank sizes", "[multi_gpu][sort]", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = get_comms();
  auto rng   = make_rng(C2H_SEED(2));

  std::vector<c2h::host_vector<T>> input(comms.size());
  for (cuda::std::size_t rank = 0; rank < input.size(); ++rank)
  {
    fill_random(input[rank], rank == 0 ? 20 : 1, rng);
  }

  SECTION("ascending comparator")
  {
    check_sort_case(comms, input, cuda::std::less<>{});
  }

  SECTION("descending comparator")
  {
    check_sort_case(comms, input, cuda::std::greater<>{});
  }
}

C2H_CCCLRT_TEST("nonstandard comparator", "[multi_gpu][sort]")
{
  auto comms = get_comms();
  std::vector<c2h::host_vector<int>> input(comms.size());

  for (cuda::std::size_t rank = 0; rank < input.size(); ++rank)
  {
    auto& local = input[rank];
    local.resize(10);

    for (cuda::std::size_t item = 0; item < local.size(); ++item)
    {
      const auto magnitude = static_cast<int>((rank + item) % 5);
      local[item]          = item % 2 == 0 ? magnitude : -magnitude;
    }
  }

  check_sort_case(comms, input, abs_less{});
}
