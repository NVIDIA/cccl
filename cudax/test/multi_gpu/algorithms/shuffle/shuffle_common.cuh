// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CUDAX_TEST_MULTI_GPU_ALGORITHMS_SHUFFLE_SHUFFLE_COMMON_CUH
#define CUDAX_TEST_MULTI_GPU_ALGORITHMS_SHUFFLE_SHUFFLE_COMMON_CUH

#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/memory_resource>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/span>
#include <cuda/std/type_traits>

#include <cuda/experimental/mgmn/shuffle>

#include <algorithm>
#include <random>
#include <vector>

#include <algorithm_common.h>
#include <nccl_test_common.h>

#include <c2h/catch2_test_helper.h>

namespace shuffle_test
{
struct payload
{
  cuda::std::uint64_t occurrence;
  cuda::std::uint64_t value;
};

using types = c2h::type_list<cuda::std::int64_t, double, payload>;

template <class T>
[[nodiscard]] T make_value(cuda::std::uint64_t id)
{
  if constexpr (cuda::std::is_same_v<T, payload>)
  {
    return {id, id % 7};
  }
  else
  {
    return static_cast<T>(id);
  }
}

template <class T>
[[nodiscard]] cuda::std::uint64_t identity(T value)
{
  if constexpr (cuda::std::is_same_v<T, payload>)
  {
    REQUIRE(value.value == value.occurrence % 7);
    return value.occurrence;
  }
  else
  {
    return static_cast<cuda::std::uint64_t>(value);
  }
}

template <class T>
[[nodiscard]] std::vector<T> read(const cuda::device_buffer<T>& input)
{
  std::vector<T> result(input.size());
  cuda::copy_bytes(
    input.stream(),
    input,
    cuda::std::span<T>{result},
    cuda::copy_configuration{input.stream().device(), cuda::host_memory_location, cuda::source_access_order::stream});
  input.stream().sync();
  return result;
}

template <class T, bool SingleComm>
void check(cuda::std::span<cudax::mgmn::nccl_communicator_ref> comms,
           const std::vector<cuda::std::size_t>& sizes,
           bool in_place,
           unsigned seed,
           bool duplicates = false)
{
  auto streams_owned = nccl_test_util::make_streams();
  auto streams       = std::vector<cuda::stream_ref>{streams_owned.begin(), streams_owned.end()};
  std::vector<cuda::device_buffer<T>> inputs;
  std::vector<cuda::device_buffer<T>> outputs;
  std::vector<T*> input_iters;
  std::vector<T*> output_iters;
  std::vector<std::vector<T>> original;
  std::vector<cuda::std::uint64_t> expected;
  cuda::std::uint64_t first           = 0;
  constexpr cuda::std::uint64_t guard = 999999999;
  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    auto& host = original.emplace_back(sizes[i] + 2, make_value<T>(guard));
    for (cuda::std::size_t j = 0; j < sizes[i]; ++j)
    {
      const auto id = duplicates ? (first + j) % 7 : first + j;
      host[j + 1]   = make_value<T>(id);
      expected.push_back(id);
    }
    first += sizes[i];
    const auto device = comms[i].logical_device().underlying_device();
    inputs.emplace_back(cuda::make_device_buffer<T>(streams[i], device, host));
    outputs.emplace_back(
      cuda::make_device_buffer<T>(streams[i], device, std::vector<T>(sizes[i] + 2, make_value<T>(guard))));
    input_iters.push_back(inputs.back().data() + 1);
    output_iters.push_back((in_place ? inputs.back().data() : outputs.back().data()) + 1);
  }
  if constexpr (SingleComm)
  {
    run_threaded(comms.size(), [&](cuda::std::size_t i) {
      std::mt19937 generator{seed};
      if (in_place)
      {
        cudax::mgmn::shuffle(cudax::distributed, comms[i], streams[i], input_iters[i], sizes[i], generator);
      }
      else
      {
        cudax::mgmn::shuffle_copy(
          cudax::distributed, comms[i], streams[i], input_iters[i], sizes[i], output_iters[i], generator);
      }
    });
  }
  else
  {
    std::mt19937 generator{seed};
    if (in_place)
    {
      cudax::mgmn::shuffle(cudax::distributed, comms, streams, input_iters, sizes, generator);
    }
    else
    {
      cudax::mgmn::shuffle_copy(cudax::distributed, comms, streams, input_iters, sizes, output_iters, generator);
    }
  }
  std::vector<cuda::std::uint64_t> actual;
  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto input  = read(inputs[i]);
    const auto output = in_place ? input : read(outputs[i]);
    REQUIRE(output.size() == sizes[i] + 2);
    REQUIRE(identity(output.front()) == guard);
    REQUIRE(identity(output.back()) == guard);
    for (cuda::std::size_t j = 0; j < sizes[i]; ++j)
    {
      actual.push_back(identity(output[j + 1]));
    }
    if (!in_place)
    {
      for (cuda::std::size_t j = 0; j < input.size(); ++j)
      {
        REQUIRE(identity(input[j]) == identity(original[i][j]));
      }
    }
  }
  if (!duplicates && actual.size() > 100)
  {
    // Movement is a smoke check, not evidence of uniform random sampling.
    REQUIRE(actual != expected);
  }
  std::sort(actual.begin(), actual.end());
  std::sort(expected.begin(), expected.end());
  REQUIRE(actual == expected);
}

template <class T, bool SingleComm>
void cases(cuda::std::span<cudax::mgmn::nccl_communicator_ref> comms)
{
  const auto shape    = GENERATE(0, 1, 2, 3, 4, 5);
  const auto in_place = GENERATE(false, true);
  const auto seed     = GENERATE(0U, 1234U);
  std::vector<cuda::std::size_t> sizes(comms.size());
  for (cuda::std::size_t i = 0; i < sizes.size(); ++i)
  {
    switch (shape)
    {
      case 0:
        sizes[i] = 0;
        break;
      case 1:
        sizes[i] = i == 0 ? 1 : 0;
        break;
      case 2:
        sizes[i] = 1;
        break;
      case 3:
        sizes[i] = i * 7 + 37;
        break;
      case 4:
        sizes[i] = i == sizes.size() - 1 ? 65539 : 0;
        break;
      case 5:
        sizes[i] = 16387 + i;
        break;
    }
  }
  CAPTURE(shape, in_place, seed);
  check<T, SingleComm>(comms, sizes, in_place, seed);
}
} // namespace shuffle_test

#endif // CUDAX_TEST_MULTI_GPU_ALGORITHMS_SHUFFLE_SHUFFLE_COMMON_CUH
