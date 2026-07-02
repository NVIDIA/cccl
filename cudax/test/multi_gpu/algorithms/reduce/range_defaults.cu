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
#include <cuda/std/cstdint>
#include <cuda/std/execution>
#include <cuda/std/type_traits>

#include <cuda/experimental/__multi_gpu/algorithm/reduce/reduce.h>

#include <numeric>
#include <vector>

#include <testing.cuh>

#include "../../communicators/nccl/nccl_test_helpers.cuh"

namespace
{
using value_type = cuda::std::int32_t;

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
} // namespace

MGMN_TEST("reduce, range overload with default initializer and default plus", )
{
  auto comms   = this->communicators();
  auto streams = nccl_test_util::make_streams();

  std::vector<cuda::device_buffer<value_type>> in;
  std::vector<cuda::device_buffer<value_type>> out;
  std::vector<decltype(::cuda::std::execution::env{::cuda::stream_ref{streams[0]}})> envs;
  std::vector<value_type> reference;

  in.reserve(comms.size());
  out.reserve(comms.size());
  envs.reserve(comms.size());
  for (int i = 0; i < comms.size(); ++i)
  {
    const auto values = {static_cast<value_type>(comms[i].rank())};
    in.emplace_back(
      cuda::make_device_buffer<value_type>(streams[i], comms[i].logical_device().underlying_device(), values));
    out.emplace_back(cuda::make_device_buffer<value_type>(
      streams[i], comms[i].logical_device().underlying_device(), 1, cuda::no_init));
    envs.emplace_back(::cuda::std::execution::env{::cuda::stream_ref{streams[i]}});
  }
  for (int r = 0; r < comms.front().size(); ++r)
  {
    reference.push_back(static_cast<value_type>(r));
  }

  auto outputs = make_output_iterators(out);

  cudax::reduce(comms, envs, in, outputs);

  for (auto& stream : streams)
  {
    stream.sync();
  }

  const value_type expected = std::accumulate(reference.begin(), reference.end(), value_type{});

  for (const auto& buf : out)
  {
    auto pool         = cuda::mr::legacy_pinned_memory_resource{};
    const auto actual = cuda::make_buffer(buf.stream(), pool, buf);

    REQUIRE(actual.size() == 1);
    REQUIRE(actual.front() == expected);
  }
}

MGMN_TEST("reduce, range overload with explicit initializer and default plus", )
{
  const value_type init{10};

  auto comms   = this->communicators();
  auto streams = nccl_test_util::make_streams();

  std::vector<cuda::device_buffer<value_type>> in;
  std::vector<cuda::device_buffer<value_type>> out;
  std::vector<decltype(::cuda::std::execution::env{::cuda::stream_ref{streams[0]}})> envs;
  std::vector<value_type> reference;

  in.reserve(comms.size());
  out.reserve(comms.size());
  envs.reserve(comms.size());
  for (int i = 0; i < comms.size(); ++i)
  {
    const auto values = {static_cast<value_type>(comms[i].rank())};
    in.emplace_back(
      cuda::make_device_buffer<value_type>(streams[i], comms[i].logical_device().underlying_device(), values));
    out.emplace_back(cuda::make_device_buffer<value_type>(
      streams[i], comms[i].logical_device().underlying_device(), 1, cuda::no_init));
    envs.emplace_back(::cuda::std::execution::env{::cuda::stream_ref{streams[i]}});
  }
  for (int r = 0; r < comms.front().size(); ++r)
  {
    reference.push_back(static_cast<value_type>(r));
  }

  auto outputs = make_output_iterators(out);

  cudax::reduce(comms, envs, in, outputs, init);

  for (auto& stream : streams)
  {
    stream.sync();
  }

  const value_type expected = std::accumulate(reference.begin(), reference.end(), init);

  for (const auto& buf : out)
  {
    auto pool         = cuda::mr::legacy_pinned_memory_resource{};
    const auto actual = cuda::make_buffer(buf.stream(), pool, buf);

    REQUIRE(actual.size() == 1);
    REQUIRE(actual.front() == expected);
  }
}
