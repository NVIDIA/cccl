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

#include <algorithm_common.h>
#include <nccl_test_common.h>
#include <testing.cuh>

MULTI_GPU_TEST("reduce, range overloads default values", )
{
  using T  = cuda::std::int32_t;
  using Op = ::cuda::std::plus<>;

  constexpr auto init = T{};
  constexpr T ident   = cuda::identity_element<Op, T>();
  constexpr auto op   = Op{};

  auto comms   = this->communicators();
  auto streams = nccl_test_util::make_streams();

  std::vector<cuda::device_buffer<T>> in;
  std::vector<cuda::device_buffer<T>> out;
  std::vector<decltype(::cuda::std::execution::env{::cuda::stream_ref{streams[0]}})> envs;

  in.reserve(comms.size());
  out.reserve(comms.size());
  envs.reserve(comms.size());
  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto values = {static_cast<T>(comms[i].rank())};

    in.emplace_back(cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), values));
    out.emplace_back(
      cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), 1, cuda::no_init));
    envs.emplace_back(::cuda::std::execution::env{::cuda::stream_ref{streams[i]}});
  }

  auto outputs = make_output_iterators(out);

  const auto expected = [&] {
    std::vector<T> reference;

    reference.reserve(comms.front().size());
    for (int r = 0; r < comms.front().size(); ++r)
    {
      reference.push_back(r);
    }

    const auto val = std::accumulate(reference.begin(), reference.end(), init, op);

    return cuda::make_buffer(cuda::stream_ref{::CUstream{}}, cuda::mr::legacy_pinned_memory_resource{}, 1, val);
  }();

  SECTION("Default init, op, ident (all)")
  {
    cudax::reduce(comms, envs, in, outputs);

    for (const auto& buf : out)
    {
      REQUIRE_THAT(buf, Equals(expected));
    }
  }

  SECTION("Default op, ident")
  {
    cudax::reduce(comms, envs, in, outputs, init);

    for (const auto& buf : out)
    {
      REQUIRE_THAT(buf, Equals(expected));
    }
  }

  SECTION("Default ident")
  {
    cudax::reduce(comms, envs, in, outputs, init, op);

    for (const auto& buf : out)
    {
      REQUIRE_THAT(buf, Equals(expected));
    }
  }

  SECTION("Default none")
  {
    cudax::reduce(comms, envs, in, outputs, init, op, ident);

    for (const auto& buf : out)
    {
      REQUIRE_THAT(buf, Equals(expected));
    }
  }
}
