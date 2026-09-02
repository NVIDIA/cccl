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
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/std/type_traits>

#include <cuda/experimental/__multi_gpu/algorithm/reduce/segmented_reduce.h>

#include <exception>
#include <future>
#include <numeric>
#include <vector>

#include <algorithm_common.h>
#include <nccl_test_common.h>
#include <testing.cuh>

MULTI_GPU_TEST("segmented_reduce single-comm, overloads default values", )
{
  using T      = cuda::std::int32_t;
  using Op     = ::cuda::std::plus<>;
  using offset = cuda::std::int32_t;

  constexpr auto init = T{};
  constexpr T ident   = cuda::identity_element<Op, T>();
  constexpr auto op   = Op{};

  auto comms   = this->communicators();
  auto streams = nccl_test_util::make_streams();

  constexpr offset values_per_segment = 10;
  const std::vector<offset> offset_values{0, values_per_segment, 2 * values_per_segment};
  // The last offset closes the final segment, so there is one less segment than offsets.
  const cuda::std::size_t num_segments = offset_values.size() - 1;

  std::vector<cuda::device_buffer<T>> in;
  std::vector<cuda::device_buffer<offset>> offsets;
  std::vector<cuda::device_buffer<T>> out;
  std::vector<decltype(::cuda::std::execution::env{::cuda::stream_ref{streams[0]}})> envs;

  in.reserve(comms.size());
  offsets.reserve(comms.size());
  out.reserve(comms.size());
  envs.reserve(comms.size());
  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto device = comms[i].logical_device().underlying_device();
    const auto rank   = comms[i].rank();

    std::vector<T> values(values_per_segment, static_cast<T>(rank));
    values.insert(values.end(), values_per_segment, static_cast<T>(rank + 1));

    in.emplace_back(cuda::make_device_buffer<T>(streams[i], device, values));
    offsets.emplace_back(cuda::make_device_buffer<offset>(streams[i], device, offset_values));
    out.emplace_back(cuda::make_device_buffer<T>(streams[i], device, num_segments, cuda::no_init));
    envs.emplace_back(::cuda::std::execution::env{::cuda::stream_ref{streams[i]}});
  }

  const auto expected = [&] {
    std::vector<T> reference(num_segments, init);

    for (int r = 0; r < comms.front().size(); ++r)
    {
      reference[0] += static_cast<T>(r) * values_per_segment;
      reference[1] += static_cast<T>(r + 1) * values_per_segment;
    }

    return cuda::make_buffer<T>(cuda::stream_ref{::CUstream{}}, cuda::mr::legacy_pinned_memory_resource{}, reference);
  }();

  // Segment `s` covers [begin_offsets[s], end_offsets[s]), so the end offsets are just the begin
  // offsets shifted by one. The single-communicator overload blocks on a collective, so every
  // rank runs on its own thread and all Catch2 assertions stay on the main thread after the join.
  SECTION("Default init, op, ident (all)")
  {
    run_threaded(comms.size(), [&](cuda::std::size_t i) {
      cudax::segmented_reduce(
        cudax::broadcasted,
        comms[i],
        envs[i],
        in[i].begin(),
        num_segments,
        offsets[i].begin(),
        offsets[i].begin() + 1,
        out[i].begin());
    });

    for (const auto& buf : out)
    {
      REQUIRE_THAT(buf, Equals(expected));
    }
  }

  SECTION("Default op, ident")
  {
    run_threaded(comms.size(), [&](cuda::std::size_t i) {
      cudax::segmented_reduce(
        cudax::broadcasted,
        comms[i],
        envs[i],
        in[i].begin(),
        num_segments,
        offsets[i].begin(),
        offsets[i].begin() + 1,
        out[i].begin(),
        init);
    });

    for (const auto& buf : out)
    {
      REQUIRE_THAT(buf, Equals(expected));
    }
  }

  SECTION("Default ident")
  {
    run_threaded(comms.size(), [&](cuda::std::size_t i) {
      cudax::segmented_reduce(
        cudax::broadcasted,
        comms[i],
        envs[i],
        in[i].begin(),
        num_segments,
        offsets[i].begin(),
        offsets[i].begin() + 1,
        out[i].begin(),
        init,
        op);
    });

    for (const auto& buf : out)
    {
      REQUIRE_THAT(buf, Equals(expected));
    }
  }

  SECTION("Default none")
  {
    run_threaded(comms.size(), [&](cuda::std::size_t i) {
      cudax::segmented_reduce(
        cudax::broadcasted,
        comms[i],
        envs[i],
        in[i].begin(),
        num_segments,
        offsets[i].begin(),
        offsets[i].begin() + 1,
        out[i].begin(),
        init,
        op,
        ident);
    });

    for (const auto& buf : out)
    {
      REQUIRE_THAT(buf, Equals(expected));
    }
  }
}
