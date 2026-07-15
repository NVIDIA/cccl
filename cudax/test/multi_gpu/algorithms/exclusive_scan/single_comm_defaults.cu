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
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/std/iterator>

#include <cuda/experimental/__multi_gpu/algorithm/scan/scan.h>

#include <numeric>
#include <vector>

#include <algorithm_common.h>
#include <nccl_test_common.h>
#include <testing.cuh>

MULTI_GPU_TEST("exclusive_scan single-comm, overloads default values", )
{
  using T  = cuda::std::int32_t;
  using Op = ::cuda::std::plus<>;

  constexpr auto init = T{};
  constexpr T ident   = cuda::identity_element<Op, T>();
  constexpr auto op   = Op{};

  const auto comms   = this->communicators();
  const auto streams = nccl_test_util::make_streams();

  std::vector<cuda::device_buffer<T>> in;
  std::vector<cuda::device_buffer<T>> out;
  std::vector<cuda::stream_ref> envs;

  in.reserve(comms.size());
  out.reserve(comms.size());
  envs.reserve(comms.size());

  constexpr auto values_per_rank = 3;

  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto first                                  = static_cast<T>(comms[i].rank() * values_per_rank + 1);
    const cuda::std::array<T, values_per_rank> values = {first, first + 1, first + 2};

    in.emplace_back(cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), values));
    out.emplace_back(cuda::make_device_buffer<T>(
      streams[i], comms[i].logical_device().underlying_device(), cuda::std::size(values), cuda::no_init));
    envs.emplace_back(streams[i]);
  }

  auto outputs = make_output_iterators(out);

  const auto expected_values = [&] {
    std::vector<T> reference(static_cast<cuda::std::size_t>(comms.front().size()) * values_per_rank);

    std::iota(reference.begin(), reference.end(), T{1});

    std::vector<T> expected_values(reference.size());

    std::exclusive_scan(reference.begin(), reference.end(), expected_values.begin(), init, op);

    return expected_values;
  }();

  const auto check_outputs = [&] {
    const auto exp_span = cuda::std::span{expected_values};

    for (cuda::std::size_t i = 0; i < out.size(); ++i)
    {
      const auto expected_for_rank = exp_span.subspan(comms[i].rank() * values_per_rank, values_per_rank);
      const auto expected =
        cuda::make_buffer<T>(out[i].stream(), cuda::mr::legacy_pinned_memory_resource{}, expected_for_rank);

      REQUIRE_THAT(out[i], Equals(expected));
    }
  };

  SECTION("Default init, op, ident (all)")
  {
    run_threaded(comms.size(), [&](cuda::std::size_t i) {
      cudax::exclusive_scan(comms[i], envs[i], in[i], outputs[i]);
    });
    check_outputs();
  }

  SECTION("Default op, ident")
  {
    run_threaded(comms.size(), [&](cuda::std::size_t i) {
      cudax::exclusive_scan(comms[i], envs[i], in[i], outputs[i], init);
    });
    check_outputs();
  }

  SECTION("Default ident")
  {
    run_threaded(comms.size(), [&](cuda::std::size_t i) {
      cudax::exclusive_scan(comms[i], envs[i], in[i], outputs[i], init, op);
    });
    check_outputs();
  }

  SECTION("Default none")
  {
    run_threaded(comms.size(), [&](cuda::std::size_t i) {
      cudax::exclusive_scan(comms[i], envs[i], in[i], outputs[i], init, op, ident);
    });
    check_outputs();
  }
}
