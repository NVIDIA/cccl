//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/ranges>
#include <cuda/std/span>

#include <cuda/experimental/__multi_gpu/algorithm/sort/sort.h>

#include <algorithm>
#include <vector>

#include <nccl_test_common.h>
#include <testing.cuh>

#include "sort_common.cuh"
#include <c2h/catch2_test_helper.h>

namespace
{
// Orders exactly like the defaulted `cuda::std::less<>`, but is a distinct type, so the explicit
// section really does exercise a separate instantiation instead of re-running the default one.
struct custom_less : cuda::std::less<>
{};
} // namespace

MULTI_GPU_TEST("sort, range overloads default values", )
{
  using T = cuda::std::int32_t;

  constexpr auto cmp = custom_less{};

  auto comms = this->communicators();
  auto rng   = sort_test_util::make_rng(C2H_SEED(2));

  // A shape with a bit of everything: unequal rank sizes and duplicate keys, so a defaulted
  // comparator that silently differed from `less` would show up in the comparison.
  constexpr auto values_per_rank = 100;
  std::vector<std::vector<T>> host_inputs(comms.size());
  for (cuda::std::size_t rank = 0; rank < host_inputs.size(); ++rank)
  {
    sort_test_util::fill_random(host_inputs[rank], values_per_rank + rank, rng);
  }

  const auto expected = sort_test_util::sorted_reference(host_inputs, cmp);

  auto streams      = nccl_test_util::make_streams();
  auto environments = std::vector<cuda::stream_ref>{streams.begin(), streams.end()};

  // Both overloads must land on the same global order, so they share one checker. `device_vec` is
  // rebuilt per section because `sort` permutes it in place.
  const auto check_sorted = [&](const auto& device_vec) {
    sort_test_util::check_rank_sizes(comms, device_vec, host_inputs);

    const auto output = sort_test_util::gather_outputs(comms, device_vec);

    REQUIRE(std::is_sorted(output.begin(), output.end(), cmp));
    sort_test_util::check_matches(streams.front(), output, expected);
  };

  SECTION("Default comparator")
  {
    auto device_vec = sort_test_util::make_device_inputs(comms, environments, host_inputs);

    cudax::sort(cudax::distributed,
                comms,
                environments,
                device_vec | cuda::std::views::transform(cuda::std::ranges::begin),
                device_vec | cuda::std::views::transform(cuda::std::ranges::size));

    check_sorted(device_vec);
  }

  SECTION("Default none")
  {
    auto device_vec = sort_test_util::make_device_inputs(comms, environments, host_inputs);

    cudax::sort(cudax::distributed,
                comms,
                environments,
                device_vec | cuda::std::views::transform(cuda::std::ranges::begin),
                device_vec | cuda::std::views::transform(cuda::std::ranges::size),
                cmp);

    check_sorted(device_vec);
  }
}
