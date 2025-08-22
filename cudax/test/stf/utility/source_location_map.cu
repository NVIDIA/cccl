//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/source_location>

#include <cuda/experimental/__stf/utility/source_location.cuh>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

// Create a map indexed by source locations
::std::unordered_map<::cuda::std::source_location, int, reserved::source_location_hash, reserved::source_location_equal>
  stats_map;

void update_counter(::cuda::std::source_location loc = ::cuda::std::source_location::current())
{
  stats_map[loc]++;
}

void funcA()
{
  update_counter();
}

void funcB()
{
  update_counter();
}

int main()
{
  for (size_t i = 0; i < 10; i++)
  {
    funcA();
    funcB();
    funcB();
  }

  for (auto& e : stats_map)
  {
    auto& loc = e.first;
    fprintf(stderr, "loc.function_name() %s : count %d\n", loc.function_name(), e.second);
  }
}
