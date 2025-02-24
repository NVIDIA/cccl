//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief An example to query statistics about graph instantiation
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main()
{
  async_resources_handle handle;
  for (size_t i = 0; i < 10; i++)
  {
    graph_ctx ctx(handle);

    // If i is a multiple of 3 we enable the cache, the first iteration will fill the cache
    ctx.set_graph_cache_policy([i]() {
      return (i % 3) == 0;
    });

    auto lA = ctx.logical_data(shape_of<slice<size_t>>(64));
    ctx.launch(lA.write())->*[] _CCCL_DEVICE(auto t, slice<size_t> A) {
      for (auto i : t.apply_partition(shape(A)))
      {
        A(i) = 2 * i;
      }
    };
    ctx.finalize();

    // Query statistics about the graph context : the first iteration needs to
    // instantiate the graph, then we will reuse graphs saved in the handle.
    auto* st = ctx.graph_get_cache_stat();

    // For the first iteration, or non multiple of 3 we have to instantiate, otherwise we should have a cache hit
    if (i == 0 || (i % 3) != 0)
    {
      EXPECT(st->instantiate_cnt == 1);
      EXPECT(st->update_cnt == 0);
    }
    else
    {
      EXPECT(st->instantiate_cnt == 0);
      EXPECT(st->update_cnt == 1);
    }
  }
}
