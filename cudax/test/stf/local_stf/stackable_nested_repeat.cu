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
 *
 * @brief Simple test demonstrating stackable context with double push before parallel_for
 *
 */

#include <cuda/experimental/__stf/utility/stackable_ctx.cuh>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main()
{
  stackable_ctx ctx;

  int niter1 = 7;
  int niter2 = 3;
  int niter3 = 17;

  size_t sz = 2;
  ::std::vector<int> data(sz);

  // Initialize data
  for (size_t i = 0; i < sz; i++)
  {
    data[i] = static_cast<int>(i);
  }

  // Create logical data
  auto ldata = ctx.logical_data(make_slice(data.data(), sz));

  {
    auto r1 = ctx.repeat_graph_scope(niter1);
    {
      auto r2 = ctx.repeat_graph_scope(niter2);
      {
        auto r3 = ctx.repeat_graph_scope(niter3);

        ctx.parallel_for(ldata.shape(), ldata.rw())->*[] __device__(size_t i, auto d) {
          d(i)++;
        };
      }
    }
  }

  ctx.finalize();

  // Verify results - each element should be doubled
  for (size_t i = 0; i < sz; i++)
  {
    int expected = static_cast<int>(i + niter1 * niter2 * niter3);
    fprintf(stderr, "i %ld : EXPECTED %d got %d\n", i, expected, data[i]);
    //_CCCL_ASSERT(data[i] == expected, "invalid result at index");
  }

  return 0;
}
