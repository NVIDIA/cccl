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
#if _CCCL_CTK_BELOW(12, 4)
  fprintf(stderr, "Waiving test: while_graph_scope is only available since CUDA 12.4.\n");
  return 0;
#else
  stackable_ctx ctx;

  size_t sz = 1024;
  ::std::vector<int> data(sz);

  // Initialize data
  for (size_t i = 0; i < sz; i++)
  {
    data[i] = static_cast<int>(i);
  }

  // Create logical data
  auto ldata = ctx.logical_data(make_slice(data.data(), sz));

  auto liter1 = ctx.logical_data(shape_of<scalar_view<int>>());
  auto liter2 = ctx.logical_data(shape_of<scalar_view<int>>());

  int max_iter1 = 2;
  int max_iter2 = 3;

  // First scope with first context push and first data push
  {
    // Initialize iteration counter
    ctx.parallel_for(box(1), liter1.write())->*[] __device__(size_t, auto iter1) {
      *iter1 = 0;
    };

    auto while_guard_1 = ctx.while_graph_scope();

    // NESTED second scope with second context push and second data push
    {
      ctx.parallel_for(box(1), liter2.write())->*[] __device__(size_t, auto iter2) {
        *iter2 = 0;
      };

      auto while_guard_2 = ctx.while_graph_scope();

      // Now do the parallel_for operation - double each element
      ctx.parallel_for(ldata.shape(), ldata.rw())->*[] __device__(size_t i, auto d) {
        d(i)++;
      };

      while_guard_2.update_cond(liter2.rw())->*[max_iter2] __device__(auto iter2) {
        (*iter2)++;
        return (*iter2 < max_iter2); // Continue if not converged and under limit
      };
    }

    while_guard_1.update_cond(liter1.rw())->*[max_iter1] __device__(auto iter1) {
      (*iter1)++;
      return (*iter1 < max_iter1); // Continue if not converged and under limit
    };
  }

  ctx.finalize();

  // Verify results - each element should be doubled
  for (size_t i = 0; i < sz; i++)
  {
    int expected = static_cast<int>(i + max_iter1 * max_iter2);
    _CCCL_ASSERT(data[i] == expected, "invalid result at index");
  }

  return 0;
#endif // !_CCCL_CTK_BELOW(12, 4)
}
