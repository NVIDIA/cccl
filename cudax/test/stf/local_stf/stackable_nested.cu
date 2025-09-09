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
  stackable_ctx sctx;

  size_t sz = 1024;
  ::std::vector<int> data(sz);

  // Initialize data
  for (size_t i = 0; i < sz; i++)
  {
    data[i] = static_cast<int>(i);
  }

  // Create logical data
  auto ldata = sctx.logical_data(make_slice(data.data(), sz));

  // First scope with first context push and first data push
  {
    stackable_ctx::graph_scope_guard scope1{sctx};
    ldata.push(access_mode::read);

    // NESTED second scope with second context push and second data push
    {
      stackable_ctx::graph_scope_guard scope2{sctx};
      ldata.push(access_mode::rw);

      // Now do the parallel_for operation - double each element
      sctx.parallel_for(ldata.shape(), ldata.rw())->*[] __device__(size_t i, auto d) {
        d(i) *= 2;
      };
    }
  }

  sctx.finalize();

  // Verify results - each element should be doubled
  for (size_t i = 0; i < sz; i++)
  {
    int expected = static_cast<int>(i * 2);
    _CCCL_ASSERT(data[i] == expected, "invalid result at index");
  }

  return 0;
}
