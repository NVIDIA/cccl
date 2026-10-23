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
 * @brief Test that ensures we catch programming errors with inconsistent access modes in nested contexts
 */

#include <cuda/experimental/stf.cuh>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

using namespace cuda::experimental::stf;

int main()
{
  stackable_ctx sctx;

  const size_t sz = 1024;
  ::std::vector<int> data(sz);

  // Initialize data
  for (size_t i = 0; i < sz; i++)
  {
    data[i] = static_cast<int>(i);
  }

  // Create logical data
  auto ldata = sctx.logical_data(make_slice(data.data(), sz));

  bool caught = false;

  // First scope: push with READ access mode
  {
    const stackable_ctx::graph_scope_guard scope1{sctx};
    ldata.push(access_mode::read);

    // NESTED second scope: attempt to escalate from read to rw access mode.
    // This is an invalid access mode transition and must be reported.
    {
      const stackable_ctx::graph_scope_guard scope2{sctx};
      try
      {
        ldata.push(access_mode::rw);
      }
      catch (const ::std::logic_error& e)
      {
        caught = true;
        fprintf(stderr, "Caught expected error: %s\n", e.what());
      }
    }
  }

  sctx.finalize();

  return caught ? EXIT_SUCCESS : EXIT_FAILURE;
}
