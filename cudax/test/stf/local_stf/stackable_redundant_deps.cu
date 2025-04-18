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
 * @brief Ensure we can use the same logical data multiple times in the same
 *        task even with different access modes (which should be combined)
 */

#include <cuda/experimental/__stf/utility/stackable_ctx.cuh>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

// a = b + 1;
template <typename T>
__global__ void add(T* a, const T* b)
{
  *a = *b + 1;
}

int main()
{
  stackable_ctx ctx;

  int var         = 42;
  auto var_handle = ctx.logical_data(make_slice(&var, 1));

  ctx.push();

  // da and db are for the same variable : we expect it to be equivalent to a RW access
  // ctx.task(var_handle.write(), var_handle.read())->*[](cudaStream_t stream, auto da, auto db) {
  ctx.task(var_handle.write(), var_handle.read())->*[](cudaStream_t stream, auto da, auto db) {
    add<<<1, 1, 0, stream>>>(da.data_handle(), db.data_handle());
  };

  ctx.pop();

  // Read that value on the host
  ctx.host_launch(var_handle.read())->*[](auto da) {
    int result = *da.data_handle();
    (void) result;
    fprintf(stderr, "RESULT %d\n", result);
    assert(result == 43);
  };

  ctx.finalize();
}
