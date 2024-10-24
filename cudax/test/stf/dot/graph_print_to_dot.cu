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
 * @brief This test makes sure we can generate a dot file
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

__global__ void dummy() {}

int main()
{
// TODO (miscco): Make it work for windows
#if !defined(_CCCL_COMPILER_MSVC)
  // Generate a random filename
  int r = rand();

  char filename[64];
  snprintf(filename, 64, "output_%d.dot", r);
  // fprintf(stderr, "filename %s\n", filename);

  graph_ctx ctx;

  auto lA = ctx.logical_data(shape_of<slice<char>>(64));
  ctx.task(lA.write())->*[](cudaStream_t s, auto) {
    dummy<<<1, 1, 0, s>>>();
  };
  ctx.task(lA.rw())->*[](cudaStream_t s, auto) {
    dummy<<<1, 1, 0, s>>>();
  };
  ctx.print_to_dot(filename, cudaGraphDebugDotFlagsVerbose);
  ctx.finalize();

  // Make sure the file exists, and erase it
  EXPECT(access(filename, F_OK) != -1);

  EXPECT(unlink(filename) == 0);
#endif // !_CCCL_COMPILER_MSVC
}
