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
 * @brief This test makes sure we can generate a dot file with events
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main()
{
// TODO (miscco): Make it work for windows
#if !defined(_CCCL_COMPILER_MSVC)
  // Generate a random filename
  int r = rand();

  char filename[64];
  snprintf(filename, 64, "output_%d.dot", r);
  // fprintf(stderr, "filename %s\n", filename);
  setenv("CUDASTF_DOT_FILE", filename, 1);
  setenv("CUDASTF_DOT_IGNORE_PREREQS", "0", 1);

  stream_ctx ctx;

  auto lA = ctx.logical_data(shape_of<slice<char>>(64));
  ctx.task(lA.write())->*[](cudaStream_t, auto) {};
  ctx.task(lA.rw())->*[](cudaStream_t, auto) {};
  ctx.finalize();

  // Call this explicitely for the purpose of the test
  reserved::dot::instance().finish();

  // Make sure the file exists, and erase it
  // fprintf(stderr, "ERASE. ...\n");
  EXPECT(access(filename, F_OK) != -1);

  EXPECT(unlink(filename) == 0);
#endif // !_CCCL_COMPILER_MSVC
}
