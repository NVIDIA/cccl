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
 * @brief This test makes sure we can generate a dot file with sections
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main()
{
// TODO (miscco): Make it work for windows
#if !_CCCL_COMPILER(MSVC)
  // Generate a random filename
  int r = rand();

  char filename[64];
  snprintf(filename, 64, "output_%d.dot", r);
  // fprintf(stderr, "filename %s\n", filename);
  setenv("CUDASTF_DOT_FILE", filename, 1);

  context ctx;

  auto lA = ctx.logical_data(shape_of<slice<char>>(64));
  auto lB = ctx.logical_data(shape_of<slice<char>>(64));
  auto lC = ctx.logical_data(shape_of<slice<char>>(64));
  ctx.task(lA.write()).set_symbol("initA")->*[](cudaStream_t, auto) {};
  ctx.task(lB.write()).set_symbol("initB")->*[](cudaStream_t, auto) {};
  ctx.task(lC.write()).set_symbol("initC")->*[](cudaStream_t, auto) {};
  for (size_t j = 0; j < 3; j++)
  {
    ctx.task(lA.rw()).set_symbol("f1")->*[](cudaStream_t, auto) {};
    auto guard = ctx.dot_section("sec_loop " + ::std::to_string(j));
    for (size_t i = 0; i < 2; i++)
    {
      auto guard_inner = ctx.dot_section("sec_inner_loop " + ::std::to_string(i));
      ctx.task(lA.read(), lB.rw()).set_symbol("f2")->*[](cudaStream_t, auto, auto) {};
      ctx.task(lA.read(), lC.rw()).set_symbol("f2")->*[](cudaStream_t, auto, auto) {};
      ctx.task(lB.read(), lC.read(), lA.rw()).set_symbol("f3")->*[](cudaStream_t, auto, auto, auto) {};
    }
  }
  ctx.finalize();

  // Call this explicitly for the purpose of the test
  reserved::dot::instance().finish();

  // Make sure the file exists, and erase it
  // fprintf(stderr, "ERASE. ...\n");
  EXPECT(access(filename, F_OK) != -1);

  EXPECT(unlink(filename) == 0);
#endif // !_CCCL_COMPILER(MSVC)
}
