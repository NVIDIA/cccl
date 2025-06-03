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
  auto epoch = ctx.epoch();

  char *A = nullptr, *B = nullptr, *C = nullptr;
  cuda_safe_call(cudaMalloc(&A, 64));
  cuda_safe_call(cudaMalloc(&B, 64));
  cuda_safe_call(cudaMalloc(&C, 64));

  ctx.parallel_for(box(64), epoch /*lA.write()*/).set_symbol("initA")->*[=]__device__(size_t i) {
    A[i] = 0;
  };
  ctx.parallel_for(box(64), epoch /*lB.write()*/).set_symbol("initB")->*[=]__device__(size_t i) {
    B[i] = 1;
  };
  ctx.parallel_for(box(64), epoch /*lC.write()*/).set_symbol("initC")->*[=]__device__(size_t i) {
    C[i] = 2;
  };

  for (size_t j = 0; j < 3; j++)
  {
    epoch++;
    ctx.parallel_for(box(64), epoch /*lA.rw()*/).set_symbol("f1")->*[=]__device__(size_t i) {
      ++A[i];
    };
    auto guard = ctx.dot_section("sec_loop " + ::std::to_string(j));
    for (size_t i = 0; i < 2; i++)
    {
      epoch++;
      auto guard_inner = ctx.dot_section("sec_inner_loop " + ::std::to_string(i));
      ctx.parallel_for(box(64), epoch /*lA.read(), lB.rw()*/).set_symbol("f2")->*[=]__device__(size_t i) {
        B[i] += A[i];
      };
      ctx.parallel_for(box(64), epoch /*lA.read(), lC.rw()*/).set_symbol("f2")->*[=]__device__(size_t i) {
        B[i] += A[i];
      };
      epoch++;
      ctx.parallel_for(box(64), epoch /*lB.read(), lC.read(), lA.rw()*/).set_symbol("f3")->*[=]__device__(size_t i) {
        A[i] += B[i] + C[i];
      };
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
