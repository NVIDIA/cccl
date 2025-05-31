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

  ctx.task(epoch /*lA.write()*/).set_symbol("initA")->*[=](cudaStream_t) {
    A[0] = 0;
  };
  ctx.task(epoch /*lB.write()*/).set_symbol("initB")->*[=](cudaStream_t) {
    B[0] = 1;
  };
  ctx.task(epoch /*lC.write()*/).set_symbol("initC")->*[=](cudaStream_t) {
    C[0] = 2;
  };

  for (size_t j = 0; j < 3; j++)
  {
    epoch++;
    ctx.task(epoch /*lA.rw()*/).set_symbol("f1")->*[=](cudaStream_t) {
      ++A[0];
    };
    auto guard = ctx.dot_section("sec_loop " + ::std::to_string(j));
    for (size_t i = 0; i < 2; i++)
    {
      epoch++;
      auto guard_inner = ctx.dot_section("sec_inner_loop " + ::std::to_string(i));
      ctx.task(epoch /*lA.read(), lB.rw()*/).set_symbol("f2")->*[=](cudaStream_t) {
        B[0] += A[0];
      };
      ctx.task(epoch /*lA.read(), lC.rw()*/).set_symbol("f2")->*[=](cudaStream_t) {
        B[0] += A[0];
      };
      epoch++;
      ctx.task(epoch /*lB.read(), lC.read(), lA.rw()*/).set_symbol("f3")->*[=](cudaStream_t) {
        A[0] += B[0] + C[0];
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
