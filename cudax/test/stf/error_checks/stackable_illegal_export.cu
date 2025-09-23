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
 * @brief Ensure we detect erroneous access on non exportable stackable logical
 * data after the context was popped
 *
 */

#include <cuda/experimental/stf.cuh>

#include <csignal>

#include "cuda/experimental/__stf/stackable/stackable_ctx.cuh"

using namespace cuda::experimental::stf;

bool should_abort = false;

void cleanupRoutine(int /*unused*/)
{
  if (should_abort)
  {
    exit(EXIT_SUCCESS);
  }
  else
  {
    fprintf(stderr, "Unexpected SIGABRT !\n");
    exit(EXIT_FAILURE);
  }
}

int main()
{
  /* Setup an handler to catch the SIGABRT signal during the programming error */
#if _CCCL_COMPILER(MSVC)
  signal(SIGABRT, &cleanupRoutine);
#else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC)
  struct sigaction sigabrt_action{};
  memset(&sigabrt_action, 0, sizeof(sigabrt_action));
  sigabrt_action.sa_handler = &cleanupRoutine;

  if (sigaction(SIGABRT, &sigabrt_action, nullptr) != 0)
  {
    perror("sigaction SIGABRT");
    exit(EXIT_FAILURE);
  }
#endif // !_CCCL_COMPILER(MSVC)

  stackable_ctx sctx;
  sctx.push();

  auto lB = sctx.logical_data_no_export(shape_of<slice<int>>(1024));
  lB.set_symbol("B");

  sctx.parallel_for(lB.shape(), lB.write())->*[] __device__(size_t i, auto b) {
    b(i) = 42;
  };

  sctx.pop();

  // We are going to try to access B while it was not exportable, and that the
  // context where it was created has been popped: this should raise an error.
  should_abort = true;

  sctx.host_launch(lB.read())->*[](auto b) {
    for (size_t i = 0; i < b.size(); i++)
    {
      EXPECT(b(i) == 2 * (17 - 3 * (42 + 2 * i)));
    }
  };

  _CCCL_ASSERT(false, "This should not be reached");
  return EXIT_FAILURE;
}
