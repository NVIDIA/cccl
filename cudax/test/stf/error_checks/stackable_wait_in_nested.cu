//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Ensure wait() in a nested stackable context triggers an abort
 */

#include <cuda/experimental/stf.cuh>

#include <csignal>

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

  auto lval = sctx.logical_data(shape_of<scalar_view<int>>());

  sctx.parallel_for(box(1), lval.write())->*[] __device__(size_t, auto val) {
    *val = 42;
  };

  {
    auto scope = sctx.graph_scope();

    sctx.parallel_for(box(1), lval.rw())->*[] __device__(size_t, auto val) {
      *val += 1;
    };

    should_abort = true;
    sctx.wait(lval); // wait() in nested context must abort
  }

  _CCCL_ASSERT(false, "This should not be reached");
  return EXIT_FAILURE;
}
