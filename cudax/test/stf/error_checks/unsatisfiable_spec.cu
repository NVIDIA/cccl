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
 * @brief Ensure an error is raised if we try to ask for an unreasonnable
 *        amount of resources in a thread hierarchy spec
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
  /* Setup an handler to catch the SIGABRT signal during the programming error */
#ifndef NDEBUG
#  if defined(_CCCL_COMPILER_MSVC)
  signal(SIGABRT, &cleanupRoutine);
#  else // ^^^ _CCCL_COMPILER_MSVC ^^^ / vvv !_CCCL_COMPILER_MSVC
  struct sigaction sigabrt_action
  {};
  memset(&sigabrt_action, 0, sizeof(sigabrt_action));
  sigabrt_action.sa_handler = &cleanupRoutine;

  if (sigaction(SIGABRT, &sigabrt_action, nullptr) != 0)
  {
    perror("sigaction SIGABRT");
    exit(EXIT_FAILURE);
  }
#  endif // !_CCCL_COMPILER_MSVC

  context ctx;

  int X[128];
  auto lX = ctx.logical_data(X);

  should_abort = true;

  // We are asking an unreasonnable amount of threads per block
  auto spec = con(con<128000>());
  ctx.launch(spec, lX.rw())->*[] __device__(auto th, auto X) {
    X[th.rank()] = th.rank();
  };

  assert(0 && "This should not be reached");
  return EXIT_FAILURE;
#endif
}
