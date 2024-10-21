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
 * @brief Ensure an error is detected if we can finalize more than once
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
  // This test only works when assert() is enabled in
#ifndef NDEBUG
  /* Setup an handler to catch the SIGABRT signal during the programming error */
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

  const int n = 12;
  double X[n];

  for (int ind = 0; ind < n; ind++)
  {
    X[ind] = 1.0 * ind;
  }

  // This creates a handle that is implicitly a vector of size n
  auto lX = ctx.logical_data(X);

  ctx.task(lX.rw())->*[](cudaStream_t, auto) { /* no-op */ };

  ctx.finalize();

  should_abort = true;
  // We cannot call sync twice
  ctx.finalize();

  assert(0 && "This should not be reached");
  return EXIT_FAILURE;
#endif
}
