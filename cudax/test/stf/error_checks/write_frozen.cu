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
 * @brief Ensure temporary data are destroyed
 *
 */

#include <csignal>

#include "cudastf/__stf/graph/graph_ctx.h"
#include "cudastf/__stf/stream/stream_ctx.h"

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
  struct sigaction sigabrt_action
  {};
  memset(&sigabrt_action, 0, sizeof(sigabrt_action));
  sigabrt_action.sa_handler = &cleanupRoutine;

  if (sigaction(SIGABRT, &sigabrt_action, nullptr) != 0)
  {
    perror("sigaction SIGABRT");
    exit(EXIT_FAILURE);
  }

  stream_ctx ctx;
  const size_t N = 16;
  int X[N];

  for (size_t i = 0; i < N; i++)
  {
    X[i] = i;
  }

  auto lX = ctx.logical_data(X);

  lX.freeze(access_mode::rw, data_place::current_device());

  // This is an illegal access because we cannot make a write access on a frozen data
  should_abort = true;

  ctx.task(lX.rw())->*[](cudaStream_t, auto) {};

  ctx.finalize();

  assert(0 && "This should not be reached");
  return EXIT_FAILURE;
}