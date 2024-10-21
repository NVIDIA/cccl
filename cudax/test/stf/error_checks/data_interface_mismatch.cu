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
 * @brief Ensure an error is detected dynamically if we access a data instance
 *        with the wrong interface type
 */

#include <cuda/experimental/__stf/graph/graph_ctx.cuh>
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

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

template <typename Ctx, size_t n>
void run(double (&X)[n])
{
  Ctx ctx;
  // This creates an untyped logical data that is implicitly a vector of size
  // n. Had the code used `auto` instead of `logical_data_untyped`, errors
  // would have been rejected statically. We want to disable static checking
  // for the purposes of this test.
  logical_data_untyped handle_X = ctx.logical_data(X);

  // Here we create a dynamically-typed task, again to go around static typechecking.
  auto t = ctx.task();
  t.add_deps(handle_X.rw());

  t->*[&](auto&) {
    should_abort = true;
    // We have a programming error here with a vector of `double` accessad as a vector of `float`.
    handle_X.instance<slice<float>>(t);
    should_abort = false;
  };

  assert(0 && "This should not be reached");
}

int main()
{
  /* Setup an handler to catch the SIGABRT signal during the programming error */
#if defined(_CCCL_COMPILER_MSVC)
  signal(SIGABRT, &cleanupRoutine);
#else // ^^^ _CCCL_COMPILER_MSVC ^^^ / vvv !_CCCL_COMPILER_MSVC
  struct sigaction sigabrt_action
  {};
  memset(&sigabrt_action, 0, sizeof(sigabrt_action));
  sigabrt_action.sa_handler = &cleanupRoutine;

  if (sigaction(SIGABRT, &sigabrt_action, nullptr) != 0)
  {
    perror("sigaction SIGABRT");
    exit(EXIT_FAILURE);
  }
#endif // !_CCCL_COMPILER_MSVC

  const int n = 12;
  double X[n];

  for (int ind = 0; ind < n; ind++)
  {
    X[ind] = 1.0 * ind;
  }

  // We can't run both stream and graph tests because either will abort the program. So choose one at random.
  srand(static_cast<unsigned>(time(nullptr)));
  if (rand() % 2 == 0)
  {
    run<stream_ctx>(X);
  }
  else
  {
    run<graph_ctx>(X);
  }

  assert(0 && "This should not be reached");
  return EXIT_FAILURE;
}
