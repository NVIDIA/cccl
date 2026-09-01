//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Ensure dim4 rejects coordinates equal to an extent
 */

#include <cuda/experimental/__stf/utility/dimensions.cuh>

#include <csignal>
#include <cstdlib>
#include <cstring>

using namespace cuda::experimental::stf;

volatile ::std::sig_atomic_t should_abort = 0;

void cleanupRoutine(int /*unused*/)
{
  ::std::_Exit(should_abort ? EXIT_SUCCESS : EXIT_FAILURE);
}

int main()
{
#if _CCCL_COMPILER(MSVC)
  signal(SIGABRT, &cleanupRoutine);
#else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
  struct sigaction sigabrt_action{};
  memset(&sigabrt_action, 0, sizeof(sigabrt_action));
  sigabrt_action.sa_handler = &cleanupRoutine;

  if (sigaction(SIGABRT, &sigabrt_action, nullptr) != 0)
  {
    perror("sigaction SIGABRT");
    exit(EXIT_FAILURE);
  }
#endif // !_CCCL_COMPILER(MSVC)

  const dim4 dims(4, 5, 6, 7);

  should_abort = 1;
  (void) dims.get_index(pos4(4, 0, 0, 0));
  should_abort = 0;

  return EXIT_FAILURE;
}
