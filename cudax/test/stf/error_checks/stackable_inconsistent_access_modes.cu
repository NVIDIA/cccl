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
 * @brief Test that ensures we catch programming errors with inconsistent access modes in nested contexts
 * 
 * This test verifies that attempting to escalate from read-only to read-write access mode
 * in nested stackable contexts is properly caught and produces a clear error message.
 *
 */

#include <cuda/experimental/stf.cuh>

#include <csignal>

#include "cuda/experimental/__stf/utility/stackable_ctx.cuh"

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

  size_t sz = 1024;
  ::std::vector<int> data(sz);
  
  // Initialize data
  for (size_t i = 0; i < sz; i++)
  {
    data[i] = static_cast<int>(i);
  }

  // Create logical data
  auto ldata = sctx.logical_data(make_slice(data.data(), sz));

  // First scope: push with READ access mode
  {
    stackable_ctx::graph_scope_guard scope1{sctx};
    ldata.push(access_mode::read);
    
    // We are going to try to escalate from read to rw access mode in nested context:
    // this should raise an error.
    should_abort = true;
    
    // NESTED second scope: attempt to push with RW access mode
    // This should be caught as an invalid access mode escalation
    {
      stackable_ctx::graph_scope_guard scope2{sctx};
      ldata.push(access_mode::rw);  // This should trigger abort()!
    }
  }

  _CCCL_ASSERT(false, "This should not be reached");
  return EXIT_FAILURE;
}
