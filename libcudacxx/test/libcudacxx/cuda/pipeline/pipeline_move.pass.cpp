//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: pre-sm-70

// UNSUPPORTED: force-tile
// error: asm statement unsupported in tile mode

#include <cuda/pipeline>
#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "test_macros.h"

// Emulates one thread of a two-thread group from a single thread.
template <int Rank>
struct two_thread_test_group
{
  TEST_HOST_DEVICE_FUNC void sync() const {}
  TEST_HOST_DEVICE_FUNC int size() const
  {
    return 2;
  }
  TEST_HOST_DEVICE_FUNC int thread_rank() const
  {
    return Rank;
  }
};

template <cuda::thread_scope Scope>
TEST_HOST_DEVICE_FUNC void test()
{
  cuda::pipeline_shared_state<Scope, 1> shared_state;
  {
    auto pipe  = cuda::make_pipeline(two_thread_test_group<0>{}, &shared_state);
    auto moved = cuda::std::move(pipe);
  } // Only `moved` quits; the moved-from `pipe` must not quit again.

  // Rank 1 is the last to quit, so it must release the shared state.
  auto last = cuda::make_pipeline(two_thread_test_group<1>{}, &shared_state);
  assert(last.quit());
}

int main(int, char**)
{
  test<cuda::thread_scope_system>();
  test<cuda::thread_scope_device>();
  test<cuda::thread_scope_block>();

  return 0;
}
