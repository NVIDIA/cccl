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
#include <cuda/std/optional>
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
TEST_HOST_DEVICE_FUNC void test_move_construct()
{
  cuda::pipeline_shared_state<Scope, 1> shared_state;
  cuda::std::optional<cuda::pipeline<Scope>> moved;
  {
    auto pipe = cuda::make_pipeline(two_thread_test_group<0>{}, &shared_state);
    moved.emplace(cuda::std::move(pipe));
  } // The moved-from `pipe` must not quit.

  // `moved` still holds rank 0's reference, so rank 1 is not the last to quit.
  auto other = cuda::make_pipeline(two_thread_test_group<1>{}, &shared_state);
  assert(!other.quit());
  assert(moved->quit());
}

template <cuda::thread_scope Scope>
TEST_HOST_DEVICE_FUNC void test_move_assign()
{
  cuda::pipeline_shared_state<Scope, 1> lhs_state;
  cuda::pipeline_shared_state<Scope, 1> rhs_state;
  auto lhs = cuda::make_pipeline(two_thread_test_group<0>{}, &lhs_state);
  {
    auto rhs = cuda::make_pipeline(two_thread_test_group<0>{}, &rhs_state);
    lhs      = cuda::std::move(rhs);
  } // The moved-from `rhs` must not quit.

  // The assignment released rank 0's reference to `lhs_state`, so rank 1 is the last to quit.
  auto lhs_other = cuda::make_pipeline(two_thread_test_group<1>{}, &lhs_state);
  assert(lhs_other.quit());

  // `lhs` now holds rank 0's reference to `rhs_state`.
  auto rhs_other = cuda::make_pipeline(two_thread_test_group<1>{}, &rhs_state);
  assert(!rhs_other.quit());
  assert(lhs.quit());
}

template <cuda::thread_scope Scope>
TEST_HOST_DEVICE_FUNC void test()
{
  test_move_construct<Scope>();
  test_move_assign<Scope>();
}

int main(int, char**)
{
  test<cuda::thread_scope_system>();
  test<cuda::thread_scope_device>();
  test<cuda::thread_scope_block>();

  return 0;
}
