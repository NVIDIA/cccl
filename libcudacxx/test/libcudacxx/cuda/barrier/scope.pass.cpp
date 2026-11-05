//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/barrier>
#include <cuda/std/type_traits>

#include "test_macros.h"

// shared_memory_barrier is available with NVCC 13.4 and later.
#if TEST_CUDA_COMPILER(NVCC, >=, 13, 4)
static_assert(cuda::shared_memory_barrier<>::scope == cuda::thread_scope_block);
static_assert(cuda::shared_memory_barrier<cuda::thread_scope_block>::scope == cuda::thread_scope_block);
static_assert(cuda::shared_memory_barrier<cuda::thread_scope_cluster>::scope == cuda::thread_scope_cluster);
static_assert(cuda::std::is_same_v<decltype(cuda::shared_memory_barrier<>::scope), const cuda::thread_scope>);
static_assert(cuda::std::is_same_v<decltype(cuda::shared_memory_barrier<cuda::thread_scope_block>::scope),
                                   const cuda::thread_scope>);
static_assert(cuda::std::is_same_v<decltype(cuda::shared_memory_barrier<cuda::thread_scope_cluster>::scope),
                                   const cuda::thread_scope>);
#endif // TEST_CUDA_COMPILER(NVCC, >=, 13, 4)

struct completion
{
  TEST_FUNC void operator()() {}
};

template <cuda::thread_scope Scope>
TEST_FUNC void test()
{
  using Barrier       = cuda::barrier<Scope>;
  using CustomBarrier = cuda::barrier<Scope, completion>;

  static_assert(Barrier::scope == Scope);
  static_assert(CustomBarrier::scope == Scope);
  static_assert(cuda::std::is_same_v<decltype(Barrier::scope), const cuda::thread_scope>);
  static_assert(cuda::std::is_same_v<decltype(CustomBarrier::scope), const cuda::thread_scope>);
}

int main(int, char**)
{
  test<cuda::thread_scope_system>();
  test<cuda::thread_scope_device>();
  test<cuda::thread_scope_block>();
  test<cuda::thread_scope_thread>();

  return 0;
}
