//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/mutex>

// template <thread_scope Scope = thread_scope_system>
// struct cuda::once_flag;
// using cuda::std::once_flag = cuda::once_flag<thread_scope_system>;

// UNSUPPORTED: libcpp-has-no-threads
// The host side launches through std::thread.

// UNSUPPORTED: pre-sm-70
// Device atomics used by call_once require sm_70.

// UNSUPPORTED: force-tile
// error: asm statement is not supported

#include <cuda/std/cassert>
#include <cuda/std/mutex>
#include <cuda/std/type_traits>

#include "concurrent_agents.h"
#include "cuda_space_selector.h"
#include "test_macros.h"

static_assert(cuda::std::is_same_v<cuda::once_flag<>, cuda::once_flag<cuda::thread_scope_system>>, "");
static_assert(cuda::std::is_same_v<cuda::once_flag<>, cuda::std::once_flag>, "");
static_assert(cuda::once_flag<cuda::thread_scope_thread>::_Scope == cuda::thread_scope_thread, "");
static_assert(cuda::once_flag<cuda::thread_scope_block>::_Scope == cuda::thread_scope_block, "");
static_assert(cuda::once_flag<cuda::thread_scope_cluster>::_Scope == cuda::thread_scope_cluster, "");
static_assert(cuda::once_flag<cuda::thread_scope_device>::_Scope == cuda::thread_scope_device, "");
static_assert(cuda::once_flag<cuda::thread_scope_system>::_Scope == cuda::thread_scope_system, "");

struct count_call
{
  int* calls;

  TEST_HOST_DEVICE_FUNC void operator()()
  {
    ++*calls;
  }
};

template <typename Flag, template <typename, typename> class Selector, typename Initializer = default_initializer>
TEST_HOST_DEVICE_FUNC void test()
{
  Selector<Flag, Initializer> sel;
  SHARED Flag* flag;
  flag = sel.construct();

  SHARED int calls_storage;
  int* calls = &calls_storage;
  execute_on_main_thread([&] {
    *calls = 0;
  });

  auto caller = LAMBDA()
  {
    cuda::std::call_once(*flag, count_call{calls});
    cuda::std::call_once(*flag, count_call{calls});
  };
  concurrent_agents_launch(caller, caller);

  execute_on_main_thread([&] {
    assert(*calls == 1);
    cuda::std::call_once(*flag, count_call{calls});
    assert(*calls == 1);
  });
}

int main(int, char**)
{
  NV_IF_TARGET(
    NV_IS_HOST,
    (cuda_thread_count = 2;

     test<cuda::std::once_flag, local_memory_selector>();
     test<cuda::once_flag<cuda::thread_scope_thread>, local_memory_selector>();
     test<cuda::once_flag<cuda::thread_scope_block>, local_memory_selector>();
     test<cuda::once_flag<cuda::thread_scope_device>, local_memory_selector>();
     test<cuda::once_flag<cuda::thread_scope_cluster>, local_memory_selector>();
     test<cuda::once_flag<cuda::thread_scope_system>, local_memory_selector>();))

  NV_IF_TARGET(
    NV_IS_DEVICE,
    (test<cuda::std::once_flag, shared_memory_selector>();
     test<cuda::once_flag<cuda::thread_scope_thread>, shared_memory_selector>();
     test<cuda::once_flag<cuda::thread_scope_block>, shared_memory_selector>();
     test<cuda::once_flag<cuda::thread_scope_device>, shared_memory_selector>();
     test<cuda::once_flag<cuda::thread_scope_system>, shared_memory_selector>();

     test<cuda::std::once_flag, global_memory_selector>();
     test<cuda::once_flag<cuda::thread_scope_thread>, global_memory_selector>();
     test<cuda::once_flag<cuda::thread_scope_block>, global_memory_selector>();
     test<cuda::once_flag<cuda::thread_scope_device>, global_memory_selector>();
     test<cuda::once_flag<cuda::thread_scope_system>, global_memory_selector>();))

  // Cluster atomics are not available below sm_90.
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (test<cuda::once_flag<cuda::thread_scope_cluster>, shared_memory_selector>();
                test<cuda::once_flag<cuda::thread_scope_cluster>, global_memory_selector>();))

  return 0;
}
