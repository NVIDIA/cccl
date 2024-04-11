//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: pre-sm-70

#include <cuda/barrier>

#include "cuda_space_selector.h"

template <cuda::thread_scope Sco, template <typename, typename> class BarrierSelector>
__host__ __device__ void test()
{
  cuda::barrier<Sco> b(3);

  init(&b, 2);

  auto token = b.arrive();
  b.arrive_and_wait();
  b.wait(std::move(token));
}

template <cuda::thread_scope Sco>
__host__ __device__ void test_select_barrier()
{
  test<Sco, local_memory_selector>();
  NV_IF_TARGET(NV_IS_DEVICE, (test<Sco, shared_memory_selector>(); test<Sco, global_memory_selector>();))
}

int main(int argc, char** argv)
{
  test_select_barrier<cuda::thread_scope_system>();
  test_select_barrier<cuda::thread_scope_device>();
  test_select_barrier<cuda::thread_scope_block>();
  test_select_barrier<cuda::thread_scope_thread>();

  return 0;
}
