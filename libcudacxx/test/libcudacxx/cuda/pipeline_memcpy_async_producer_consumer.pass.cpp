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

#include <cuda/pipeline>

#include "cuda_space_selector.h"
#include <cooperative_groups.h>

template <class T, cuda::thread_scope PipelineScope>
__device__ __noinline__ void test_producer(T* dest, T* source, cuda::pipeline<PipelineScope>& pipe)
{
  pipe.producer_acquire();
  cuda::memcpy_async(static_cast<void*>(&dest[0]), static_cast<void*>(&source[0]), sizeof(T), pipe);
  pipe.producer_commit();

  pipe.producer_acquire();
  cuda::memcpy_async(static_cast<void*>(&dest[1]), static_cast<void*>(&source[1]), sizeof(T), pipe);
  pipe.producer_commit();

  pipe.producer_acquire();
  cuda::memcpy_async(static_cast<void*>(&dest[2]), static_cast<void*>(&source[2]), sizeof(T), pipe);
  pipe.producer_commit();
}

template <class T, cuda::thread_scope PipelineScope>
__device__ __noinline__ void test_consumer(T* dest, T* source, cuda::pipeline<PipelineScope>& pipe)
{
  pipe.consumer_wait();
  assert(source[0] == 12);
  assert(dest[0] == 12);
  pipe.consumer_release();

  pipe.consumer_wait();
  assert(source[1] == 24);
  assert(dest[1] == 24);
  pipe.consumer_release();

  pipe.consumer_wait();
  assert(source[2] == 36);
  assert(dest[2] == 36);
  pipe.consumer_release();
}

template <class T,
          template <typename, typename>
          class PipelineSelector,
          cuda::thread_scope PipelineScope,
          uint8_t PipelineStages>
__device__ __noinline__ void test_fully_specialized()
{
  __shared__ T dest[3];
  __shared__ T* source;

  PipelineSelector<cuda::pipeline_shared_state<PipelineScope, PipelineStages>, constructor_initializer> pipe_state_sel;
  cuda::pipeline_shared_state<PipelineScope, PipelineStages>* pipe_state = pipe_state_sel.construct();

  auto group = cooperative_groups::this_thread_block();

  if (group.thread_rank() == 0)
  {
    source = new T[3];
    for (size_t i = 0; i < 3; ++i)
    {
      source[i] = 12 * (i + 1);
    }
  }
  group.sync();

  {
    if (group.thread_rank() == 0)
    {
      memset(dest, 0, sizeof(T) * 3);
    }
    group.sync();

    const cuda::pipeline_role role =
      (group.thread_rank() % 2) ? cuda::pipeline_role::producer : cuda::pipeline_role::consumer;
    cuda::pipeline<PipelineScope> pipe = make_pipeline(group, pipe_state, role);

    if (role == cuda::pipeline_role::producer)
    {
      test_producer(dest, source, pipe);
    }
    else
    {
      test_consumer(dest, source, pipe);
    }

    pipe.quit();
  }

  group.sync();

  {
    if (group.thread_rank() == 0)
    {
      memset(dest, 0, sizeof(T) * 3);
    }
    group.sync();

    const cuda::pipeline_role role =
      (group.thread_rank() < 32) ? cuda::pipeline_role::producer : cuda::pipeline_role::consumer;
    cuda::pipeline<PipelineScope> pipe = make_pipeline(group, pipe_state, group.size() / 2);

    if (role == cuda::pipeline_role::producer)
    {
      test_producer(dest, source, pipe);
    }
    else
    {
      test_consumer(dest, source, pipe);
    }

    pipe.quit();
  }
}

template <class T, template <typename, typename> class PipelineSelector, cuda::thread_scope PipelineScope>
__host__ __device__ __noinline__ void test_select_stages()
{
  test_fully_specialized<T, PipelineSelector, PipelineScope, 1>();
  test_fully_specialized<T, PipelineSelector, PipelineScope, 8>();
}

template <class T, template <typename, typename> class PipelineSelector>
__host__ __device__ __noinline__ void test_select_scope()
{
  test_select_stages<T, PipelineSelector, cuda::thread_scope_block>();
  test_select_stages<T, PipelineSelector, cuda::thread_scope_device>();
  test_select_stages<T, PipelineSelector, cuda::thread_scope_system>();
}

template <class T>
__host__ __device__ __noinline__ void test_select_pipeline()
{
  test_select_scope<T, shared_memory_selector>();
  test_select_scope<T, global_memory_selector>();
}

int main(int argc, char** argv)
{
  NV_IF_ELSE_TARGET(
    NV_IS_HOST, cuda_thread_count = 64;
    ,
    (test_select_pipeline<uint8_t>(); test_select_pipeline<uint16_t>(); test_select_pipeline<uint32_t>();
     test_select_pipeline<uint64_t>();))

  return 0;
}
