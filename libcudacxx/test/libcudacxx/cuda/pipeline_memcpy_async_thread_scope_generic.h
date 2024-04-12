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
#include "large_type.h"
#include <cooperative_groups.h>

template <cuda::thread_scope scope, uint8_t PipelineStages>
__host__ __device__ cuda::pipeline<scope> get_pipeline(cuda::pipeline_shared_state<scope, PipelineStages>* pipe_state)
{
  NV_DISPATCH_TARGET(NV_IS_DEVICE,
                     (auto group = cooperative_groups::this_thread_block(); return make_pipeline(group, pipe_state);),
                     NV_IS_HOST,
                     (auto group = cuda::__single_thread_group{}; return make_pipeline(group, pipe_state);))
}

template <cuda::thread_scope Scope,
          class T,
          template <typename, typename>
          class SourceSelector,
          template <typename, typename>
          class DestSelector,
          template <typename, typename>
          class PipelineSelector,
          uint8_t PipelineStages>
__host__ __device__ __noinline__ void test_fully_specialized()
{
  SourceSelector<T, constructor_initializer> source_sel;
  typename DestSelector<T, constructor_initializer>::template offsetted<decltype(source_sel)::shared_offset> dest_sel;
  PipelineSelector<cuda::pipeline_shared_state<Scope, PipelineStages>, constructor_initializer> pipe_state_sel;

  T* source                                                      = source_sel.construct(static_cast<T>(12));
  T* dest                                                        = dest_sel.construct(static_cast<T>(0));
  cuda::pipeline_shared_state<Scope, PipelineStages>* pipe_state = pipe_state_sel.construct();

  auto pipe = get_pipeline(pipe_state);

  assert(*source == 12);
  assert(*dest == 0);

  pipe.producer_acquire();
  cuda::memcpy_async(dest, source, sizeof(T), pipe);
  pipe.producer_commit();
  pipe.consumer_wait();

  assert(*source == 12);
  assert(*dest == 12);

  pipe.consumer_release();

  *source = 24;

  pipe.producer_acquire();
  cuda::memcpy_async(static_cast<void*>(dest), static_cast<void*>(source), sizeof(T), pipe);
  pipe.producer_commit();
  pipe.consumer_wait_for(cuda::std::chrono::seconds(30));

  assert(*source == 24);
  assert(*dest == 24);

  pipe.consumer_release();

  *source = 42;

  pipe.producer_acquire();
  cuda::memcpy_async(static_cast<void*>(dest), static_cast<void*>(source), sizeof(T), pipe);
  pipe.producer_commit();
  pipe.consumer_wait_until(cuda::std::chrono::system_clock::now() + cuda::std::chrono::seconds(30));

  assert(*source == 42);
  assert(*dest == 42);

  pipe.consumer_release();
}

template <cuda::thread_scope Scope,
          class T,
          template <typename, typename>
          class SourceSelector,
          template <typename, typename>
          class DestSelector>
__host__ __device__ __noinline__ void test_select_pipeline()
{
  constexpr uint8_t stages_count = 2;
  test_fully_specialized<Scope, T, SourceSelector, DestSelector, local_memory_selector, stages_count>();
  NV_IF_TARGET(
    NV_IS_DEVICE,
    (test_fully_specialized<Scope, T, SourceSelector, DestSelector, shared_memory_selector, stages_count>();
     test_fully_specialized<Scope, T, SourceSelector, DestSelector, global_memory_selector, stages_count>();))
}

template <cuda::thread_scope Scope, class T, template <typename, typename> class SourceSelector>
__host__ __device__ __noinline__ void test_select_destination()
{
  test_select_pipeline<Scope, T, SourceSelector, local_memory_selector>();
  NV_IF_TARGET(NV_IS_DEVICE,
               (test_select_pipeline<Scope, T, SourceSelector, shared_memory_selector>();
                test_select_pipeline<Scope, T, SourceSelector, global_memory_selector>();))
}

template <cuda::thread_scope Scope, class T>
__host__ __device__ __noinline__ void test_select_source()
{
  test_select_destination<Scope, T, local_memory_selector>();
  NV_IF_TARGET(NV_IS_DEVICE,
               (test_select_destination<Scope, T, shared_memory_selector>();
                test_select_destination<Scope, T, global_memory_selector>();))
}
