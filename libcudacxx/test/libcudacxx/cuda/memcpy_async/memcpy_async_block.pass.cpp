//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: pre-sm-70

// clang-cuda < 20 errors out with "fatal error: error in backend: Cannot cast between two non-generic address spaces"
// XFAIL: clang-14 && !nvcc
// XFAIL: clang-15 && !nvcc
// XFAIL: clang-16 && !nvcc
// XFAIL: clang-17 && !nvcc
// XFAIL: clang-18 && !nvcc
// XFAIL: clang-19 && !nvcc

#include <cuda/barrier>

#include "cuda_space_selector.h"

inline constexpr int thread_block_size = 64;

template <class T,
          template <typename, typename> class SourceSelector,
          template <typename, typename> class DestSelector,
          template <typename, typename> class BarrierSelector,
          cuda::thread_scope BarrierScope,
          typename... CompletionF>
__device__ __noinline__ void test_fully_specialized()
{
  // these tests focus on non-trivial thread ids and concurrent calls in the presence of other threads

  struct data_t
  {
    T data[thread_block_size];
  };

  SourceSelector<data_t, default_initializer> source_sel;
  typename DestSelector<data_t, default_initializer>::template offsetted<decltype(source_sel)::shared_offset> dest_sel;
  BarrierSelector<cuda::barrier<BarrierScope, CompletionF...>, constructor_initializer> bar_sel;

  data_t* source                                   = source_sel.construct();
  data_t* dest                                     = dest_sel.construct();
  cuda::barrier<BarrierScope, CompletionF...>* bar = bar_sel.construct(+thread_block_size);

  // init SMEM
  source->data[threadIdx.x] = 12;
  dest->data[threadIdx.x]   = 0;
  __barrier_sync(0);

  // single thread
  if (threadIdx.x == 0)
  {
    cuda::memcpy_async(dest->data, source->data, sizeof(T), *bar);
  }
  bar->arrive_and_wait();
  assert(dest->data[threadIdx.x] == (threadIdx.x == 0 ? 12 : 0)); // 12 0 0 0 0 0 ...

  // void* overload, single thread, different id, two calls with 3 and 2 items
  source->data[threadIdx.x] = 24;
  __barrier_sync(0);

  if (threadIdx.x == 42)
  {
    static_assert(42 < thread_block_size);
    cuda::memcpy_async(static_cast<void*>(dest->data), static_cast<void*>(source->data), sizeof(T) * 3, *bar);
    cuda::memcpy_async(static_cast<void*>(dest->data + 3), static_cast<void*>(source->data + 3), sizeof(T) * 2, *bar);
  }
  bar->arrive_and_wait();
  assert(dest->data[threadIdx.x] == (threadIdx.x < 5 ? 24 : 0)); // 24 24 24 24 24 0 0 0 ...

  // use 3 threads to perform 8 copies of 2 items each, spaced at 4 bytes in the destination
  source->data[threadIdx.x] = 48;
  __barrier_sync(0);

  if (threadIdx.x < 3)
  {
    static_assert(thread_block_size >= 64);
    cuda::memcpy_async(dest->data + threadIdx.x * 4, source->data + threadIdx.x * 2, sizeof(T) * 2, *bar);
  }
  bar->arrive_and_wait();

  assert(dest->data[0] == 48);
  assert(dest->data[1] == 48);
  assert(dest->data[2] == 24);
  assert(dest->data[3] == 24);
  assert(dest->data[4] == 48);
  assert(dest->data[5] == 48);
  assert(dest->data[6] == 0);
  assert(dest->data[7] == 0);
  assert(dest->data[8] == 48);
  assert(dest->data[9] == 48);
  if (threadIdx.x >= 10)
  {
    assert(dest->data[threadIdx.x] == 0);
  }
}

struct completion
{
  __device__ void operator()() const {}
};

template <class T,
          template <typename, typename> class SourceSelector,
          template <typename, typename> class DestSelector,
          template <typename, typename> class BarrierSelector>
__device__ __noinline__ void test_select_scope()
{
  test_fully_specialized<T, SourceSelector, DestSelector, BarrierSelector, cuda::thread_scope_system>();
  test_fully_specialized<T, SourceSelector, DestSelector, BarrierSelector, cuda::thread_scope_device>();
  test_fully_specialized<T, SourceSelector, DestSelector, BarrierSelector, cuda::thread_scope_block>();
  // Test one of the scopes with a non-default completion. Testing them all would make this test take twice as much time
  // to compile. Selected block scope because the block scope barrier with the default completion has a special path, so
  // this tests both that the API entrypoints accept barriers with arbitrary completion function, and that the
  // synchronization mechanism detects it correctly.
  test_fully_specialized<T, SourceSelector, DestSelector, BarrierSelector, cuda::thread_scope_block, completion>();
}

template <class T, template <typename, typename> class SourceSelector, template <typename, typename> class DestSelector>
__device__ __noinline__ void test_select_barrier()
{
  test_select_scope<T, SourceSelector, DestSelector, shared_memory_selector>();
  test_select_scope<T, SourceSelector, DestSelector, global_memory_selector>();
}

template <class T, template <typename, typename> class SourceSelector>
__device__ __noinline__ void test_select_destination()
{
  test_select_barrier<T, SourceSelector, shared_memory_selector>();
  test_select_barrier<T, SourceSelector, global_memory_selector>();
}

template <class T>
__device__ __noinline__ void test_select_source()
{
  test_select_destination<T, shared_memory_selector>();
  test_select_destination<T, global_memory_selector>();
}

int main(int argc, char** argv)
{
  // important: `cuda__thread__count =` (typo on purpose) needs to be followed by an integer literal, otherwise nvrtcc
  // cannot regex-match it
  NV_IF_TARGET(NV_IS_HOST, (cuda_thread_count = 64;), ({
                 assert(blockDim.x == thread_block_size);
                 test_select_source<int32_t>();
               }))

  return 0;
}
