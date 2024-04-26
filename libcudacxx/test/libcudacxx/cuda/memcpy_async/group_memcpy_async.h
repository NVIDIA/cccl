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
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

template <typename T>
struct storage
{
  // A prime to avoid accidental alignment of the size with smaller element types.
  constexpr static int size = 61;

  __host__ __device__ storage(T val = 0)
  {
    for (cuda::std::size_t i = 0; i < size; ++i)
    {
      data[i] = val + i;
    }
  }

  storage(const storage&)            = default;
  storage& operator=(const storage&) = default;

  __host__ __device__ friend bool operator==(const storage& lhs, const storage& rhs)
  {
    for (cuda::std::size_t i = 0; i < size; ++i)
    {
      if (lhs.data[i] != rhs.data[i])
      {
        return false;
      }
    }

    return true;
  }

  __host__ __device__ friend bool operator==(const storage& lhs, const T& rhs)
  {
    for (cuda::std::size_t i = 0; i < size; ++i)
    {
      if (lhs.data[i] != static_cast<T>(rhs + i))
      {
        return false;
      }
    }

    return true;
  }

  T data[size];
};

#if !defined(TEST_COMPILER_NVRTC) && (!defined(__GNUC__) || __GNUC__ >= 5 || defined(__clang__))
static_assert(std::is_trivially_copy_constructible<storage<int8_t>>::value, "");
static_assert(std::is_trivially_copy_constructible<storage<uint16_t>>::value, "");
static_assert(std::is_trivially_copy_constructible<storage<int32_t>>::value, "");
static_assert(std::is_trivially_copy_constructible<storage<uint64_t>>::value, "");
#endif

template <class T,
          template <typename, typename>
          class SourceSelector,
          template <typename, typename>
          class DestSelector,
          template <typename, typename>
          class BarrierSelector,
          cuda::thread_scope BarrierScope,
          typename... CompletionF>
__device__ __noinline__ void test_fully_specialized()
{
  SourceSelector<T, constructor_initializer> source_sel;
  typename DestSelector<T, constructor_initializer>::template offsetted<decltype(source_sel)::shared_offset> dest_sel;
  BarrierSelector<cuda::barrier<BarrierScope, CompletionF...>, constructor_initializer> bar_sel;

  __shared__ T* source;
  __shared__ T* dest;
  __shared__ cuda::barrier<BarrierScope, CompletionF...>* bar;

  source = source_sel.construct(static_cast<T>(12));
  dest   = dest_sel.construct(static_cast<T>(0));
  bar    = bar_sel.construct(4);

  assert(*source == 12);
  assert(*dest == 0);

  cuda::memcpy_async(cg::this_thread_block(), dest, source, sizeof(T), *bar);

  bar->arrive_and_wait();

  assert(*source == 12);
  assert(*dest == 12);

  if (cg::this_thread_block().thread_rank() == 0)
  {
    *source = 24;
  }
  cg::this_thread_block().sync();

  cuda::memcpy_async(cg::this_thread_block(), static_cast<void*>(dest), static_cast<void*>(source), sizeof(T), *bar);

  bar->arrive_and_wait();

  assert(*source == 24);
  assert(*dest == 24);
}

struct completion
{
  __host__ __device__ void operator()() const {}
};

template <class T,
          template <typename, typename>
          class SourceSelector,
          template <typename, typename>
          class DestSelector,
          template <typename, typename>
          class BarrierSelector>
__host__ __device__ __noinline__ void test_select_scope()
{
  test_fully_specialized<T, SourceSelector, DestSelector, BarrierSelector, cuda::thread_scope_system>();
  test_fully_specialized<T, SourceSelector, DestSelector, BarrierSelector, cuda::thread_scope_device>();
  test_fully_specialized<T, SourceSelector, DestSelector, BarrierSelector, cuda::thread_scope_block>();
  // Test one of the scopes with a non-default completion. Testing them all would make this test take twice as much time
  // to compile. Selected block scope because the block scope barrier with the default completion has a special path, so
  // this tests both that the API entrypoints accept barriers with arbitrary completion function, and that the
  // synchronization mechanism detects it correctly.
  test_fully_specialized<T, SourceSelector, DestSelector, BarrierSelector, cuda::thread_scope_block, completion>();
  test_fully_specialized<T, SourceSelector, DestSelector, BarrierSelector, cuda::thread_scope_thread>();
}

template <class T, template <typename, typename> class SourceSelector, template <typename, typename> class DestSelector>
__host__ __device__ __noinline__ void test_select_barrier()
{
  NV_IF_TARGET(NV_IS_DEVICE,
               (test_select_scope<T, SourceSelector, DestSelector, shared_memory_selector>();
                test_select_scope<T, SourceSelector, DestSelector, global_memory_selector>();))
}

template <class T, template <typename, typename> class SourceSelector>
__host__ __device__ __noinline__ void test_select_destination()
{
  NV_IF_TARGET(NV_IS_DEVICE,
               (test_select_barrier<T, SourceSelector, shared_memory_selector>();
                test_select_barrier<T, SourceSelector, global_memory_selector>();))
}

template <class T>
__host__ __device__ __noinline__ void test_select_source()
{
  NV_IF_TARGET(
    NV_IS_DEVICE,
    (test_select_destination<T, shared_memory_selector>(); test_select_destination<T, global_memory_selector>();))
}
