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
#include <cuda/std/type_traits>

#include <cooperative_groups.h>

#include "cuda_space_selector.h"

namespace cg = cooperative_groups;

inline constexpr int thread_block_size = 64;

template <typename T>
struct storage
{
  // A prime to avoid accidental alignment of the size with smaller element types.
  constexpr static int size = 67;
  static_assert(size >= thread_block_size);

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

#if !TEST_COMPILER(NVRTC) && !TEST_COMPILER(CLANG)
static_assert(cuda::std::is_trivially_copy_constructible_v<storage<int8_t>>, "");
static_assert(cuda::std::is_trivially_copy_constructible_v<storage<uint16_t>>, "");
static_assert(cuda::std::is_trivially_copy_constructible_v<storage<int32_t>>, "");
static_assert(cuda::std::is_trivially_copy_constructible_v<storage<uint64_t>>, "");
#endif

template <class T,
          template <typename, typename> class SourceSelector,
          template <typename, typename> class DestSelector,
          template <typename, typename> class BarrierSelector,
          cuda::thread_scope BarrierScope,
          typename... CompletionF>
__device__ __noinline__ void test_fully_specialized()
{
  SourceSelector<storage<T>, constructor_initializer> source_sel;
  typename DestSelector<storage<T>, constructor_initializer>::template offsetted<decltype(source_sel)::shared_offset>
    dest_sel;
  BarrierSelector<cuda::barrier<BarrierScope, CompletionF...>, constructor_initializer> bar_sel;

  __shared__ storage<T>* source;
  __shared__ storage<T>* dest;
  __shared__ cuda::barrier<BarrierScope, CompletionF...>* bar;

  source = source_sel.construct(static_cast<storage<T>>(12));
  dest   = dest_sel.construct(static_cast<storage<T>>(0));
  bar    = bar_sel.construct(+thread_block_size);
  assert(*source == 12);
  assert(*dest == 0);

  // test normal version
  cuda::memcpy_async(cg::this_thread_block(), dest, source, sizeof(storage<T>), *bar);
  bar->arrive_and_wait();
  assert(*dest == 12);

  // prepare source
  if (cg::this_thread_block().thread_rank() == 0)
  {
    *source = 24;
  }
  cg::this_thread_block().sync();
  assert(*source == 24);

  // test void* overload and use just warp 1 for copy
  auto warps = cg::tiled_partition<32>(cg::this_thread_block());
  if (warps.meta_group_rank() == 1)
  {
    assert(threadIdx.x >= 32 && threadIdx.x < 64);
    static_assert(thread_block_size >= 64);
    cuda::memcpy_async(warps, static_cast<void*>(dest), static_cast<void*>(source), sizeof(storage<T>), *bar);
  }
  bar->arrive_and_wait();
  assert(*dest == 24);

  // prepare source
  if (cg::this_thread_block().thread_rank() == 0)
  {
    *source = 48;
  }
  cg::this_thread_block().sync();
  assert(*source == 48);

  // use 2 groups of 4 threads to copy 8 items each, but spread them 16 bytes
  auto tiled_groups = cg::tiled_partition<4>(cg::this_thread_block());
  if (threadIdx.x < 8)
  {
    static_assert(thread_block_size >= 8);
    printf("%u copying 8 items at meta group rank %u\n", threadIdx.x, tiled_groups.meta_group_rank());
    cuda::memcpy_async(
      tiled_groups,
      &dest->data[tiled_groups.meta_group_rank() * 16],
      &source->data[tiled_groups.meta_group_rank() * 16],
      sizeof(T) * 8,
      *bar);
  }
  bar->arrive_and_wait();

  for (int i = 0; i < 8; ++i)
  {
    assert(dest->data[i + 0] == static_cast<T>(48 + (i + 0))); // 8 copied items from first group
    assert(dest->data[i + 8] == static_cast<T>(24 + (i + 8))); // 8 untouched items between
    assert(dest->data[i + 16] == static_cast<T>(48 + (i + 16))); // 8 copied items from second group
  }
  for (int i = 24; i < storage<T>::size; ++i)
  {
    assert(dest->data[i] == static_cast<T>(24 + i)); // untouched items afterwards
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
  test_fully_specialized<T, SourceSelector, DestSelector, BarrierSelector, cuda::thread_scope_thread>();
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
__host__ __device__ __noinline__ void test_select_source()
{
  NV_IF_TARGET(
    NV_IS_DEVICE,
    (test_select_destination<T, shared_memory_selector>(); test_select_destination<T, global_memory_selector>();))
}
