// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once
#include <cub/config.cuh>

#include <cub/block/block_load.cuh>
#include <cub/detail/uninitialized.cuh>
#include <cub/thread/thread_load.cuh>
#include <cub/util_arch.cuh>

#include <thrust/detail/raw_reference_cast.h>
#include <thrust/type_traits/is_contiguous_iterator.h>

#include <cuda/__iterator/transform_iterator.h>
#include <cuda/std/__type_traits/integral_constant.h>

#if !_CCCL_HAS_NV_ATOMIC_BUILTINS()
#  include <cuda/atomic>
#endif // !_CCCL_HAS_NV_ATOMIC_BUILTINS()

CUB_NAMESPACE_BEGIN
namespace detail::find
{
template <int ThreadsPerBlock,
          int ItemsPerThread,
          int VecSize,
          CacheLoadModifier LoadModifier,
          typename InputIteratorT,
          typename OffsetT,
          typename PredicateT>
struct agent_t
{
  static constexpr int tile_size = ThreadsPerBlock * ItemsPerThread;

  // Only contiguous inputs use the arrangement of vectors, because item-wise loads of other inputs in that arrangement
  // are slower than coalesced ones
  static constexpr bool attempt_vectorization =
    (VecSize > 1) && (ItemsPerThread % VecSize == 0) && THRUST_NS_QUALIFIER::is_contiguous_iterator_v<InputIteratorT>;

  static constexpr CacheLoadModifier load_modifier = LoadModifier;

  // Shared memory type required by this thread block
  struct _TempStorage
  {
    OffsetT global_result;
    OffsetT block_result;
  };

  // Alias wrapper allowing storage to be unioned
  using TempStorage = Uninitialized<_TempStorage>;

  _TempStorage& temp_storage;
  InputIteratorT d_in;
  PredicateT predicate;
  OffsetT* found_pos_ptr;
  OffsetT num_items;

  _CCCL_DEVICE _CCCL_FORCEINLINE bool ConsumeTile(OffsetT tile_offset, ::cuda::std::true_type /*CAN_VECTORIZE*/)
  {
    // Only full tiles are loaded with vectors. Checking this here instead of in Process() avoids a duplicated
    // item-wise loop when vectorization is not attempted
    if (tile_offset + tile_size > num_items)
    {
      return ConsumeTile(tile_offset, ::cuda::std::false_type{});
    }

    bool predicates[ItemsPerThread];
    LoadDirectStripedVectorized<ThreadsPerBlock, VecSize, LoadModifier>(
      static_cast<int>(threadIdx.x), ::cuda::transform_iterator{d_in + tile_offset, predicate}, predicates);

    for (int i = 0; i < ItemsPerThread; ++i)
    {
      OffsetT nth_vector_of_thread = i / VecSize;
      OffsetT element_in_vector    = i % VecSize;
      OffsetT vector_of_tile       = nth_vector_of_thread * ThreadsPerBlock + threadIdx.x;

      OffsetT index = tile_offset + vector_of_tile * VecSize + element_in_vector;

      if (index < num_items && predicates[i])
      {
        atomicMin(&temp_storage.block_result, index);
        // every thread goes over multiple elements per thread for every tile. If a thread finds a local minimum it
        // doesn't need to proceed further (inner early exit).
        return true;
      }
    }

    return false;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE bool ConsumeTile(OffsetT tile_offset, ::cuda::std::false_type /*CAN_VECTORIZE*/)
  {
    for (int i = 0; i < ItemsPerThread; ++i)
    {
      const auto index = tile_offset + threadIdx.x + i * blockDim.x;
      if (index < num_items)
      {
        // using raw_reference_cast and passing directly to predicate should avoid creating a copy, and thus prevent
        // bugs like: http://github.com/NVIDIA/cccl/issues/3591
        if (predicate(THRUST_NS_QUALIFIER::raw_reference_cast(d_in[index])))
        {
          atomicMin(&temp_storage.block_result, index);
          return true;
        }
      }
    }

    return false;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void Process()
  {
    if (threadIdx.x == 0)
    {
      temp_storage.block_result = num_items;
    }
    __syncthreads();

    // use a grid strided loop
    OffsetT grid_stride = static_cast<OffsetT>(tile_size) * static_cast<OffsetT>(gridDim.x);
    for (OffsetT tile_offset = static_cast<OffsetT>(blockIdx.x) * static_cast<OffsetT>(tile_size);
         tile_offset < num_items;
         tile_offset += grid_stride)
    {
      // Only one thread reads atomically and propagates it to other threads of the block through shared memory
      if (threadIdx.x == 0)
      {
#if _CCCL_HAS_NV_ATOMIC_BUILTINS()
        // __nv_atomic_load is a compiler build-in and compiles a lot faster
        __nv_atomic_load(found_pos_ptr, &temp_storage.global_result, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
#else // ^^^ _CCCL_HAS_NV_ATOMIC_BUILTINS() ^^^ / vvv !_CCCL_HAS_NV_ATOMIC_BUILTINS() vvv
        temp_storage.global_result = ::cuda::atomic_ref<OffsetT, ::cuda::std::thread_scope_device>{*found_pos_ptr}.load(
          ::cuda::std::memory_order_relaxed);
#endif // !_CCCL_HAS_NV_ATOMIC_BUILTINS()
      }
      __syncthreads();

      // early exit
      if (temp_storage.global_result < tile_offset)
      {
        return;
      }

      const bool found_thread = ConsumeTile(tile_offset, ::cuda::std::bool_constant<attempt_vectorization>{});

      const bool found_block = __syncthreads_or(found_thread);
      if (found_block)
      {
        // our block found it, update global position and exit
        if (threadIdx.x == 0 && temp_storage.block_result < num_items)
        {
          atomicMin(found_pos_ptr, temp_storage.block_result);
        }
        return;
      }
    }
  }
};
} // namespace detail::find

CUB_NAMESPACE_END
