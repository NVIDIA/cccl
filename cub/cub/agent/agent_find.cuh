// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once
#include <cub/config.cuh>

#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/thread/thread_load.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_type.cuh>

#include <thrust/detail/raw_reference_cast.h>
#include <thrust/type_traits/is_trivially_relocatable.h>

#include <cuda/__memory/is_aligned.h>
#include <cuda/atomic>
#include <cuda/std/__type_traits/integral_constant.h>

CUB_NAMESPACE_BEGIN
namespace detail::find
{
template <int NominalBlockThreads4B,
          int NominalItemsPerThread4B,
          typename ComputeT,
          int VectorLoadLength,
          CacheLoadModifier LoadModifier,
          typename ScalingType = MemBoundScaling<NominalBlockThreads4B, NominalItemsPerThread4B, ComputeT>>
struct agent_find_policy_t : ScalingType
{
  // Number of items per vectorized load
  static constexpr int vector_load_length = VectorLoadLength;

  // Cache load modifier for reading input elements
  static constexpr CacheLoadModifier load_modifier = LoadModifier;
};

template <typename AgentFindPolicy, typename InputIteratorT, typename OffsetT, typename PredicateT>
struct agent_t
{
  // The input value type
  using InputT = typename ::cuda::std::iterator_traits<InputIteratorT>::value_type;

  // Vector type of InputT for data movement
  using VectorT = typename CubVector<InputT, AgentFindPolicy::vector_load_length>::Type;

  static constexpr int block_threads      = AgentFindPolicy::BLOCK_THREADS;
  static constexpr int items_per_thread   = AgentFindPolicy::ITEMS_PER_THREAD;
  static constexpr int tile_items         = block_threads * items_per_thread;
  static constexpr int vector_load_length = AgentFindPolicy::vector_load_length;

  // Can vectorize according to the policy if the input iterator is a native pointer to a primitive type
  static constexpr bool attempt_vectorization =
    (vector_load_length > 1) && (items_per_thread % vector_load_length == 0)
    && (::cuda::std::contiguous_iterator<InputIteratorT>) && THRUST_NS_QUALIFIER::is_trivially_relocatable_v<InputT>;

  static constexpr CacheLoadModifier load_modifier = AgentFindPolicy::load_modifier;

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

  template <typename Iterator = InputIteratorT, bool CanVectorize = attempt_vectorization>
  _CCCL_DEVICE _CCCL_FORCEINLINE bool is_aligned_and_full_tile(OffsetT tile_offset)
  {
    if constexpr (CanVectorize)
    {
      static_assert(::cuda::std::is_pointer_v<Iterator>);

      // Retrieve the value type from the iterator to determine the vector type
      using InputT  = typename ::cuda::std::iterator_traits<Iterator>::value_type;
      using VectorT = typename CubVector<InputT, vector_load_length>::Type;

      const bool full_tile = (tile_offset + tile_items) <= num_items;

      // Check alignment at the actual load position (d_in + tile_offset)
      return full_tile && ::cuda::is_aligned(d_in + tile_offset, sizeof(VectorT));
    }
    else
    {
      return false;
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE bool
  ConsumeTile(OffsetT tile_offset, ::cuda::std::integral_constant<bool, true> /*CAN_VECTORIZE*/)
  {
    using InputT  = typename ::cuda::std::iterator_traits<InputIteratorT>::value_type;
    using VectorT = typename CubVector<InputT, vector_load_length>::Type;

    // vectorized loads begin
    auto load_ptr = reinterpret_cast<const VectorT*>(d_in + tile_offset + (threadIdx.x * vector_load_length));
    CacheModifiedInputIterator<AgentFindPolicy::load_modifier, VectorT> d_vec_in(load_ptr);

    InputT input_items[items_per_thread];
    auto* vec_items = reinterpret_cast<VectorT*>(input_items);

    constexpr int number_of_vectors = items_per_thread / vector_load_length;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < number_of_vectors; ++i)
    {
      vec_items[i] = d_vec_in[block_threads * i];
    }
    // vectorized loads end

    for (int i = 0; i < items_per_thread; ++i)
    {
      OffsetT nth_vector_of_thread = i / vector_load_length;
      OffsetT element_in_vector    = i % vector_load_length;
      OffsetT vector_of_tile       = nth_vector_of_thread * block_threads + threadIdx.x;

      OffsetT index = tile_offset + vector_of_tile * vector_load_length + element_in_vector;

      if (index < num_items && predicate(input_items[i]))
      {
        atomicMin(&temp_storage.block_result, index);
        // every thread goes over multiple elements per thread for every tile. If a thread finds a local minimum it
        // doesn't need to proceed further (inner early exit).
        return true;
      }
    }

    return false;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE bool
  ConsumeTile(OffsetT tile_offset, ::cuda::std::integral_constant<bool, false> /*CAN_VECTORIZE*/)
  {
    for (int i = 0; i < items_per_thread; ++i)
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
    OffsetT grid_stride = static_cast<OffsetT>(tile_items) * static_cast<OffsetT>(gridDim.x);
    for (OffsetT tile_offset = static_cast<OffsetT>(blockIdx.x) * static_cast<OffsetT>(tile_items);
         tile_offset < num_items;
         tile_offset += grid_stride)
    {
      // Only one thread reads atomically and propagates it to other threads of the block through shared memory
      if (threadIdx.x == 0)
      {
        // __nv_atomic_load is a compiler build-in and compiles a lot faster
#if _CCCL_CUDA_COMPILER(NVCC, >=, 12, 8)
        __nv_atomic_load(found_pos_ptr, &temp_storage.global_result, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
#else
        temp_storage.global_result = ::cuda::atomic_ref<OffsetT, ::cuda::std::thread_scope_device>{*found_pos_ptr}.load(
          ::cuda::std::memory_order_relaxed);
#endif
      }
      __syncthreads();

      // early exit
      if (temp_storage.global_result < tile_offset)
      {
        return;
      }

      const bool found_thread =
        is_aligned_and_full_tile(tile_offset)
          ? ConsumeTile(tile_offset, ::cuda::std::bool_constant<attempt_vectorization>{})
          : ConsumeTile(tile_offset, ::cuda::std::false_type{});

      const bool found_block = syncthreads_or(found_thread);
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
