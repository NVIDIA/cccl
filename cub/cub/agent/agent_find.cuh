// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once
#include <cub/config.cuh>

#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/thread/thread_load.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_type.cuh>

#include <thrust/type_traits/is_trivially_relocatable.h>

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

template <typename AgentFindPolicy, typename InputIteratorT, typename OffsetT>
struct agent_t
{
  // The input value type
  using InputT = typename ::cuda::std::iterator_traits<InputIteratorT>::value_type;

  // Vector type of InputT for data movement
  using VectorT = typename CubVector<InputT, AgentFindPolicy::vector_load_length>::Type;

  // Input iterator wrapper type (for applying cache modifier)
  // Wrap the native input pointer with CacheModifiedInputIterator
  // or directly use the supplied input iterator type
  using WrappedInputIteratorT =
    ::cuda::std::_If<::cuda::std::contiguous_iterator<InputIteratorT>,
                     CacheModifiedInputIterator<AgentFindPolicy::load_modifier, InputT, OffsetT>,
                     InputIteratorT>;

  static constexpr int block_threads      = AgentFindPolicy::BLOCK_THREADS;
  static constexpr int items_per_thread   = AgentFindPolicy::ITEMS_PER_THREAD;
  static constexpr int tile_items         = block_threads * items_per_thread;
  static constexpr int vector_load_length = ::cuda::std::min(items_per_thread, AgentFindPolicy::vector_load_length);

  // Can vectorize according to the policy if the input iterator is a native
  // pointer to a primitive type
  static constexpr bool attempt_vectorization =
    (vector_load_length > 1) && (items_per_thread % vector_load_length == 0)
    && (::cuda::std::contiguous_iterator<InputIteratorT>) && THRUST_NS_QUALIFIER::is_trivially_relocatable_v<InputT>;

  static constexpr CacheLoadModifier load_modifier = AgentFindPolicy::load_modifier;

  // Shared memory type required by this thread block
  struct _TempStorage
  {
    OffsetT block_result;
  };

  // Alias wrapper allowing storage to be unioned
  using TempStorage = Uninitialized<_TempStorage>;

  _TempStorage& temp_storage;
  InputIteratorT d_in;

  template <typename Iterator, bool CanVectorize = attempt_vectorization>
  static _CCCL_DEVICE _CCCL_FORCEINLINE bool
  is_aligned_and_full_tile(Iterator d_in, int tile_offset, int tile_size, OffsetT num_items)
  {
    if constexpr (CanVectorize)
    {
      // Retrieve the value type from the iterator to determine the vector type
      using InputT  = typename ::cuda::std::iterator_traits<Iterator>::value_type;
      using VectorT = typename CubVector<InputT, vector_load_length>::Type;

      const bool full_tile = (tile_offset + tile_size) <= num_items;

      // Check alignment at the actual load position (d_in + tile_offset)
      // when CanVectorize is true, implying d_in is a raw pointer.
      const bool is_aligned =
        reinterpret_cast<::cuda::std::uintptr_t>(d_in + tile_offset) % uintptr_t{sizeof(VectorT)} == 0;

      return full_tile && is_aligned;
    }
    else
    {
      return false;
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE agent_t(TempStorage& temp_storage, InputIteratorT d_in)
      : temp_storage(temp_storage.Alias())
      , d_in(d_in)
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE bool ConsumeTile(
    OffsetT tile_offset,
    OffsetT* found_pos_ptr,
    OffsetT num_items,
    ::cuda::std::integral_constant<bool, true> /*CAN_VECTORIZE*/)
  {
    using InputT  = typename ::cuda::std::iterator_traits<InputIteratorT>::value_type;
    using VectorT = typename CubVector<InputT, vector_load_length>::Type;

    if (threadIdx.x == 0)
    {
      temp_storage.block_result = num_items;
    }

    __syncthreads();

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

    bool found = false;
    for (int i = 0; i < items_per_thread; ++i)
    {
      OffsetT nth_vector_of_thread = i / vector_load_length;
      OffsetT element_in_vector    = i % vector_load_length;
      OffsetT vector_of_tile       = nth_vector_of_thread * block_threads + threadIdx.x;

      OffsetT index = tile_offset + vector_of_tile * vector_load_length + element_in_vector;

      if (index < num_items && input_items[i])
      {
        found = true;
        atomicMin(&temp_storage.block_result, index);
        break; // every thread goes over multiple elements per thread
               // for every tile. If a thread finds a local minimum it doesn't
               // need to proceed further (inner early exit).
      }
    }

    if (syncthreads_or(found))
    {
      if (threadIdx.x == 0 && temp_storage.block_result < num_items)
      {
        atomicMin(found_pos_ptr, temp_storage.block_result);
      }
      return true;
    }
    return false;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE bool ConsumeTile(
    OffsetT tile_offset,
    OffsetT* found_pos_ptr,
    OffsetT num_items,
    ::cuda::std::integral_constant<bool, false> /*CAN_VECTORIZE*/)
  {
    if (threadIdx.x == 0)
    {
      temp_storage.block_result = num_items;
    }
    __syncthreads();

    bool found = false;
    for (int i = 0; i < items_per_thread; ++i)
    {
      const auto index = tile_offset + threadIdx.x + i * blockDim.x;
      if (index < num_items)
      {
        // Load into local variable to handle proxy types (e.g., device_reference)
        // and avoid issues with non-device-compatible destructors (e.g., tuple)
        InputT value = d_in[index];
        if (value)
        {
          found = true;
          atomicMin(&temp_storage.block_result, index);
          break;
        }
      }
    }
    if (syncthreads_or(found))
    {
      if (threadIdx.x == 0 && temp_storage.block_result < num_items)
      {
        atomicMin(found_pos_ptr, temp_storage.block_result);
      }
      return true;
    }
    return false;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void Process(OffsetT* found_pos_ptr, OffsetT num_items)
  {
    // use a grid strided loop
    OffsetT grid_stride = static_cast<OffsetT>(tile_items) * static_cast<OffsetT>(gridDim.x);
    for (OffsetT tile_offset = static_cast<OffsetT>(blockIdx.x) * static_cast<OffsetT>(tile_items);
         tile_offset < num_items;
         tile_offset += grid_stride)
    {
      // Only one thread reads atomically and propagates it to other threads of the block through shared memory
      if (threadIdx.x == 0)
      {
        temp_storage.block_result = atomicAdd(found_pos_ptr, 0); // TODO(bgruber): should be atomic load relaxed
      }
      __syncthreads();

      // early exit
      if (temp_storage.block_result < tile_offset)
      {
        return;
      }

      const bool found =
        is_aligned_and_full_tile(d_in, tile_offset, tile_items, num_items)
          ? ConsumeTile(tile_offset, found_pos_ptr, num_items, ::cuda::std::bool_constant<attempt_vectorization>{})
          : ConsumeTile(tile_offset, found_pos_ptr, num_items, ::cuda::std::false_type{});

      if (found)
      {
        return;
      }
    }
  }
};
} // namespace detail::find

CUB_NAMESPACE_END
