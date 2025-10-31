// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once
#include <cub/config.cuh>

#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/thread/thread_load.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/__type_traits/integral_constant.h>

CUB_NAMESPACE_BEGIN
namespace detail::find
{
template <int NominalBlockThreads4B,
          int NominalItemsPerThread4B,
          typename ComputeT,
          int VectorLoadLength,
          CacheLoadModifier LoadModifier,
          typename ScalingType = cub::detail::MemBoundScaling<NominalBlockThreads4B, NominalItemsPerThread4B, ComputeT>>
struct AgentFindPolicy : ScalingType
{
  // Number of items per vectorized load
  static constexpr int vector_load_length = VectorLoadLength;

  // Cache load modifier for reading input elements
  static constexpr CacheLoadModifier load_modifier = LoadModifier;
};

template <typename AgentFindPolicy, typename InputIteratorT, typename OutputIteratorT, typename OffsetT, typename ScanOpT>
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
    ::cuda::std::_If<::cuda::std::is_pointer_v<InputIteratorT>,
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
    && (::cuda::std::is_pointer<InputIteratorT>::value) && detail::is_primitive<InputT>::value;

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
  ScanOpT scan_op; // Binary reduction operator

  template <typename T>
  static _CCCL_DEVICE _CCCL_FORCEINLINE bool is_aligned_and_full_tile(
    T* d_in,
    int tile_offset,
    int tile_size,
    OffsetT num_items,
    ::cuda::std::integral_constant<bool, true> /*CAN_VECTORIZE*/)
  {
    // Create an agent_t and extract these two as type member in the encapsulating struct
    using InputT  = T;
    using VectorT = typename CubVector<InputT, vector_load_length>::Type;
    //
    const bool full_tile  = (tile_offset + tile_size) <= num_items;
    const bool is_aligned = reinterpret_cast<::cuda::std::uintptr_t>(d_in) % uintptr_t{sizeof(VectorT)} == 0;
    return full_tile && is_aligned;
  }

  template <typename Iterator>
  static _CCCL_DEVICE _CCCL_FORCEINLINE bool is_aligned_and_full_tile(
    Iterator /*d_in*/,
    int /*tile_offset*/,
    int /*tile_size*/,
    std::size_t /*num_items*/,
    ::cuda::std::integral_constant<bool, false> /*CAN_VECTORIZE*/)
  {
    return false;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE agent_t(TempStorage& temp_storage, InputIteratorT d_in, ScanOpT scan_op)
      : temp_storage(temp_storage.Alias())
      , d_in(d_in)
      , scan_op(scan_op)
  {}

  template <typename Pred>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ConsumeTile(
    int tile_offset,
    Pred pred,
    OffsetT* result,
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

    constexpr int number_of_vectors = items_per_thread / vector_load_length;
    // vectorized loads begin
    InputT* d_in_unqualified = const_cast<InputT*>(d_in) + tile_offset + (threadIdx.x * vector_load_length);

    cub::CacheModifiedInputIterator<AgentFindPolicy::load_modifier, VectorT> d_vec_in(
      reinterpret_cast<VectorT*>(d_in_unqualified));

    InputT input_items[items_per_thread];
    VectorT* vec_items = reinterpret_cast<VectorT*>(input_items);

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

      if (index < num_items && pred(input_items[i]))
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
        atomicMin(result, temp_storage.block_result);
      }
    }
  }

  template <typename Pred>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ConsumeTile(
    int tile_offset,
    Pred pred,
    OffsetT* result,
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
      auto index = tile_offset + threadIdx.x + i * blockDim.x;

      if (index < num_items)
      {
        if (pred(*(d_in + index)))
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
        atomicMin(result, temp_storage.block_result);
      }
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void Process(OffsetT* value_temp_storage, OffsetT num_items)
  {
    for (int tile_offset = blockIdx.x * tile_items; tile_offset < num_items; tile_offset += tile_items * gridDim.x)
    {
      // Only one thread reads atomically and propagates it to the
      // the other threads of the block through shared memory
      if (threadIdx.x == 0)
      {
        temp_storage.block_result = atomicAdd(value_temp_storage, 0);
      }
      __syncthreads();

      // early exit
      if (temp_storage.block_result < tile_offset)
      {
        return;
      }

      is_aligned_and_full_tile(
        d_in, tile_offset, tile_items, num_items, ::cuda::std::integral_constant<bool, attempt_vectorization>())
        ? ConsumeTile(tile_offset,
                      scan_op,
                      value_temp_storage,
                      num_items,
                      ::cuda::std::integral_constant<bool, attempt_vectorization>())
        : ConsumeTile(
            tile_offset, scan_op, value_temp_storage, num_items, ::cuda::std::integral_constant<bool, false>());
    }
  }
};
} // namespace detail::find

CUB_NAMESPACE_END
