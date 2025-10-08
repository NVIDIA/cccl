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

/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

/**
 * Parameterizable tuning policy type for AgentFind
 * @tparam NominalBlockThreads4B Threads per thread block
 * @tparam NominalItemsPerThread4B Items per thread (per tile of input)
 * @tparam _VECTOR_LOAD_LENGTH Number of items per vectorized load
 * @tparam _LOAD_MODIFIER Cache load modifier for reading input elements
 */
template <int NominalBlockThreads4B,
          int NominalItemsPerThread4B,
          typename ComputeT,
          int VectorLoadLength,
          CacheLoadModifier _LOAD_MODIFIER,
          typename ScalingType = cub::detail::MemBoundScaling<NominalBlockThreads4B, NominalItemsPerThread4B, ComputeT>>
struct AgentFindPolicy : ScalingType
{
  /// Number of items per vectorized load
  static constexpr int VECTOR_LOAD_LENGTH = VectorLoadLength;

  /// Cache load modifier for reading input elements
  static constexpr CacheLoadModifier LOAD_MODIFIER = _LOAD_MODIFIER;
};

template <typename AgentFindPolicy,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ScanOpT> // @giannis OutputiteratorT not needed
struct AgentFind
{
  /// The input value type
  using InputT = typename ::cuda::std::iterator_traits<InputIteratorT>::value_type;

  /// Vector type of InputT for data movement
  using VectorT = typename CubVector<InputT, AgentFindPolicy::VECTOR_LOAD_LENGTH>::Type;

  /// Input iterator wrapper type (for applying cache modifier)
  // Wrap the native input pointer with CacheModifiedInputIterator
  // or directly use the supplied input iterator type
  using WrappedInputIteratorT =
    ::cuda::std::_If<::cuda::std::is_pointer<InputIteratorT>::value,
                     CacheModifiedInputIterator<AgentFindPolicy::LOAD_MODIFIER, InputT, OffsetT>,
                     InputIteratorT>;

  /// Constants
  static constexpr int BLOCK_THREADS      = AgentFindPolicy::BLOCK_THREADS;
  static constexpr int ITEMS_PER_THREAD   = AgentFindPolicy::ITEMS_PER_THREAD;
  static constexpr int TILE_ITEMS         = BLOCK_THREADS * ITEMS_PER_THREAD;
  static constexpr int VECTOR_LOAD_LENGTH = ::cuda::std::min(ITEMS_PER_THREAD, AgentFindPolicy::VECTOR_LOAD_LENGTH);

  // Can vectorize according to the policy if the input iterator is a native
  // pointer to a primitive type
  static constexpr bool ATTEMPT_VECTORIZATION =
    (VECTOR_LOAD_LENGTH > 1) && (ITEMS_PER_THREAD % VECTOR_LOAD_LENGTH == 0)
    && (::cuda::std::is_pointer<InputIteratorT>::value) && detail::is_primitive<InputT>::value;

  static constexpr CacheLoadModifier LOAD_MODIFIER = AgentFindPolicy::LOAD_MODIFIER;

  /// Shared memory type required by this thread block
  using _TempStorage = OffsetT;

  /// Alias wrapper allowing storage to be unioned
  using TempStorage = Uninitialized<_TempStorage>;

  _TempStorage& sresult; ///< Reference to temp_storage
  InputIteratorT d_in; ///< Input data to find
  // OutputIteratorT d_out;
  // OffsetT num_items;
  // OffsetT* value_temp_storage;
  // WrappedInputIteratorT d_wrapped_in; ///< Wrapped input data to find
  ScanOpT scan_op; ///< Binary reduction operator

  template <typename T>
  static _CCCL_DEVICE _CCCL_FORCEINLINE bool IsAlignedAndFullTile(
    T* d_in,
    int tile_offset,
    int tile_size,
    OffsetT num_items,
    ::cuda::std::integral_constant<bool, true> /*CAN_VECTORIZE*/)
  {
    /// Create an AgentFindIf and extract these two as type member in the encapsulating struct
    using InputT  = T;
    using VectorT = typename CubVector<InputT, VECTOR_LOAD_LENGTH>::Type;
    ///
    const bool full_tile  = (tile_offset + tile_size) <= num_items;
    const bool is_aligned = reinterpret_cast<::cuda::std::uintptr_t>(d_in) % uintptr_t{sizeof(VectorT)} == 0;
    return full_tile && is_aligned;
  }

  template <typename Iterator>
  static _CCCL_DEVICE _CCCL_FORCEINLINE bool IsAlignedAndFullTile(
    Iterator /*d_in*/,
    int /*tile_offset*/,
    int /*tile_size*/,
    std::size_t /*num_items*/,
    ::cuda::std::integral_constant<bool, false> /*CAN_VECTORIZE*/)
  {
    return false;
  }

  /**
   * @brief Constructor
   * @param sresult Reference to temp_storage
   * @param d_in Input data to search
   * @param scan_op Binary scan operator
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE AgentFind(TempStorage& sresult, InputIteratorT d_in, ScanOpT scan_op)
      : sresult(sresult.Alias())
      , d_in(d_in)
      , scan_op(scan_op)
  {}

  template <typename Pred>
  __device__ void ConsumeTile(
    int tile_offset,
    Pred pred,
    OffsetT* result,
    OffsetT num_items,
    ::cuda::std::integral_constant<bool, true> /*CAN_VECTORIZE*/)
  {
    using InputT  = typename ::cuda::std::iterator_traits<InputIteratorT>::value_type;
    using VectorT = typename CubVector<InputT, VECTOR_LOAD_LENGTH>::Type;

    __shared__ OffsetT block_result;

    if (threadIdx.x == 0)
    {
      block_result = num_items;
    }

    __syncthreads();

    constexpr int NUMBER_OF_VECTORS = ITEMS_PER_THREAD / VECTOR_LOAD_LENGTH;
    //// vectorized loads begin
    InputT* d_in_unqualified = const_cast<InputT*>(d_in) + tile_offset + (threadIdx.x * VECTOR_LOAD_LENGTH);

    cub::CacheModifiedInputIterator<AgentFindPolicy::LOAD_MODIFIER, VectorT> d_vec_in(
      reinterpret_cast<VectorT*>(d_in_unqualified));

    InputT input_items[ITEMS_PER_THREAD];
    VectorT* vec_items = reinterpret_cast<VectorT*>(input_items);

#pragma unroll
    for (int i = 0; i < NUMBER_OF_VECTORS; ++i)
    {
      vec_items[i] = d_vec_in[BLOCK_THREADS * i];
    }
    //// vectorized loads end

    bool found = false;
    for (int i = 0; i < ITEMS_PER_THREAD; ++i)
    {
      OffsetT nth_vector_of_thread = i / VECTOR_LOAD_LENGTH;
      OffsetT element_in_vector    = i % VECTOR_LOAD_LENGTH;
      OffsetT vector_of_tile       = nth_vector_of_thread * BLOCK_THREADS + threadIdx.x;

      OffsetT index = tile_offset + vector_of_tile * VECTOR_LOAD_LENGTH + element_in_vector;

      if (index < num_items)
      {
        if (pred(input_items[i]))
        {
          found = true;
          atomicMin(&block_result, index);
          break; // every thread goes over multiple elements per thread
                 // for every tile. If a thread finds a local minimum it doesn't
                 // need to proceed further (inner early exit).
        }
      }
    }

    if (syncthreads_or(found))
    {
      if (threadIdx.x == 0)
      {
        if (block_result < num_items)
        {
          atomicMin(result, block_result);
        }
      }
    }
  }

  template <typename Pred>
  __device__ void ConsumeTile(
    int tile_offset,
    Pred pred,
    OffsetT* result,
    OffsetT num_items,
    ::cuda::std::integral_constant<bool, false> /*CAN_VECTORIZE*/)
  {
    __shared__ int block_result;

    if (threadIdx.x == 0)
    {
      block_result = num_items;
    }

    __syncthreads();

    bool found = false;
    for (int i = 0; i < ITEMS_PER_THREAD; ++i)
    {
      auto index = tile_offset + threadIdx.x + i * blockDim.x;

      if (index < num_items)
      {
        if (pred(*(d_in + index)))
        {
          found = true;
          atomicMin(&block_result, index);
          break;
        }
      }
    }
    if (syncthreads_or(found))
    {
      if (threadIdx.x == 0)
      {
        if (block_result < num_items)
        {
          atomicMin(result, block_result);
        }
      }
    }
  }

  __device__ void Process(OffsetT* value_temp_storage, OffsetT num_items)
  {
    for (int tile_offset = blockIdx.x * TILE_ITEMS; tile_offset < num_items; tile_offset += TILE_ITEMS * gridDim.x)
    {
      // Only one thread reads atomically and propagates it to the
      // the other threads of the block through shared memory
      if (threadIdx.x == 0)
      {
        sresult = atomicAdd(value_temp_storage, 0);
      }
      __syncthreads();

      // early exit
      if (sresult < tile_offset)
      {
        return;
      }

      IsAlignedAndFullTile(
        d_in, tile_offset, TILE_ITEMS, num_items, ::cuda::std::integral_constant<bool, ATTEMPT_VECTORIZATION>())
        ? ConsumeTile(tile_offset,
                      scan_op,
                      value_temp_storage,
                      num_items,
                      ::cuda::std::integral_constant<bool, ATTEMPT_VECTORIZATION>())
        : ConsumeTile(
            tile_offset, scan_op, value_temp_storage, num_items, ::cuda::std::integral_constant<bool, false>());
    }
  }
};

CUB_NAMESPACE_END
