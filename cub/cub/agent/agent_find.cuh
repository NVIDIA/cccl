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
#if !_CCCL_HAS_NV_ATOMIC_BUILTINS()
#  include <cuda/atomic>
#endif // !_CCCL_HAS_NV_ATOMIC_BUILTINS()
#include <cuda/std/__type_traits/integral_constant.h>

CUB_NAMESPACE_BEGIN
namespace detail::find
{
template <int BlockThreads,
          int ItemsPerThread,
          int VecSize,
          CacheLoadModifier LoadModifier,
          typename InputIteratorT,
          typename OffsetT,
          typename PredicateT>
struct agent_t
{
  // The input value type
  using InputT = typename ::cuda::std::iterator_traits<InputIteratorT>::value_type;

  // Vector type of InputT for data movement
  using VectorT = typename CubVector<InputT, VecSize>::Type;

  static constexpr int tile_size = BlockThreads * ItemsPerThread;

  // Can vectorize according to the policy if the input iterator is a native pointer to a primitive type
  static constexpr bool attempt_vectorization =
    (VecSize > 1) && (ItemsPerThread % VecSize == 0) && (::cuda::std::contiguous_iterator<InputIteratorT>)
    && THRUST_NS_QUALIFIER::is_trivially_relocatable_v<InputT>;

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

  template <typename Iterator = InputIteratorT, bool CanVectorize = attempt_vectorization>
  _CCCL_DEVICE _CCCL_FORCEINLINE bool is_aligned_and_full_tile(OffsetT tile_offset)
  {
    if constexpr (CanVectorize)
    {
      static_assert(::cuda::std::is_pointer_v<Iterator>);

      // Retrieve the value type from the iterator to determine the vector type
      using InputT  = typename ::cuda::std::iterator_traits<Iterator>::value_type;
      using VectorT = typename CubVector<InputT, VecSize>::Type;

      const bool full_tile = (tile_offset + tile_size) <= num_items;

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
    using VectorT = typename CubVector<InputT, VecSize>::Type;

    // vectorized loads begin
    auto load_ptr = reinterpret_cast<const VectorT*>(d_in + tile_offset + (threadIdx.x * VecSize));
    CacheModifiedInputIterator<LoadModifier, VectorT> d_vec_in(load_ptr);

    alignas(InputT) unsigned char input_bytes[ItemsPerThread * sizeof(InputT)];
    auto* vec_items = reinterpret_cast<VectorT*>(input_bytes);

    constexpr int number_of_vectors = ItemsPerThread / VecSize;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < number_of_vectors; ++i)
    {
      vec_items[i] = d_vec_in[BlockThreads * i];
    }

    for (int i = 0; i < ItemsPerThread; ++i)
    {
      OffsetT nth_vector_of_thread = i / VecSize;
      OffsetT element_in_vector    = i % VecSize;
      OffsetT vector_of_tile       = nth_vector_of_thread * BlockThreads + threadIdx.x;

      OffsetT index = tile_offset + vector_of_tile * VecSize + element_in_vector;

      auto* input_items = reinterpret_cast<InputT*>(input_bytes);
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

      const bool found_thread =
        is_aligned_and_full_tile(tile_offset)
          ? ConsumeTile(tile_offset, ::cuda::std::bool_constant<attempt_vectorization>{})
          : ConsumeTile(tile_offset, ::cuda::std::false_type{});

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
