// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once
#include <cub/config.cuh>

#include <cub/block/block_load_to_shared.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/thread/thread_load.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_type.cuh>

#include <thrust/detail/raw_reference_cast.h>
#include <thrust/type_traits/is_contiguous_iterator.h>
#include <thrust/type_traits/is_trivially_relocatable.h>

#include <cuda/__memory/is_aligned.h>
#include <cuda/atomic>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/optional>

CUB_NAMESPACE_BEGIN
namespace detail::find
{
template <int BlockThreads,
          int ItemsPerThread,
          bool AttemptBlockLoadToShared,
          int VectorLoadLength,
          CacheLoadModifier LoadModifier,
          typename InputIteratorT,
          typename OffsetT,
          typename PredicateT>
struct agent_t
{
  // The input value type
  using InputT = typename ::cuda::std::iterator_traits<InputIteratorT>::value_type;

  // Vector type of InputT for data movement
  using VectorT = typename CubVector<InputT, VectorLoadLength>::Type;

  static constexpr int tile_size = BlockThreads * ItemsPerThread;

  static constexpr bool use_bl2sh =
    AttemptBlockLoadToShared && (sizeof(InputT) == alignof(InputT))
    && THRUST_NS_QUALIFIER::is_trivially_relocatable_v<InputT>
    && THRUST_NS_QUALIFIER::is_contiguous_iterator_v<InputIteratorT>;
  static constexpr int input_buffer_align = cub::detail::LoadToSharedBufferAlignBytes<InputT>();
  static constexpr int input_buffer_size =
    cub::detail::LoadToSharedBufferSizeBytes<InputT>(ItemsPerThread * BlockThreads);
  using block_load_to_shared = cub::detail::BlockLoadToShared<BlockThreads>;

  // Can vectorize according to the policy if the input iterator is a native pointer to a primitive type
  static constexpr bool attempt_vectorization =
    (VectorLoadLength > 1) && (ItemsPerThread % VectorLoadLength == 0)
    && (THRUST_NS_QUALIFIER::is_contiguous_iterator_v<InputIteratorT>)
    && THRUST_NS_QUALIFIER::is_trivially_relocatable_v<InputT>;

  static constexpr CacheLoadModifier load_modifier = LoadModifier;
  struct alignas(input_buffer_align) input_buffer_t
  {
    char c_array[input_buffer_size];
  };

  // Shared memory type required by this thread block
  struct temp_storage_without_bl2sh
  {
    OffsetT global_result;
    OffsetT block_result;
  };

  struct temp_storage_with_bl2sh : temp_storage_without_bl2sh
  {
    typename block_load_to_shared::TempStorage load2sh;
    input_buffer_t input_buffer;
  };

  using _TempStorage = ::cuda::std::conditional_t<use_bl2sh, temp_storage_with_bl2sh, temp_storage_without_bl2sh>;

  // Alias wrapper allowing storage to be unioned
  using TempStorage = Uninitialized<_TempStorage>;

  _TempStorage& temp_storage;
  InputIteratorT d_in;
  PredicateT predicate;
  OffsetT* found_pos_ptr;
  OffsetT num_items;

  template <typename Iterator, bool CanVectorize = attempt_vectorization>
  _CCCL_DEVICE _CCCL_FORCEINLINE bool is_aligned_and_full_tile(Iterator in_iter, OffsetT tile_offset)
  {
    if constexpr (CanVectorize)
    {
      static_assert(THRUST_NS_QUALIFIER::is_contiguous_iterator_v<Iterator>);

      const bool full_tile = (tile_offset + tile_size) <= num_items;

      // Check alignment at the actual load position (in_iter + tile_offset)
      return full_tile
          && ::cuda::is_aligned(
               THRUST_NS_QUALIFIER::unwrap_contiguous_iterator(in_iter + (use_bl2sh ? 0 : tile_offset)),
               alignof(VectorT));
    }
    else
    {
      return false;
    }
  }

  template <class Iterator>
  _CCCL_DEVICE _CCCL_FORCEINLINE bool
  ConsumeTile(Iterator in_iter, OffsetT tile_offset, ::cuda::std::integral_constant<bool, true> /*CAN_VECTORIZE*/)
  {
    // vectorized loads begin
    const auto local_thread_offset = static_cast<int>(threadIdx.x) * VectorLoadLength;
    const auto thread_offset       = tile_offset + static_cast<OffsetT>(local_thread_offset);
    const auto load_ptr            = reinterpret_cast<const VectorT*>(
      THRUST_NS_QUALIFIER::unwrap_contiguous_iterator(in_iter + (use_bl2sh ? local_thread_offset : thread_offset)));
    using modified_iterator_t = typename ::cuda::std::
      conditional_t<use_bl2sh, decltype(load_ptr), CacheModifiedInputIterator<LoadModifier, VectorT>>;
    const auto d_vec_in = modified_iterator_t{load_ptr};

    alignas(VectorT) unsigned char input_bytes[ItemsPerThread * sizeof(InputT)];
    auto* vec_items = reinterpret_cast<VectorT*>(input_bytes);

    constexpr int number_of_vectors = ItemsPerThread / VectorLoadLength;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < number_of_vectors; ++i)
    {
      vec_items[i] = d_vec_in[BlockThreads * i];
    }

    for (int i = 0; i < ItemsPerThread; ++i)
    {
      OffsetT nth_vector_of_thread = i / VectorLoadLength;
      OffsetT element_in_vector    = i % VectorLoadLength;
      OffsetT vector_of_tile       = nth_vector_of_thread * BlockThreads + threadIdx.x;

      OffsetT index = tile_offset + vector_of_tile * VectorLoadLength + element_in_vector;

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

  template <class Iterator>
  _CCCL_DEVICE _CCCL_FORCEINLINE bool
  ConsumeTile(Iterator in_iter, OffsetT tile_offset, ::cuda::std::integral_constant<bool, false> /*CAN_VECTORIZE*/)
  {
    for (int i = 0; i < ItemsPerThread; ++i)
    {
      const int local_index = static_cast<int>(threadIdx.x) + i * BlockThreads;
      const OffsetT index   = tile_offset + static_cast<OffsetT>(local_index);
      if (index < num_items)
      {
        // using raw_reference_cast and passing directly to predicate should avoid creating a copy, and thus prevent
        // bugs like: http://github.com/NVIDIA/cccl/issues/3591
        if (predicate(THRUST_NS_QUALIFIER::raw_reference_cast(in_iter[use_bl2sh ? local_index : index])))
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
    [[maybe_unused]] auto load2sh = [&] {
      if constexpr (use_bl2sh)
      {
        return block_load_to_shared{temp_storage.load2sh};
      }
      else
      {
        return NullType{};
      }
    }();
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
      using token_type = typename block_load_to_shared::CommitToken;
      [[maybe_unused]] ::cuda::std::optional<token_type> token{};
      [[maybe_unused]] InputT* s_in{};
      if constexpr (use_bl2sh)
      {
        auto buffer                 = ::cuda::std::span{temp_storage.input_buffer.c_array};
        const auto shared_num_items = ::cuda::std::min(static_cast<OffsetT>(tile_size), num_items - tile_offset);
        auto src = ::cuda::std::span{THRUST_NS_QUALIFIER::unwrap_contiguous_iterator(d_in + tile_offset),
                                     static_cast<::cuda::std::size_t>(shared_num_items)};
        src      = load2sh.CopyAsync(buffer, src);
        token    = load2sh.Commit();
        s_in     = src.data();
      }
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
      auto consume_tile = [&](auto iter) {
        return is_aligned_and_full_tile(iter, tile_offset)
               ? ConsumeTile(iter, tile_offset, ::cuda::std::bool_constant<attempt_vectorization>{})
               : ConsumeTile(iter, tile_offset, ::cuda::std::false_type{});
      };
      auto found_thread = false;
      if constexpr (use_bl2sh)
      {
        load2sh.Wait(::cuda::std::move(token).value());
        // TODO: As we always have a full tile of smem, we could still vectorize even when we have a partial tile.
        //       The vectorized ConsumeTile() does only apply the predicate for in-bounds items.
        found_thread = consume_tile(s_in);
      }
      else
      {
        found_thread = consume_tile(d_in);
      }

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
