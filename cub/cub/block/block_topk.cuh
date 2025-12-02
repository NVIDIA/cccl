// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! The @c cub::BlockTopK class provides a :ref:`collective <collective-primitives>` method for selecting the top-k
//! elements from a set of items within a CUDA thread block.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/__execution/env.h>

CUB_NAMESPACE_BEGIN

namespace detail
{



enum class BlockTopKAlgorithm
{
  air_top_k
};

#if false
template <typename T, select SelectDirection>
struct twiddle_keys_in_op_t
{
  using sort_key_t = typename Traits<T>::UnsignedBits;

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE sort_key_t operator()(T key) const
  {
    auto sort_key = reinterpret_cast<typename Traits<T>::UnsignedBits&>(key);
    sort_key      = Traits<T>::TwiddleIn(sort_key);
    if constexpr (SelectDirection != select::min)
    {
      sort_key = ~sort_key;
    }
    return sort_key;
  }
};

template <typename T, select SelectDirection>
struct twiddle_keys_out_op_t
{
  using sort_key_t = typename Traits<T>::UnsignedBits;

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE T operator()(sort_key_t sort_key) const
  {
    if constexpr (SelectDirection != select::min)
    {
      sort_key = ~sort_key;
    }
    sort_key = Traits<T>::TwiddleOut(sort_key);
    return reinterpret_cast<T&>(sort_key);
  }
};

template <typename T, int BlockThreads, int RadixBits = 11>
class NonDeterministicAirTopK
{
_CCCL_DEVICE _CCCL_FORCEINLINE void process_range(InputItT in, const OffsetT num_items, FuncT f)
  {
    key_in_t thread_data[items_per_thread];

    const OffsetT items_per_pass   = tile_items * gridDim.x;
    const OffsetT total_num_blocks = ::cuda::ceil_div(num_items, tile_items);

    const OffsetT num_remaining_elements = num_items % tile_items;
    const OffsetT last_block_id          = (total_num_blocks - 1) % gridDim.x;

    OffsetT tile_base = blockIdx.x * tile_items;
    OffsetT offset    = threadIdx.x * items_per_thread + tile_base;

    for (int i_block = blockIdx.x; i_block < total_num_blocks - 1; i_block += gridDim.x)
    {
      // Ensure that the temporary storage from previous iteration can be reused
      __syncthreads();

      block_load_input_t(temp_storage.load_input).Load(in + tile_base, thread_data);
      for (int j = 0; j < items_per_thread; ++j)
      {
        f(thread_data[j], offset + j);
      }
      tile_base += items_per_pass;
      offset += items_per_pass;
    }

    // Last tile specialized code-path
    if (blockIdx.x == last_block_id)
    {
      // Ensure that the temporary storage from the previous loop can be reused
      __syncthreads();

      if (num_remaining_elements == 0)
      {
        block_load_input_t(temp_storage.load_input).Load(in + tile_base, thread_data);
      }
      else
      {
        block_load_input_t(temp_storage.load_input).Load(in + tile_base, thread_data, num_remaining_elements);
      }

      for (int j = 0; j < items_per_thread; ++j)
      {
        if ((offset + j) < num_items)
        {
          f(thread_data[j], offset + j);
        }
      }
    }
  }
}

// TODO (elstehle): Add documentation
template <typename T, int BlockThreads, int RadixBits = 11>
class NonDeterministicAirTopK
{
private:
  static constexpr int block_threads = BlockThreads;
  static constexpr int num_buckets   = (1 << RadixBits);
  static constexpr int buckets_per_thread = ::cuda::ceil_div(num_buckets, block_threads);

  using histo_counter_t = ::cuda::std::uint32_t;
  using block_scan_t    = BlockScan<histo_counter_t, block_threads, BLOCK_SCAN_WARP_SCANS>;

  union _TempStorage
  {
    /// Histogram storage
    histo_counter_t histogram[num_buckets];
    block_scan_t::TempStorage scan_temp_storage;
  }

  /// Shared storage reference
  _TempStorage& temp_storage;

  /// Linear thread-id
  ::cuda::std::uint32_t linear_tid;

public:
  /// @smemstorage{BlockTopK}
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  _CCCL_DEVICE _CCCL_FORCEINLINE NonDeterministicAirTopK(TempStorage& temp_storage, ::cuda::std::uint32_t linear_tid)
      : temp_storage(temp_storage.Alias())
      , linear_tid(linear_tid)
  {}

  // Initialize histogram bins to zero
  _CCCL_DEVICE _CCCL_FORCEINLINE void init_histograms()
  {
    // Initialize histogram bin counts to zeros
    ::cuda::std::int32_t histo_offset = 0;

    // Loop unrolling is beneficial for performance here
    _CCCL_PRAGMA_UNROLL_FULL()
    for (; histo_offset + block_threads <= num_buckets; histo_offset += block_threads)
    {
      temp_storage.histogram[histo_offset + threadIdx.x] = 0;
    }
    // Finish up with guarded initialization if necessary
    if ((num_buckets % block_threads != 0) && (histo_offset + threadIdx.x < num_buckets))
    {
      temp_storage.histogram[histo_offset + threadIdx.x] = 0;
    }
  }

  // Compute prefix sum over buckets
  _CCCL_DEVICE _CCCL_FORCEINLINE void compute_bin_offsets()
  {
    histo_counter_t thread_buckets[buckets_per_thread];
    
    // Load histogram counts into thread-local storage
    block_load_trans_t(temp_storage.load_trans).Load(temp_storage.histogram, thread_buckets, num_buckets);
    
    // Compute the prefix sum over the buckets
    block_scan_t(temp_storage.scan_temp_storage).InclusiveSum(thread_buckets, thread_buckets);
    
    // Store the updated bucket counts back to shared memory
    block_store_trans_t(temp_storage.store_trans).Store(temp_storage.histogram, thread_buckets, num_buckets);
  }

  // Compute prefix sum over buckets
  _CCCL_DEVICE _CCCL_FORCEINLINE void choose_bucket()
  {
    auto body = [&](::cuda::std::int32_t histo_offset) {
      const int bin_idx  = histo_offset + threadIdx.x;
      const histo_counter_t prev = (bin_idx == 0) ? 0 : temp_storage.histogram[bin_idx - 1];
      const histo_counter_t cur  = temp_storage.histogram[bin_idx];

      // Identify the bin that the k-th item falls into. One and only one thread will satisfy this condition, so counter
      // is written by only one thread
      if (prev < k && cur >= k)
      {
        // The number of items that are yet to be identified
        counter->k = k - prev;

        // The number of candidates in the next pass
        counter->len                                   = cur - prev;
        typename Traits<key_in_t>::UnsignedBits bucket = bin_idx;
        // Update the "splitter" key by adding the radix digit of the k-th item bin of this pass
        const int start_bit = calc_start_bit<key_in_t, bits_per_pass>(pass);
        counter->kth_key_bits |= bucket << start_bit;
      }
    };

    
    ::cuda::std::int32_t histo_offset = 0;
    
    // Loop unrolling is beneficial for performance here
    _CCCL_PRAGMA_UNROLL_FULL()
    for (; histo_offset + block_threads <= num_buckets; histo_offset += block_threads)
    {
      body(histo_offset);
    }
    // Finish up with guarded initialization if necessary
    if ((num_buckets % block_threads != 0) && (histo_offset + threadIdx.x < num_buckets))
    {
      body(histo_offset);
    }
  }

  template <typename OnChipStorageT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void process_pass(const OnChipStorageT& key_storage)
  {
    // Zero-initialize histograms
    init_histograms();

    // Ensure all threads have completed histogram initialization
    __syncthreads();

    // Compute histogram
    
    process_data(key_storage, fun);

    // Ensure all threads have completed contributing to the bucket counts of their items'
    // Also ensures that we can reuse temporary storage for scan
    __syncthreads();

    // Identify bucket into which the k-th item falls
    compute_bin_offsets();

    // Ensure all the results from the prefix sum have been written back to shared memory
     __syncthreads();

    // Identify the bucket that the k-th item falls into
    choose_bucket(k, pass);
  }

  template <int ItemsPerThread>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Min()
  {
    // storage can be registers or shared memory
    // storage is a data provider. it can be lazy or eager
    // process_data uses a data provider and a functor to process data
    

    
    // target arrangement: blocked, block-strided, warp-strided, arbitrary
    // possible processing:
    // blocked: each thread processes its items
    // block-strided: each thread processes its items in a strided manner
    // warp-strided: each warp processes its items in a strided manner
    // arbitrary: each thread processes its items in an arbitrary manner
    load_keys();

    // process each radix pass
    for (int pass = 0; pass < num_passes; ++pass)
    {
      process_pass(key_storage);
    }

    
  }

  template <int ItemsPerThread>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Max()
  {}
};

// struct on_chip_storage_t
// {

// };

// struct on_chip_smem_storage_t
// {

// };

// struct on_chip_reg_storage_t
// {

// };

// };
#endif

// TODO (elstehle): Add documentation
template <typename T, int BlockDimX, typename EnvT = ::cuda::std::execution::env<>>
class BlockTopK
{
private:
  static constexpr int block_dim_x = BlockDimX;
  static constexpr int block_dim_y = 1;
  static constexpr int block_dim_z = 1;

  static constexpr int block_threads = BlockDimX * block_dim_y * block_dim_z;

  static constexpr BlockTopKAlgorithm selected_algorithm = BlockTopKAlgorithm::air_top_k;

  /// Define the delegate type for the desired algorithm
  using InternalBlockTopK = ::cuda::std::_If<selected_algorithm == BlockTopKAlgorithm::air_top_k, AirTopK, void>;

  /// Shared memory storage layout type for BlockTopK
  using _TempStorage = typename InternalBlockTopK::TempStorage;

  /// Shared storage reference
  _TempStorage& temp_storage;

  /// Linear thread-id
  ::cuda::std::uint32_t linear_tid;

  /// Internal storage allocator
  _CCCL_DEVICE _CCCL_FORCEINLINE _TempStorage& PrivateStorage()
  {
    __shared__ _TempStorage private_storage;
    return private_storage;
  }

public:
  /// @smemstorage{BlockTopK}
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  _CCCL_DEVICE _CCCL_FORCEINLINE BlockTopK()
      : temp_storage(PrivateStorage())
      , linear_tid(RowMajorTid(BlockDimX, block_dim_y, block_dim_z))
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE BlockTopK(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(RowMajorTid(BlockDimX, block_dim_y, block_dim_z))
  {}

  template <int ItemsPerThread>
  _CCCL_DEVICE _CCCL_FORCEINLINE void TopK(T (&input)[ItemsPerThread], T (&output)[ItemsPerThread])
  {
    InternalBlockTopK(temp_storage).TopK(input, output, linear_tid);
  }
};

} // namespace detail

CUB_NAMESPACE_END
