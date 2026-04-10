// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! cub::AgentTopK implements a stateful abstraction of CUDA thread blocks for participating in device-wide topK.
#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/radix_rank_sort_operations.cuh>
#include <cub/device/dispatch/tuning/tuning_topk.cuh>
#include <cub/util_type.cuh>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__type_traits/conditional.h>

CUB_NAMESPACE_BEGIN

namespace detail::topk
{
//! @brief Parameterizable tuning policy type for AgentTopK
//!
//! @tparam BlockThreads
//!   Threads per thread block
//!
//! @tparam ItemsPerThread
//!   Items per thread (per tile of input)
//!
//! @tparam BitsPerPass
//!   Number of bits processed per pass
//!
//! @tparam LoadAlgorithm
//!   The BlockLoad algorithm to use
//!
//! @tparam ScanAlgorithm
//!   The BlockScan algorithm to use
//!
template <int BlockThreads,
          int ItemsPerThread,
          int BitsPerPass,
          BlockLoadAlgorithm LoadAlgorithm,
          BlockScanAlgorithm ScanAlgorithm,
          smem_write_mode WriteMode = smem_write_mode::smem_coalescing_two_phase>
struct AgentTopKPolicy
{
  static constexpr int block_threads                 = BlockThreads;
  static constexpr int items_per_thread              = ItemsPerThread;
  static constexpr int bits_per_pass                 = BitsPerPass;
  static constexpr BlockLoadAlgorithm load_algorithm = LoadAlgorithm;
  static constexpr BlockScanAlgorithm SCAN_ALGORITHM = ScanAlgorithm;
  static constexpr smem_write_mode write_mode        = WriteMode;
};

template <typename KeyT, bool CanTwiddle = detail::radix::can_twiddle<KeyT>>
struct key_prefix_storage_t;

template <typename KeyT>
struct key_prefix_storage_t<KeyT, true>
{
  using bits_t = typename Traits<KeyT>::UnsignedBits;
  bits_t bits;
};

// Calculates the number of passes needed for a type T with BitsPerPass bits processed per pass.
template <typename T>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr int calc_num_passes(int bits_per_pass)
{
  return ::cuda::ceil_div<int>(sizeof(T) * 8, bits_per_pass);
}

template <int BitsPerPass>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int calc_num_passes(const int total_bits)
{
  return ::cuda::ceil_div<int>(total_bits, BitsPerPass);
}

// Calculates the starting bit for a given pass (bit 0 is the least significant (rightmost) bit).
// We process the input from the most to the least significant bit. This way, we can skip some passes in the end.
template <typename T, int BitsPerPass>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr int calc_start_bit(const int pass)
{
  int start_bit = int{sizeof(T)} * 8 - (pass + 1) * BitsPerPass;
  if (start_bit < 0)
  {
    start_bit = 0;
  }
  return start_bit;
}

template <int BitsPerPass>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int calc_start_bit(const int total_bits, const int pass)
{
  int start_bit = total_bits - (pass + 1) * BitsPerPass;
  if (start_bit < 0)
  {
    start_bit = 0;
  }
  return start_bit;
}

// Bit-vector for accumulating prefix digits via funnel shift. Each pass shifts the existing
// contents left by BitsPerPass and ORs the new bucket at the bottom. Sized to hold all
// decomposed bits of KeyT plus headroom for the shift padding of the last pass.
template <typename KeyT>
struct key_prefix_storage_t<KeyT, false>
{
  static constexpr int num_words = ::cuda::ceil_div<int>(sizeof(KeyT) * 8 + 31, 32);
  unsigned int words[num_words];

  // Funnel-shifts the entire bit-vector left by `shift` positions and inserts `value` into the
  // vacated low bits. Each word receives carry bits from its lower neighbor (high-to-low order
  // so each word reads its neighbor's original value). The final word is filled from `value`.
  _CCCL_DEVICE _CCCL_FORCEINLINE void shift_or(int shift, unsigned int value)
  {
    _CCCL_ASSERT(shift > 0 && shift < 32, "shift_or requires 0 < shift < 32");
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = num_words - 1; i > 0; --i)
    {
      words[i] = __funnelshift_l(words[i - 1], words[i], shift);
    }
    words[0] = (words[0] << shift) | value;
  }
};

template <typename KeyT, int BitsPerPass>
_CCCL_DEVICE _CCCL_FORCEINLINE void
set_kth_key_bits(key_prefix_storage_t<KeyT>& prefix, const int pass, const int bin_index)
{
  if constexpr (detail::radix::can_twiddle<KeyT>)
  {
    using bits_t        = typename Traits<KeyT>::UnsignedBits;
    const int start_bit = calc_start_bit<KeyT, BitsPerPass>(pass);
    bits_t bucket       = bin_index;
    prefix.bits |= static_cast<bits_t>(bucket) << start_bit;
  }
  else
  {
    prefix.shift_or(BitsPerPass, bin_index);
  }
}

template <typename KeyInT, typename OffsetT, typename OutOffsetT>
struct alignas(128) Counter
{
  // We are processing the items in multiple passes, from most-significant to least-significant bits. In each pass, we
  // keep the length of input (`len`) and the `k` of current pass, and update them at the end of the pass.
  OutOffsetT k;
  OffsetT len;

  // `previous_len` is the length of the input in the previous pass. Note that `previous_len` rather than `len` is used
  // for the filtering step because filtering is indeed for previous pass.
  OffsetT previous_len;

  // We determine the bits of the k_th key inside the mask processed by the pass. The
  // already known bits are stored in `kth_key_bits`. It's used to discriminate a
  // element is a result (written to `out`), a candidate for next pass (written to
  // `out_buf`), or not useful (discarded). The bits that are not yet processed do not
  // matter for this purpose.
  key_prefix_storage_t<KeyInT> kth_key_bits;

  // Record how many elements have passed filtering. It's used to determine the position
  // in the `out_buf` where an element should be written.
  alignas(128) OffsetT filter_cnt;

  // For a row inside a batch, we may launch multiple thread blocks. This counter is
  // used to determine if the current block is the last running block. If so, this block
  // will execute compute_bin_offsets() and choose_bucket().
  alignas(128) unsigned int finished_block_cnt;

  // Record how many elements have been written to the front of `out`. Elements less (if
  // SelectMin==true) than the k-th key are written from front to back.
  alignas(128) OutOffsetT out_cnt;

  // Record how many elements have been written to the back of `out`. Elements equal to
  // the k-th key are written from back to front. We need to keep count of them
  // separately because the number of elements that <= the k-th key might exceed k.
  alignas(128) OutOffsetT out_back_cnt;
  // The 'alignas' is necessary to improve the performance of global memory accessing by isolating the request,
  // especially for the segment version.
};

enum class candidate_class
{
  // The given candidate is definitely amongst the top-k items
  selected,
  // The given candidate may or may not be amongst the top-k items
  candidate,
  // The given candidate is definitely not amongst the top-k items
  rejected
};

//! @brief AgentTopK implements a stateful abstraction of CUDA thread blocks for participating in
//! device-wide topK
//!
//! @tparam AgentTopKPolicyT
//!   Parameterized AgentTopKPolicy tuning policy type
//!
//! @tparam KeyInputIteratorT
//!   **[inferred]** Random-access input iterator type for reading input keys @iterator
//!
//! @tparam KeyOutputIteratorT
//!   **[inferred]** Random-access output iterator type for writing output keys @iterator
//!
//! @tparam ValueInputIteratorT
//!   **[inferred]** Random-access input iterator type for reading input values @iterator
//!
//! @tparam ValueOutputIteratorT
//!   **[inferred]** Random-access output iterator type for writing output values @iterator
//!
//! @tparam ExtractBinOpT
//!   Operations to extract the bin from the input key values
//!
//! @tparam IdentifyCandidatesOpT
//!    Operations to filter the input key values
//!
//! @tparam OffsetT
//!   Type of variable num_items
//!
//! @tparam OutOffsetT
//!   Type of variable k
//!
template <typename AgentTopKPolicyT,
          typename KeyInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueInputIteratorT,
          typename ValueOutputIteratorT,
          typename ExtractBinOpT,
          typename IdentifyCandidatesOpT,
          typename OffsetT,
          typename OutOffsetT>
struct AgentTopK
{
  //---------------------------------------------------------------------
  // Types and constants
  //---------------------------------------------------------------------
  // The key and value type
  using key_in_t   = it_value_t<KeyInputIteratorT>;
  using value_in_t = it_value_t<ValueInputIteratorT>;

  static constexpr int block_threads    = AgentTopKPolicyT::block_threads;
  static constexpr int items_per_thread = AgentTopKPolicyT::items_per_thread;
  static constexpr int bits_per_pass    = AgentTopKPolicyT::bits_per_pass;
  static constexpr int tile_items       = block_threads * items_per_thread;
  static constexpr int num_buckets      = 1 << bits_per_pass;

  static constexpr bool keys_only      = ::cuda::std::is_same_v<value_in_t, NullType>;
  static constexpr int bins_per_thread = ::cuda::ceil_div(num_buckets, block_threads);

  static constexpr smem_write_mode write_mode = AgentTopKPolicyT::write_mode;

  // For keys_only kernels, two_phase is equivalent to smem_coalescing
  static constexpr smem_write_mode effective_write_mode =
    (keys_only && write_mode == smem_write_mode::smem_coalescing_two_phase)
      ? smem_write_mode::smem_coalescing
      : write_mode;

  // Parameterized BlockLoad type for input data
  using block_load_input_t = BlockLoad<key_in_t, block_threads, items_per_thread, AgentTopKPolicyT::load_algorithm>;
  using block_load_trans_t = BlockLoad<OffsetT, block_threads, bins_per_thread, BLOCK_LOAD_TRANSPOSE>;
  // Parameterized BlockScan type
  using block_scan_t = BlockScan<OffsetT, block_threads, AgentTopKPolicyT::SCAN_ALGORITHM>;
  // Parameterized BlockStore type
  using block_store_trans_t = BlockStore<OffsetT, block_threads, bins_per_thread, BLOCK_STORE_TRANSPOSE>;

  struct noop_tile_epilogue
  {
    _CCCL_DEVICE void operator()() const {}
  };

  //---------------------------------------------------------------------
  // Staging buffer type selection
  //---------------------------------------------------------------------

  struct _StagingDisabled
  {
    key_in_t keys[1];
    OffsetT indices[1];
  };

  struct _StagingCoalescing
  {
    key_in_t keys[tile_items];
    OffsetT indices[tile_items];
  };

  struct _StagingKeysOnly
  {
    key_in_t keys[tile_items];
    OffsetT indices[1];
  };

  // Two-phase: keys and indices share storage via anonymous union
  struct _StagingTwoPhase
  {
    union
    {
      key_in_t keys[tile_items];
      OffsetT indices[tile_items];
    };
  };

  using staging_t = ::cuda::std::conditional_t<
    effective_write_mode == smem_write_mode::no_smem_coalescing,
    _StagingDisabled,
    ::cuda::std::conditional_t<effective_write_mode == smem_write_mode::smem_coalescing_two_phase,
                               _StagingTwoPhase,
                               ::cuda::std::conditional_t<keys_only, _StagingKeysOnly, _StagingCoalescing>>>;

  // Shared memory
  struct _TempStorage
  {
    union
    {
      // Smem needed for loading
      typename block_load_input_t::TempStorage load_input;
      typename block_load_trans_t::TempStorage load_trans;
      // Smem needed for scan
      typename block_scan_t::TempStorage scan;
      // Smem needed for storing
      typename block_store_trans_t::TempStorage store_trans;

      staging_t staging;
    };
    OffsetT histogram[num_buckets];

    // Write coordination counters
    OffsetT smem_filter_cnt;
    OutOffsetT smem_out_cnt;
    OffsetT block_filter_base;
    OutOffsetT block_out_base;
    OutOffsetT block_out_back_base;
  };
  /// Alias wrapper allowing storage to be unioned
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //---------------------------------------------------------------------
  // Per-thread fields
  //---------------------------------------------------------------------
  _TempStorage& temp_storage; // Reference to temp_storage
  KeyInputIteratorT d_keys_in; // Input keys
  KeyOutputIteratorT d_keys_out; // Output keys
  ValueInputIteratorT d_values_in; // Input values
  ValueOutputIteratorT d_values_out; // Output values
  OffsetT num_items; // Total number of input items
  OutOffsetT k; // Total number of output items
  OffsetT buffer_length; // Size of the buffer for storing intermediate candidates
  ExtractBinOpT extract_bin_op; // The operation for bin
  IdentifyCandidatesOpT identify_candidates_op; // The operation for filtering

  //---------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------
  //! @param temp_storage
  //!   Reference to temp_storage
  //!
  //! @param d_keys_in
  //!   Input data, keys
  //!
  //! @param d_keys_out
  //!   Output data, keys
  //!
  //! @param d_values_in
  //!   Input data, values
  //!
  //! @param d_values_out
  //!   Output data, values
  //!
  //! @param num_items
  //!   Total number of input items
  //!
  //! @param k
  //!   The K value. Will find K elements from num_items elements
  //!
  //! @param buffer_length
  //!   The size of the buffer for storing intermediate candidates
  //!
  //! @param extract_bin_op
  //!   Extract bin operator
  //!
  //! @param identify_candidates_op
  //!   Filter operator
  //!
  _CCCL_DEVICE _CCCL_FORCEINLINE AgentTopK(
    TempStorage& temp_storage,
    const KeyInputIteratorT d_keys_in,
    KeyOutputIteratorT d_keys_out,
    const ValueInputIteratorT d_values_in,
    ValueOutputIteratorT d_values_out,
    OffsetT num_items,
    OutOffsetT k,
    OffsetT buffer_length,
    ExtractBinOpT extract_bin_op,
    IdentifyCandidatesOpT identify_candidates_op)
      : temp_storage(temp_storage.Alias())
      , d_keys_in(d_keys_in)
      , d_keys_out(d_keys_out)
      , d_values_in(d_values_in)
      , d_values_out(d_values_out)
      , num_items(num_items)
      , k(k)
      , buffer_length(buffer_length)
      , extract_bin_op(extract_bin_op)
      , identify_candidates_op(identify_candidates_op)
  {}

  //---------------------------------------------------------------------
  // Utility methods for device topK
  //---------------------------------------------------------------------

  // Process a range of input data in tiles, calling f(key, index) for each element
  template <typename InputItT, typename FuncT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void process_range(InputItT in, const OffsetT num_items, FuncT f)
  {
    process_range(in, num_items, f, noop_tile_epilogue{});
  }

  // Process a range of input data in tiles, calling f(key, index) for each element
  // and tile_epilogue() after each tile completes
  template <typename InputItT, typename FuncT, typename TileEpilogueT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  process_range(InputItT in, const OffsetT num_items, FuncT f, TileEpilogueT tile_epilogue)
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

      tile_epilogue();

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

      tile_epilogue();
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void init_histograms(OffsetT* histogram)
  {
    // Initialize histogram bin counts to zeros
    int histo_offset = 0;

    // Loop unrolling is beneficial for performance here
    _CCCL_PRAGMA_UNROLL_FULL()
    for (; histo_offset + block_threads <= num_buckets; histo_offset += block_threads)
    {
      histogram[histo_offset + threadIdx.x] = 0;
    }
    // Finish up with guarded initialization if necessary
    if ((num_buckets % block_threads != 0) && (histo_offset + threadIdx.x < num_buckets))
    {
      histogram[histo_offset + threadIdx.x] = 0;
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void merge_histograms(OffsetT* global_histogram)
  {
    int histo_offset = 0;

    // Loop unrolling is beneficial for performance here
    _CCCL_PRAGMA_UNROLL_FULL()
    for (; histo_offset + block_threads <= num_buckets; histo_offset += block_threads)
    {
      if (temp_storage.histogram[histo_offset + threadIdx.x] != 0)
      {
        atomicAdd(global_histogram + (histo_offset + threadIdx.x), temp_storage.histogram[histo_offset + threadIdx.x]);
      }
    }

    // Finish up with guarded merging if necessary
    if ((num_buckets % block_threads != 0) && (histo_offset + threadIdx.x < num_buckets))
    {
      atomicAdd(global_histogram + (histo_offset + threadIdx.x), temp_storage.histogram[histo_offset + threadIdx.x]);
    }
  }

  // Fused filtering of the current pass and building histogram for the next pass
  _CCCL_DEVICE _CCCL_FORCEINLINE void filter_and_histogram(
    key_in_t* in_buf,
    OffsetT* in_idx_buf,
    key_in_t* out_buf,
    OffsetT* out_idx_buf,
    OffsetT previous_len,
    Counter<key_in_t, OffsetT, OutOffsetT>* counter,
    OffsetT* histogram,
    bool early_stop,
    bool load_from_original_input)
  {
    // Initialize shared memory histogram
    init_histograms(temp_storage.histogram);

    if constexpr (effective_write_mode != smem_write_mode::no_smem_coalescing)
    {
      if (threadIdx.x == 0)
      {
        temp_storage.smem_filter_cnt = 0;
        temp_storage.smem_out_cnt    = 0;
      }
    }

    // Make sure the histogram and counters were initialized
    __syncthreads();

    OffsetT* p_filter_cnt = &counter->filter_cnt;
    OutOffsetT* p_out_cnt = &counter->out_cnt;

    // Histogram-only lambda is shared across all modes
    auto f_no_out_buf = [this](key_in_t key, OffsetT /*i*/) {
      const candidate_class pre_res = identify_candidates_op(key);
      if (pre_res == candidate_class::candidate)
      {
        const int bucket = extract_bin_op(key);
        atomicAdd(temp_storage.histogram + bucket, OffsetT{1});
      }
    };

    if constexpr (effective_write_mode == smem_write_mode::smem_coalescing_two_phase)
    {
      // === TWO-PHASE PATH (always !keys_only due to effective_write_mode mapping) ===

      // Per-thread register arrays that persist across per-element lambda calls within a tile
      candidate_class thread_flags[items_per_thread];
      int thread_local_pos[items_per_thread];
      OffsetT thread_resolved_idx[items_per_thread];
      int thread_item_idx = 0;

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int j = 0; j < items_per_thread; ++j)
      {
        thread_flags[j] = candidate_class::rejected;
      }

      auto f_early_stop = [&](key_in_t key, OffsetT i) {
        const int j                   = thread_item_idx++;
        const candidate_class pre_res = identify_candidates_op(key);
        if (pre_res == candidate_class::candidate || pre_res == candidate_class::selected)
        {
          thread_flags[j]                      = pre_res;
          const auto local_pos                 = atomicAdd(&temp_storage.smem_out_cnt, OutOffsetT{1});
          thread_local_pos[j]                  = static_cast<int>(local_pos);
          temp_storage.staging.keys[local_pos] = key;
          thread_resolved_idx[j]               = load_from_original_input ? i : in_idx_buf[i];
        }
      };

      auto early_stop_epilogue = [&]() {
        __syncthreads();
        const OutOffsetT n_out = temp_storage.smem_out_cnt;
        if (threadIdx.x == 0)
        {
          temp_storage.block_out_base = atomicAdd(p_out_cnt, n_out);
        }
        __syncthreads();
        const OutOffsetT out_base = temp_storage.block_out_base;
        // Phase 1: flush keys
        for (OutOffsetT i = threadIdx.x; i < n_out; i += block_threads)
        {
          d_keys_out[out_base + i] = temp_storage.staging.keys[i];
        }
        // Phase 2: reuse staging for indices
        __syncthreads();
        for (int j = 0; j < items_per_thread; ++j)
        {
          if (thread_flags[j] != candidate_class::rejected)
          {
            temp_storage.staging.indices[thread_local_pos[j]] = thread_resolved_idx[j];
          }
        }
        __syncthreads();
        for (OutOffsetT i = threadIdx.x; i < n_out; i += block_threads)
        {
          d_values_out[out_base + i] = d_values_in[temp_storage.staging.indices[i]];
        }
        // Reset for next tile
        if (threadIdx.x == 0)
        {
          temp_storage.smem_out_cnt = 0;
        }
        thread_item_idx = 0;
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int j = 0; j < items_per_thread; ++j)
        {
          thread_flags[j] = candidate_class::rejected;
        }
      };

      auto f_with_out_buf = [&](key_in_t key, OffsetT i) {
        const int j                   = thread_item_idx++;
        const candidate_class pre_res = identify_candidates_op(key);
        if (pre_res == candidate_class::candidate)
        {
          thread_flags[j]                                       = pre_res;
          const auto local_pos                                  = atomicAdd(&temp_storage.smem_filter_cnt, OffsetT{1});
          thread_local_pos[j]                                   = static_cast<int>(local_pos);
          temp_storage.staging.keys[tile_items - 1 - local_pos] = key;
          thread_resolved_idx[j]                                = load_from_original_input ? i : in_idx_buf[i];

          const int bucket = extract_bin_op(key);
          atomicAdd(temp_storage.histogram + bucket, OffsetT{1});
        }
        else if (pre_res == candidate_class::selected)
        {
          thread_flags[j]                      = pre_res;
          const auto local_pos                 = atomicAdd(&temp_storage.smem_out_cnt, OutOffsetT{1});
          thread_local_pos[j]                  = static_cast<int>(local_pos);
          temp_storage.staging.keys[local_pos] = key;
          thread_resolved_idx[j]               = in_idx_buf ? in_idx_buf[i] : i;
        }
      };

      auto with_out_buf_epilogue = [&]() {
        __syncthreads();
        const OffsetT n_candidates  = temp_storage.smem_filter_cnt;
        const OutOffsetT n_selected = temp_storage.smem_out_cnt;
        if (threadIdx.x == 0)
        {
          temp_storage.block_filter_base = atomicAdd(p_filter_cnt, n_candidates);
          temp_storage.block_out_base    = atomicAdd(p_out_cnt, n_selected);
        }
        __syncthreads();
        const OffsetT filter_base = temp_storage.block_filter_base;
        const OutOffsetT out_base = temp_storage.block_out_base;
        // Phase 1: flush keys
        for (OutOffsetT i = threadIdx.x; i < n_selected; i += block_threads)
        {
          d_keys_out[out_base + i] = temp_storage.staging.keys[i];
        }
        const OffsetT cand_start = static_cast<OffsetT>(tile_items) - n_candidates;
        for (OffsetT i = threadIdx.x; i < n_candidates; i += block_threads)
        {
          out_buf[filter_base + i] = temp_storage.staging.keys[cand_start + i];
        }
        // Phase 2: reuse staging for indices
        __syncthreads();
        for (int j = 0; j < items_per_thread; ++j)
        {
          if (thread_flags[j] == candidate_class::selected)
          {
            temp_storage.staging.indices[thread_local_pos[j]] = thread_resolved_idx[j];
          }
          else if (thread_flags[j] == candidate_class::candidate)
          {
            temp_storage.staging.indices[tile_items - 1 - thread_local_pos[j]] = thread_resolved_idx[j];
          }
        }
        __syncthreads();
        for (OutOffsetT i = threadIdx.x; i < n_selected; i += block_threads)
        {
          d_values_out[out_base + i] = d_values_in[temp_storage.staging.indices[i]];
        }
        for (OffsetT i = threadIdx.x; i < n_candidates; i += block_threads)
        {
          out_idx_buf[filter_base + i] = temp_storage.staging.indices[cand_start + i];
        }
        // Reset for next tile
        if (threadIdx.x == 0)
        {
          temp_storage.smem_filter_cnt = 0;
          temp_storage.smem_out_cnt    = 0;
        }
        thread_item_idx = 0;
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int j = 0; j < items_per_thread; ++j)
        {
          thread_flags[j] = candidate_class::rejected;
        }
      };

      if (load_from_original_input)
      {
        if (early_stop)
        {
          process_range(d_keys_in, previous_len, f_early_stop, early_stop_epilogue);
        }
        else if (out_buf)
        {
          process_range(d_keys_in, previous_len, f_with_out_buf, with_out_buf_epilogue);
        }
        else
        {
          process_range(d_keys_in, previous_len, f_no_out_buf);
        }
      }
      else
      {
        if (early_stop)
        {
          process_range(in_buf, previous_len, f_early_stop, early_stop_epilogue);
        }
        else if (out_buf)
        {
          process_range(in_buf, previous_len, f_with_out_buf, with_out_buf_epilogue);
        }
        else
        {
          process_range(in_buf, previous_len, f_no_out_buf);
        }
      }
    }
    else if constexpr (effective_write_mode == smem_write_mode::smem_coalescing)
    {
      // === SINGLE-PHASE SMEM WRITE COORDINATION PATH ===

      auto f_early_stop = [load_from_original_input, in_idx_buf, this](key_in_t key, OffsetT i) {
        const candidate_class pre_res = identify_candidates_op(key);
        if (pre_res == candidate_class::candidate || pre_res == candidate_class::selected)
        {
          const auto local_pos                 = atomicAdd(&temp_storage.smem_out_cnt, OutOffsetT{1});
          temp_storage.staging.keys[local_pos] = key;
          if constexpr (!keys_only)
          {
            temp_storage.staging.indices[local_pos] = load_from_original_input ? i : in_idx_buf[i];
          }
        }
      };

      auto early_stop_epilogue = [p_out_cnt, this]() {
        __syncthreads();
        const OutOffsetT n_out = temp_storage.smem_out_cnt;
        if (threadIdx.x == 0)
        {
          temp_storage.block_out_base = atomicAdd(p_out_cnt, n_out);
        }
        __syncthreads();
        const OutOffsetT out_base = temp_storage.block_out_base;
        for (OutOffsetT i = threadIdx.x; i < n_out; i += block_threads)
        {
          d_keys_out[out_base + i] = temp_storage.staging.keys[i];
          if constexpr (!keys_only)
          {
            d_values_out[out_base + i] = d_values_in[temp_storage.staging.indices[i]];
          }
        }
        if (threadIdx.x == 0)
        {
          temp_storage.smem_out_cnt = 0;
        }
      };

      auto f_with_out_buf = [load_from_original_input, in_idx_buf, this](key_in_t key, OffsetT i) {
        const candidate_class pre_res = identify_candidates_op(key);
        if (pre_res == candidate_class::candidate)
        {
          const auto local_pos                                  = atomicAdd(&temp_storage.smem_filter_cnt, OffsetT{1});
          temp_storage.staging.keys[tile_items - 1 - local_pos] = key;
          if constexpr (!keys_only)
          {
            temp_storage.staging.indices[tile_items - 1 - local_pos] = load_from_original_input ? i : in_idx_buf[i];
          }

          const int bucket = extract_bin_op(key);
          atomicAdd(temp_storage.histogram + bucket, OffsetT{1});
        }
        else if (pre_res == candidate_class::selected)
        {
          const auto local_pos                 = atomicAdd(&temp_storage.smem_out_cnt, OutOffsetT{1});
          temp_storage.staging.keys[local_pos] = key;
          if constexpr (!keys_only)
          {
            temp_storage.staging.indices[local_pos] = in_idx_buf ? in_idx_buf[i] : i;
          }
        }
      };

      auto with_out_buf_epilogue = [p_filter_cnt, p_out_cnt, out_buf, out_idx_buf, this]() {
        __syncthreads();
        const OffsetT n_candidates  = temp_storage.smem_filter_cnt;
        const OutOffsetT n_selected = temp_storage.smem_out_cnt;
        if (threadIdx.x == 0)
        {
          temp_storage.block_filter_base = atomicAdd(p_filter_cnt, n_candidates);
          temp_storage.block_out_base    = atomicAdd(p_out_cnt, n_selected);
        }
        __syncthreads();
        const OffsetT filter_base = temp_storage.block_filter_base;
        const OutOffsetT out_base = temp_storage.block_out_base;
        for (OutOffsetT i = threadIdx.x; i < n_selected; i += block_threads)
        {
          d_keys_out[out_base + i] = temp_storage.staging.keys[i];
          if constexpr (!keys_only)
          {
            d_values_out[out_base + i] = d_values_in[temp_storage.staging.indices[i]];
          }
        }
        const OffsetT cand_start = static_cast<OffsetT>(tile_items) - n_candidates;
        for (OffsetT i = threadIdx.x; i < n_candidates; i += block_threads)
        {
          out_buf[filter_base + i] = temp_storage.staging.keys[cand_start + i];
          if constexpr (!keys_only)
          {
            out_idx_buf[filter_base + i] = temp_storage.staging.indices[cand_start + i];
          }
        }
        if (threadIdx.x == 0)
        {
          temp_storage.smem_filter_cnt = 0;
          temp_storage.smem_out_cnt    = 0;
        }
      };

      if (load_from_original_input)
      {
        if (early_stop)
        {
          process_range(d_keys_in, previous_len, f_early_stop, early_stop_epilogue);
        }
        else if (out_buf)
        {
          process_range(d_keys_in, previous_len, f_with_out_buf, with_out_buf_epilogue);
        }
        else
        {
          process_range(d_keys_in, previous_len, f_no_out_buf);
        }
      }
      else
      {
        if (early_stop)
        {
          process_range(in_buf, previous_len, f_early_stop, early_stop_epilogue);
        }
        else if (out_buf)
        {
          process_range(in_buf, previous_len, f_with_out_buf, with_out_buf_epilogue);
        }
        else
        {
          process_range(in_buf, previous_len, f_no_out_buf);
        }
      }
    }
    else
    {
      // === NO SMEM COORDINATION PATH ===

      auto f_early_stop = [load_from_original_input, in_idx_buf, p_out_cnt, this](key_in_t key, OffsetT i) {
        const candidate_class pre_res = identify_candidates_op(key);
        if (pre_res == candidate_class::candidate || pre_res == candidate_class::selected)
        {
          const OutOffsetT pos = atomicAdd(p_out_cnt, OutOffsetT{1});
          d_keys_out[pos]      = key;
          if constexpr (!keys_only)
          {
            const OffsetT index = load_from_original_input ? i : in_idx_buf[i];
            d_values_out[pos]   = d_values_in[index];
          }
        }
      };

      auto f_with_out_buf = [load_from_original_input, in_idx_buf, out_buf, out_idx_buf, p_filter_cnt, p_out_cnt, this](
                              key_in_t key, OffsetT i) {
        const candidate_class pre_res = identify_candidates_op(key);
        if (pre_res == candidate_class::candidate)
        {
          const OffsetT pos = atomicAdd(p_filter_cnt, OffsetT{1});
          out_buf[pos]      = key;
          if constexpr (!keys_only)
          {
            const OffsetT index = load_from_original_input ? i : in_idx_buf[i];
            out_idx_buf[pos]    = index;
          }

          const int bucket = extract_bin_op(key);
          atomicAdd(temp_storage.histogram + bucket, OffsetT{1});
        }
        else if (pre_res == candidate_class::selected)
        {
          const OutOffsetT pos = atomicAdd(p_out_cnt, OutOffsetT{1});
          d_keys_out[pos]      = key;
          if constexpr (!keys_only)
          {
            const OffsetT index = in_idx_buf ? in_idx_buf[i] : i;
            d_values_out[pos]   = d_values_in[index];
          }
        }
      };

      if (load_from_original_input)
      {
        if (early_stop)
        {
          process_range(d_keys_in, previous_len, f_early_stop);
        }
        else if (out_buf)
        {
          process_range(d_keys_in, previous_len, f_with_out_buf);
        }
        else
        {
          process_range(d_keys_in, previous_len, f_no_out_buf);
        }
      }
      else
      {
        if (early_stop)
        {
          process_range(in_buf, previous_len, f_early_stop);
        }
        else if (out_buf)
        {
          process_range(in_buf, previous_len, f_with_out_buf);
        }
        else
        {
          process_range(in_buf, previous_len, f_no_out_buf);
        }
      }
    }

    // Early stop means that subsequent passes are not needed
    if (early_stop)
    {
      return;
    }

    // Ensure all threads have contributed to the histogram before accumulating in the global memory
    __syncthreads();

    // Merge the locally aggregated histogram into the global histogram
    merge_histograms(histogram);
  }

  // Replace histogram with its own prefix sum
  _CCCL_DEVICE _CCCL_FORCEINLINE void compute_bin_offsets(volatile OffsetT* histogram)
  {
    OffsetT thread_data[bins_per_thread]{};

    // Load global histogram (we can skip initializing oob-items to zero because they won't be stored back)
    block_load_trans_t(temp_storage.load_trans).Load(histogram, thread_data, num_buckets);
    __syncthreads();

    block_scan_t(temp_storage.scan).InclusiveSum(thread_data, thread_data);
    __syncthreads();

    block_store_trans_t(temp_storage.store_trans).Store(temp_storage.histogram, thread_data, num_buckets);
  }

  // Identify the bucket that the k-th value falls into
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  choose_bucket(Counter<key_in_t, OffsetT, OutOffsetT>* counter, const OutOffsetT k, const int pass)
  {
    // Initialize histogram bin counts to zeros
    int histo_offset = 0;

    auto body = [&] {
      const int bin_idx  = histo_offset + threadIdx.x;
      const OffsetT prev = (bin_idx == 0) ? 0 : temp_storage.histogram[bin_idx - 1];
      const OffsetT cur  = temp_storage.histogram[bin_idx];

      // Identify the bin that the k-th item falls into. One and only one thread will satisfy this condition, so counter
      // is written by only one thread
      if (prev < k && cur >= k)
      {
        // The number of items that are yet to be identified
        counter->k = k - prev;

        // The number of candidates in the next pass
        counter->len              = cur - prev;
        const unsigned int bucket = static_cast<unsigned int>(bin_idx);
        // Update the "splitter" key by adding the radix digit of the k-th item bin of this pass
        set_kth_key_bits<key_in_t, bits_per_pass>(counter->kth_key_bits, pass, bucket);
      }
    };

    _CCCL_PRAGMA_UNROLL_FULL()
    for (; histo_offset + block_threads <= num_buckets; histo_offset += block_threads)
    {
      body();
    }
    // Finish up with guarded initialization if necessary
    if ((num_buckets % block_threads != 0) && (histo_offset + threadIdx.x < num_buckets))
    {
      body();
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void invoke_last_filter(
    key_in_t* in_buf, OffsetT* in_idx_buf, Counter<key_in_t, OffsetT, OutOffsetT>* counter, OutOffsetT k, int pass)
  {
    const bool load_from_original_input = (pass <= 1) || counter->previous_len > buffer_length;
    const OffsetT current_len           = load_from_original_input ? num_items : counter->previous_len;
    in_idx_buf = load_from_original_input ? nullptr : in_idx_buf; // ? out_idx_buf : in_idx_buf;

    if (current_len == 0)
    {
      return;
    }

    // changed in choose_bucket(); need to reload
    OffsetT num_of_kth_needed  = counter->k;
    OutOffsetT* p_out_cnt      = &counter->out_cnt;
    OutOffsetT* p_out_back_cnt = &counter->out_back_cnt;

    if constexpr (effective_write_mode == smem_write_mode::smem_coalescing_two_phase)
    {
      // === TWO-PHASE LAST FILTER (always !keys_only) ===
      if (threadIdx.x == 0)
      {
        temp_storage.smem_filter_cnt = 0;
        temp_storage.smem_out_cnt    = 0;
      }
      __syncthreads();

      candidate_class thread_flags[items_per_thread];
      int thread_local_pos[items_per_thread];
      OffsetT thread_resolved_idx[items_per_thread];
      int thread_item_idx = 0;

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int j = 0; j < items_per_thread; ++j)
      {
        thread_flags[j] = candidate_class::rejected;
      }

      auto f = [&](key_in_t key, OffsetT i) {
        const int j               = thread_item_idx++;
        const candidate_class res = identify_candidates_op(key);
        if (res == candidate_class::selected)
        {
          thread_flags[j]                      = res;
          const auto local_pos                 = atomicAdd(&temp_storage.smem_out_cnt, OutOffsetT{1});
          thread_local_pos[j]                  = static_cast<int>(local_pos);
          temp_storage.staging.keys[local_pos] = key;
          thread_resolved_idx[j]               = load_from_original_input ? i : in_idx_buf[i];
        }
        else if (res == candidate_class::candidate)
        {
          thread_flags[j]                                       = res;
          const auto local_pos                                  = atomicAdd(&temp_storage.smem_filter_cnt, OffsetT{1});
          thread_local_pos[j]                                   = static_cast<int>(local_pos);
          temp_storage.staging.keys[tile_items - 1 - local_pos] = key;
          thread_resolved_idx[j]                                = load_from_original_input ? i : in_idx_buf[i];
        }
      };

      auto epilogue = [&]() {
        __syncthreads();
        const OutOffsetT n_selected = temp_storage.smem_out_cnt;
        const OffsetT n_candidates  = temp_storage.smem_filter_cnt;
        if (threadIdx.x == 0)
        {
          temp_storage.block_out_base      = atomicAdd(p_out_cnt, n_selected);
          temp_storage.block_out_back_base = atomicAdd(p_out_back_cnt, static_cast<OutOffsetT>(n_candidates));
        }
        __syncthreads();
        const OutOffsetT out_base  = temp_storage.block_out_base;
        const OutOffsetT back_base = temp_storage.block_out_back_base;

        // Phase 1: flush keys
        for (OutOffsetT i = threadIdx.x; i < n_selected; i += block_threads)
        {
          d_keys_out[out_base + i] = temp_storage.staging.keys[i];
        }
        const OffsetT cand_start = static_cast<OffsetT>(tile_items) - n_candidates;
        for (OffsetT i = threadIdx.x; i < n_candidates; i += block_threads)
        {
          const OutOffsetT back_pos = back_base + static_cast<OutOffsetT>(i);
          if (back_pos < num_of_kth_needed)
          {
            d_keys_out[k - 1 - back_pos] = temp_storage.staging.keys[cand_start + i];
          }
        }

        // Phase 2: reuse staging for indices
        __syncthreads();
        for (int j = 0; j < items_per_thread; ++j)
        {
          if (thread_flags[j] == candidate_class::selected)
          {
            temp_storage.staging.indices[thread_local_pos[j]] = thread_resolved_idx[j];
          }
          else if (thread_flags[j] == candidate_class::candidate)
          {
            temp_storage.staging.indices[tile_items - 1 - thread_local_pos[j]] = thread_resolved_idx[j];
          }
        }
        __syncthreads();
        for (OutOffsetT i = threadIdx.x; i < n_selected; i += block_threads)
        {
          d_values_out[out_base + i] = d_values_in[temp_storage.staging.indices[i]];
        }
        for (OffsetT i = threadIdx.x; i < n_candidates; i += block_threads)
        {
          const OutOffsetT back_pos = back_base + static_cast<OutOffsetT>(i);
          if (back_pos < num_of_kth_needed)
          {
            d_values_out[k - 1 - back_pos] = d_values_in[temp_storage.staging.indices[cand_start + i]];
          }
        }

        // Reset for next tile
        if (threadIdx.x == 0)
        {
          temp_storage.smem_filter_cnt = 0;
          temp_storage.smem_out_cnt    = 0;
        }
        thread_item_idx = 0;
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int j = 0; j < items_per_thread; ++j)
        {
          thread_flags[j] = candidate_class::rejected;
        }
      };

      if (load_from_original_input)
      {
        process_range(d_keys_in, current_len, f, epilogue);
      }
      else
      {
        process_range(in_buf, current_len, f, epilogue);
      }
    }
    else if constexpr (effective_write_mode == smem_write_mode::smem_coalescing)
    {
      // === SINGLE-PHASE LAST FILTER ===
      if (threadIdx.x == 0)
      {
        temp_storage.smem_filter_cnt = 0;
        temp_storage.smem_out_cnt    = 0;
      }
      __syncthreads();

      auto f = [this, in_idx_buf, load_from_original_input](key_in_t key, OffsetT i) {
        const candidate_class res = identify_candidates_op(key);
        if (res == candidate_class::selected)
        {
          const auto local_pos                 = atomicAdd(&temp_storage.smem_out_cnt, OutOffsetT{1});
          temp_storage.staging.keys[local_pos] = key;
          if constexpr (!keys_only)
          {
            temp_storage.staging.indices[local_pos] = load_from_original_input ? i : in_idx_buf[i];
          }
        }
        else if (res == candidate_class::candidate)
        {
          const auto local_pos                                  = atomicAdd(&temp_storage.smem_filter_cnt, OffsetT{1});
          temp_storage.staging.keys[tile_items - 1 - local_pos] = key;
          if constexpr (!keys_only)
          {
            temp_storage.staging.indices[tile_items - 1 - local_pos] = load_from_original_input ? i : in_idx_buf[i];
          }
        }
      };

      auto epilogue = [this, p_out_cnt, p_out_back_cnt, num_of_kth_needed, k]() {
        __syncthreads();
        const OutOffsetT n_selected = temp_storage.smem_out_cnt;
        const OffsetT n_candidates  = temp_storage.smem_filter_cnt;
        if (threadIdx.x == 0)
        {
          temp_storage.block_out_base      = atomicAdd(p_out_cnt, n_selected);
          temp_storage.block_out_back_base = atomicAdd(p_out_back_cnt, static_cast<OutOffsetT>(n_candidates));
        }
        __syncthreads();
        const OutOffsetT out_base  = temp_storage.block_out_base;
        const OutOffsetT back_base = temp_storage.block_out_back_base;

        for (OutOffsetT i = threadIdx.x; i < n_selected; i += block_threads)
        {
          d_keys_out[out_base + i] = temp_storage.staging.keys[i];
          if constexpr (!keys_only)
          {
            d_values_out[out_base + i] = d_values_in[temp_storage.staging.indices[i]];
          }
        }

        const OffsetT cand_start = static_cast<OffsetT>(tile_items) - n_candidates;
        for (OffsetT i = threadIdx.x; i < n_candidates; i += block_threads)
        {
          const OutOffsetT back_pos = back_base + static_cast<OutOffsetT>(i);
          if (back_pos < num_of_kth_needed)
          {
            const OutOffsetT pos = k - 1 - back_pos;
            d_keys_out[pos]      = temp_storage.staging.keys[cand_start + i];
            if constexpr (!keys_only)
            {
              d_values_out[pos] = d_values_in[temp_storage.staging.indices[cand_start + i]];
            }
          }
        }

        if (threadIdx.x == 0)
        {
          temp_storage.smem_filter_cnt = 0;
          temp_storage.smem_out_cnt    = 0;
        }
      };

      if (load_from_original_input)
      {
        process_range(d_keys_in, current_len, f, epilogue);
      }
      else
      {
        process_range(in_buf, current_len, f, epilogue);
      }
    }
    else
    {
      // === NO SMEM COORDINATION LAST FILTER ===
      auto f = [this, p_out_cnt, in_idx_buf, p_out_back_cnt, num_of_kth_needed, k, load_from_original_input](
                 key_in_t key, OffsetT i) {
        const candidate_class res = identify_candidates_op(key);
        if (res == candidate_class::selected)
        {
          const OutOffsetT pos = atomicAdd(p_out_cnt, OffsetT{1});
          d_keys_out[pos]      = key;
          if constexpr (!keys_only)
          {
            const OffsetT index = load_from_original_input ? i : in_idx_buf[i];
            d_values_out[pos]   = d_values_in[index];
          }
        }
        else if (res == candidate_class::candidate)
        {
          const OutOffsetT back_pos = atomicAdd(p_out_back_cnt, OffsetT{1});

          if (back_pos < num_of_kth_needed)
          {
            const OutOffsetT pos = k - 1 - back_pos;
            d_keys_out[pos]      = key;
            if constexpr (!keys_only)
            {
              const OffsetT new_idx = load_from_original_input ? i : in_idx_buf[i];
              d_values_out[pos]     = d_values_in[new_idx];
            }
          }
        }
      };

      if (load_from_original_input)
      {
        process_range(d_keys_in, current_len, f);
      }
      else
      {
        process_range(in_buf, current_len, f);
      }
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void invoke_filter_and_histogram(
    key_in_t* in_buf,
    OffsetT* in_idx_buf,
    key_in_t* out_buf,
    OffsetT* out_idx_buf,
    Counter<key_in_t, OffsetT, OutOffsetT>* counter,
    OffsetT* histogram,
    int pass)
  {
    const OutOffsetT current_k = counter->k;
    const OffsetT current_len  = counter->len;
    OffsetT previous_len       = counter->previous_len;

    // If current_len is 0, it means all the candidates have been found in previous passes.
    if (current_len == 0)
    {
      return;
    }

    // Early stop means that the bin containing the k-th element has been identified, and all
    // the elements in this bin are exactly the remaining k items we need to find. So we can
    // stop the process right here.
    const bool early_stop = (current_len == static_cast<OffsetT>(current_k));

    // If previous_len > buffer_length, it means we haven't started writing candidates to out_buf yet,
    // so have to make sure to load input directly from the original input.
    // Also, unless we've had the chance to do at least one filtering pass, our input is definitely the original input
    // (this is to guard against edge cases, e.g., buffer_length=num_items=1).
    const bool load_from_original_input = (pass <= 1) || previous_len > buffer_length;

    if (load_from_original_input)
    {
      in_idx_buf   = nullptr;
      previous_len = num_items;
    }

    // "current_len > buffer_length" means current pass will skip writing buffer
    if (current_len > buffer_length)
    {
      out_buf     = nullptr;
      out_idx_buf = nullptr;
    }

    // Fused filtering of candidates and histogram computation over the output-candidates
    filter_and_histogram(
      in_buf, in_idx_buf, out_buf, out_idx_buf, previous_len, counter, histogram, early_stop, load_from_original_input);

    // We need this `__threadfence()` to make sure all writes to the global memory-histogram are visible to all
    // threads before we proceed to compute the prefix sum over the histogram.
    __threadfence();

    // Identify the last block in the grid to perform the prefix sum over the histogram identify the bin that the
    // k-th item falls into
    bool is_last_block = false;
    if (threadIdx.x == 0)
    {
      unsigned int finished = atomicInc(&counter->finished_block_cnt, gridDim.x - 1);
      is_last_block         = (finished == (gridDim.x - 1));
    }

    // syncthreads ensures that the BlockLoad for loading the global histogram can reuse the temporary storage
    if (__syncthreads_or(is_last_block))
    {
      if (threadIdx.x == 0)
      {
        // If we have found the top-k items already, we can short-circuit subsequent passes
        if (early_stop)
        {
          // Signal subsequent passes to skip processing
          counter->previous_len = 0;
          counter->len          = 0;
        }
        else
        {
          // The number of output-candidates of the current pass become the input size of the next pass
          counter->previous_len = current_len;

          // Reset the counter used to coordinate writes to the output buffer
          // TODO (elstehle): This part can be skipped during the last pass.
          counter->filter_cnt = 0;
        }
      }

      // Compute prefix sum over the histogram's bin counts
      compute_bin_offsets(histogram);

      // Make sure the prefix sum has been written to shared memory before choose_bucket()
      __syncthreads();

      // Identify the bucket that the bin that the k-th item falls into
      choose_bucket(counter, current_k, pass);

      // Reset histogram for the next pass
      // TODO: Refactor calc_start_bit, calc_mask, and calc_num_passes to uniformly work with
      // total_bits (passed as a kernel parameter) instead of sizeof(KeyT), then use a single
      // unconditional path for both fundamental and non-fundamental types.
      if constexpr (detail::radix::can_twiddle<key_in_t>)
      {
        constexpr int num_passes = calc_num_passes<key_in_t>(bits_per_pass);
        if (pass != num_passes - 1)
        {
          init_histograms(histogram);
        }
      }
      else
      {
        init_histograms(histogram);
      }
    }
  }

  // Histogram-only pass: computes the histogram over the full input without filtering.
  // Used for the first radix pass before any candidates have been identified.
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  invoke_histogram_only(Counter<key_in_t, OffsetT, OutOffsetT>* counter, OffsetT* histogram, int pass)
  {
    // Initialize shared memory histogram
    init_histograms(temp_storage.histogram);
    __syncthreads();

    // Compute per-thread block histograms over the full input
    auto f = [this](key_in_t key, OffsetT /*index*/) {
      const int bucket = extract_bin_op(key);
      atomicAdd(temp_storage.histogram + bucket, OffsetT{1});
    };
    process_range(d_keys_in, num_items, f);

    // Ensure all threads have contributed to the histogram before accumulating in global memory
    __syncthreads();

    // Merge the locally aggregated histogram into the global histogram
    merge_histograms(histogram);

    // We need this `__threadfence()` to make sure all writes to the global memory-histogram are visible to all
    // threads before we proceed to compute the prefix sum over the histogram.
    __threadfence();

    // Identify the last block in the grid to perform the prefix sum over the histogram and identify the bin that
    // the k-th item falls into
    bool is_last_block = false;
    if (threadIdx.x == 0)
    {
      unsigned int finished = atomicInc(&counter->finished_block_cnt, gridDim.x - 1);
      is_last_block         = (finished == (gridDim.x - 1));
    }

    if (__syncthreads_or(is_last_block))
    {
      if (threadIdx.x == 0)
      {
        counter->previous_len = num_items;
        counter->filter_cnt   = 0;
      }

      // Compute prefix sum over the histogram's bin counts
      compute_bin_offsets(histogram);

      // Make sure the prefix sum has been written to shared memory before choose_bucket()
      __syncthreads();

      // Identify the bucket that the bin that the k-th item falls into
      choose_bucket(counter, k, pass);

      // Reset histogram for the next pass
      // TODO: Refactor calc_start_bit, calc_mask, and calc_num_passes to uniformly work with
      // total_bits (passed as a kernel parameter) instead of sizeof(KeyT), then use a single
      // unconditional path for both fundamental and non-fundamental types.
      if constexpr (detail::radix::can_twiddle<key_in_t>)
      {
        constexpr int num_passes = calc_num_passes<key_in_t>(bits_per_pass);
        if (pass != num_passes - 1)
        {
          init_histograms(histogram);
        }
      }
      else
      {
        init_histograms(histogram);
      }
    }
  }
};
} // namespace detail::topk
CUB_NAMESPACE_END
