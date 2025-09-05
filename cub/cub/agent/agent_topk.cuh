// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file
 * cub::AgentTopK implements a stateful abstraction of CUDA thread blocks for participating in device-wide topK.
 */

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
#include <cub/util_type.cuh>

#include <cuda/atomic>

CUB_NAMESPACE_BEGIN

/**
 * Parameterizable tuning policy type for AgentTopK
 *
 * @tparam BlockThreads
 *   Threads per thread block
 *
 * @tparam ItemsPerThread
 *   Items per thread (per tile of input)
 *
 * @tparam BitsPerPass
 *   Number of bits processed per pass
 *
 * @tparam CoefficientForBuffer
 *   The coefficient parameter for reducing the size of buffer.
 *   The size of buffer is `1 / CoefficientForBuffer` of original input
 *
 * @tparam LoadAlgorithm
 *   The BlockLoad algorithm to use
 *
 * @tparam ScanAlgorithm
 *   The BlockScan algorithm to use
 */

template <int BlockThreads,
          int ItemsPerThread,
          int BitsPerPass,
          BlockLoadAlgorithm LoadAlgorithm,
          BlockScanAlgorithm ScanAlgorithm>
struct AgentTopKPolicy
{
  /// Threads per thread block
  static constexpr int BLOCK_THREADS = BlockThreads;

  /// Items per thread (per tile of input)
  static constexpr int ITEMS_PER_THREAD = ItemsPerThread;

  /// Number of bits processed per pass
  static constexpr int BITS_PER_PASS = BitsPerPass;

  /// The BlockLoad algorithm to use
  static constexpr BlockLoadAlgorithm LOAD_ALGORITHM = LoadAlgorithm;

  /// The BlockScan algorithm to use
  static constexpr BlockScanAlgorithm SCAN_ALGORITHM = ScanAlgorithm;
};

namespace detail::topk
{
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
  typename Traits<KeyInT>::UnsignedBits kth_key_bits;

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

/**
 * Calculates the number of passes needed for a type T with BitsPerPass bits processed per pass.
 */
template <typename T, int BitsPerPass>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr int calc_num_passes()
{
  return ::cuda::ceil_div<int>(sizeof(T) * 8, BitsPerPass);
}

/**
 * Calculates the starting bit for a given pass (bit 0 is the least significant (rightmost) bit).
 * We process the input from the most to the least significant bit. This way, we can skip some passes in the end.
 */
template <typename T, int BitsPerPass>
_CCCL_DEVICE _CCCL_FORCEINLINE constexpr int calc_start_bit(const int pass)
{
  int start_bit = static_cast<int>(sizeof(T) * 8) - (pass + 1) * BitsPerPass;
  if (start_bit < 0)
  {
    start_bit = 0;
  }
  return start_bit;
}

/**
 * Used in the bin ID calculation to exclude bits unrelated to the current pass
 */
template <typename T, int BitsPerPass>
_CCCL_DEVICE constexpr unsigned calc_mask(const int pass)
{
  int num_bits = calc_start_bit<T, BitsPerPass>(pass - 1) - calc_start_bit<T, BitsPerPass>(pass);
  return (1 << num_bits) - 1;
}

/**
 * Get the bin ID from the value of element
 */
template <typename T, bool FlipBits, int BitsPerPass>
struct ExtractBinOp
{
  int pass{};
  int start_bit;
  unsigned mask;

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE ExtractBinOp(int pass)
      : pass(pass)
  {
    start_bit = calc_start_bit<T, BitsPerPass>(pass);
    mask      = calc_mask<T, BitsPerPass>(pass);
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int operator()(T key) const
  {
    auto bits = reinterpret_cast<typename Traits<T>::UnsignedBits&>(key);
    bits      = Traits<T>::TwiddleIn(bits);
    if constexpr (FlipBits)
    {
      bits = ~bits;
    }
    int bucket = (bits >> start_bit) & mask;
    return bucket;
  }
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

/**
 * Check if the input element is still a candidate for the target pass.
 */
template <typename T, bool FlipBits, int BitsPerPass>
struct IdentifyCandidatesOp
{
  typename Traits<T>::UnsignedBits* kth_key_bits;
  int pass;
  int start_bit;
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE IdentifyCandidatesOp(typename Traits<T>::UnsignedBits* kth_key_bits, int pass)
      : kth_key_bits(kth_key_bits)
      , pass(pass - 1)
  {
    start_bit = calc_start_bit<T, BitsPerPass>(this->pass);
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE candidate_class operator()(T key) const
  {
    auto bits = reinterpret_cast<typename Traits<T>::UnsignedBits&>(key);
    bits      = Traits<T>::TwiddleIn(bits);

    if constexpr (FlipBits)
    {
      bits = ~bits;
    }

    bits = (bits >> start_bit) << start_bit;

    return (bits < *kth_key_bits) ? candidate_class::selected
         : (bits == *kth_key_bits)
           ? candidate_class::candidate
           : candidate_class::rejected;
  }
};

/**
 * @brief AgentTopK implements a stateful abstraction of CUDA thread blocks for participating in
 * device-wide topK
 *
 * @tparam AgentTopKPolicyT
 *   Parameterized AgentTopKPolicy tuning policy type
 *
 * @tparam KeyInputIteratorT
 *   **[inferred]** Random-access input iterator type for reading input keys @iterator
 *
 * @tparam KeyOutputIteratorT
 *   **[inferred]** Random-access output iterator type for writing output keys @iterator
 *
 * @tparam ValueInputIteratorT
 *   **[inferred]** Random-access input iterator type for reading input values @iterator
 *
 * @tparam ValueOutputIteratorT
 *   **[inferred]** Random-access output iterator type for writing output values @iterator
 *
 * @tparam ExtractBinOpT
 *   Operations to extract the bin from the input key values
 *
 * @tparam IdentifyCandidatesOpT
 *   Operations to filter the input key values
 *
 * @tparam OffsetT
 *   Type of variable num_items
 *
 * @tparam OutOffsetT
 *   Type of variable k
 */
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
  using key_in_t = detail::it_value_t<KeyInputIteratorT>;

  static constexpr int BLOCK_THREADS    = AgentTopKPolicyT::BLOCK_THREADS;
  static constexpr int ITEMS_PER_THREAD = AgentTopKPolicyT::ITEMS_PER_THREAD;
  static constexpr int BITS_PER_PASS    = AgentTopKPolicyT::BITS_PER_PASS;
  static constexpr int TILE_ITEMS       = BLOCK_THREADS * ITEMS_PER_THREAD;
  static constexpr int num_buckets      = 1 << BITS_PER_PASS;

  static constexpr bool KEYS_ONLY      = ::cuda::std::is_same<ValueInputIteratorT, NullType>::value;
  static constexpr int bins_per_thread = ::cuda::ceil_div(num_buckets, BLOCK_THREADS);

  // Parameterized BlockLoad type for input data
  using block_load_input_t = BlockLoad<key_in_t, BLOCK_THREADS, ITEMS_PER_THREAD, AgentTopKPolicyT::LOAD_ALGORITHM>;
  using block_load_trans_t = BlockLoad<OffsetT, BLOCK_THREADS, bins_per_thread, BLOCK_LOAD_TRANSPOSE>;
  // Parameterized BlockScan type
  using block_scan_t = BlockScan<OffsetT, BLOCK_THREADS, AgentTopKPolicyT::SCAN_ALGORITHM>;
  // Parameterized BlockStore type
  using block_store_trans_t = BlockStore<OffsetT, BLOCK_THREADS, bins_per_thread, BLOCK_STORE_TRANSPOSE>;

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
    };
    OffsetT histogram[num_buckets];
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
  /**
   * @param temp_storage
   *   Reference to temp_storage
   *
   * @param d_keys_in
   *   Input data, keys
   *
   * @param d_keys_out
   *   Output data, keys
   *
   * @param d_values_in
   *   Input data, values
   *
   * @param d_values_out
   *   Output data, values
   *
   * @param num_items
   *   Total number of input items
   *
   * @param k
   *   The K value. Will find K elements from num_items elements
   *
   * @param buffer_length
   *   The size of the buffer for storing intermediate candidates
   *
   * @param extract_bin_op
   *   Extract bin operator
   *
   * @param identify_candidates_op
   *   Filter operator
   *
   */
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
  /**
   * Process a range of input data in tiles, calling f(key, index) for each element
   */
  template <typename InputItT, typename FuncT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void process_range(InputItT in, const OffsetT num_items, FuncT f)
  {
    key_in_t thread_data[ITEMS_PER_THREAD];

    const OffsetT ITEMS_PER_PASS   = TILE_ITEMS * gridDim.x;
    const OffsetT total_num_blocks = ::cuda::ceil_div(num_items, TILE_ITEMS);

    const OffsetT num_remaining_elements = num_items % TILE_ITEMS;
    const OffsetT last_block_id          = (total_num_blocks - 1) % gridDim.x;

    OffsetT tile_base = blockIdx.x * TILE_ITEMS;
    OffsetT offset    = threadIdx.x * ITEMS_PER_THREAD + tile_base;

    for (int i_block = blockIdx.x; i_block < total_num_blocks - 1; i_block += gridDim.x)
    {
      // Ensure that the temporary storage from previous iteration can be reused
      __syncthreads();

      block_load_input_t(temp_storage.load_input).Load(in + tile_base, thread_data);
      for (int j = 0; j < ITEMS_PER_THREAD; ++j)
      {
        f(thread_data[j], offset + j);
      }
      tile_base += ITEMS_PER_PASS;
      offset += ITEMS_PER_PASS;
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

      for (int j = 0; j < ITEMS_PER_THREAD; ++j)
      {
        if ((offset + j) < num_items)
        {
          f(thread_data[j], offset + j);
        }
      }
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void init_histograms(OffsetT* histogram)
  {
    // Initialize histogram bin counts to zeros
    int histo_offset = 0;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (; histo_offset + BLOCK_THREADS <= num_buckets; histo_offset += BLOCK_THREADS)
    {
      histogram[histo_offset + threadIdx.x] = 0;
    }
    // Finish up with guarded initialization if necessary
    if ((num_buckets % BLOCK_THREADS != 0) && (histo_offset + threadIdx.x < num_buckets))
    {
      histogram[histo_offset + threadIdx.x] = 0;
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void merge_histograms(OffsetT* local_histogram, OffsetT* global_histogram)
  {
    int histo_offset = 0;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (; histo_offset + BLOCK_THREADS <= num_buckets; histo_offset += BLOCK_THREADS)
    {
      if (temp_storage.histogram[histo_offset + threadIdx.x] != 0)
      {
        atomicAdd(global_histogram + (histo_offset + threadIdx.x), temp_storage.histogram[histo_offset + threadIdx.x]);
      }
    }

    // Finish up with guarded merging if necessary
    if ((num_buckets % BLOCK_THREADS != 0) && (histo_offset + threadIdx.x < num_buckets))
    {
      atomicAdd(global_histogram + (histo_offset + threadIdx.x), temp_storage.histogram[histo_offset + threadIdx.x]);
    }
  }

  /**
   * Fused filtering of the current pass and building histogram for the next pass
   */
  template <bool IsFirstPass>
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

    // Make sure the histogram was initialized
    __syncthreads();

    if constexpr (IsFirstPass)
    {
      // During the first pass, compute per-thread block histograms over the full input. The per-thread block histograms
      // are being added to the global histogram further down below.
      auto f = [this](key_in_t key, OffsetT index) {
        int bucket = extract_bin_op(key);
        atomicAdd(temp_storage.histogram + bucket, OffsetT{1});
      };
      process_range(d_keys_in, previous_len, f);
    }
    else
    {
      OffsetT* p_filter_cnt   = &counter->filter_cnt;
      OutOffsetT* p_out_cnt   = &counter->out_cnt;
      const auto kth_key_bits = counter->kth_key_bits;

      // Lambda for early_stop = true (i.e., we have identified the exact "splitter" key):
      // Select all items that fall into the bin of the k-th item (i.e., the 'candidates') and the ones that fall into
      // bins preceding the k-th item bin (i.e., 'selected' items), write them to output.
      // We can skip histogram computation because we don't need to further passes to refine the candidates.
      auto f_early_stop = [load_from_original_input, in_idx_buf, p_out_cnt, this](key_in_t key, OffsetT i) {
        candidate_class pre_res = identify_candidates_op(key);
        if (pre_res == candidate_class::candidate || pre_res == candidate_class::selected)
        {
          OutOffsetT pos  = atomicAdd(p_out_cnt, OutOffsetT{1});
          d_keys_out[pos] = key;
          if constexpr (!KEYS_ONLY)
          {
            OffsetT index     = load_from_original_input ? i : in_idx_buf[i];
            d_values_out[pos] = d_values_in[index];
          }
        }
      };

      // Lambda for early_stop = false, out_buf != nullptr (i.e., we need to further refine the candidates in the next
      // pass): Write out selected items to output, write candidates to out_buf, and build histogram for candidates.
      auto f_with_out_buf = [load_from_original_input, in_idx_buf, out_buf, out_idx_buf, p_filter_cnt, p_out_cnt, this](
                              key_in_t key, OffsetT i) {
        candidate_class pre_res = identify_candidates_op(key);
        if (pre_res == candidate_class::candidate)
        {
          OffsetT pos  = atomicAdd(p_filter_cnt, OffsetT{1});
          out_buf[pos] = key;
          if constexpr (!KEYS_ONLY)
          {
            OffsetT index    = load_from_original_input ? i : in_idx_buf[i];
            out_idx_buf[pos] = index;
          }

          int bucket = extract_bin_op(key);
          atomicAdd(temp_storage.histogram + bucket, OffsetT{1});
        }
        else if (pre_res == candidate_class::selected)
        {
          OutOffsetT pos  = atomicAdd(p_out_cnt, OutOffsetT{1});
          d_keys_out[pos] = key;
          if constexpr (!KEYS_ONLY)
          {
            OffsetT index     = in_idx_buf ? in_idx_buf[i] : i;
            d_values_out[pos] = d_values_in[index];
          }
        }
      };

      // Lambda for early_stop = false, out_buf = nullptr (i.e., we need to further refine the candidates in the next
      // pass, but we skip writing candidates to out_buf):
      // Just build histogram for candidates.
      // Note: We will only begin writing to d_keys_out starting from the pass in which the number of output-candidates
      // is small enough to fit into the output buffer (otherwise, we would be writing the same items to d_keys_out
      // multiple times).
      auto f_no_out_buf = [this](key_in_t key, OffsetT i) {
        candidate_class pre_res = identify_candidates_op(key);
        if (pre_res == candidate_class::candidate)
        {
          int bucket = extract_bin_op(key);
          atomicAdd(temp_storage.histogram + bucket, OffsetT{1});
        }
      };

      // Choose and invoke the appropriate lambda with the correct input source
      // If the input size exceeds the allocated buffer size, we know for sure we haven't started writing candidates to
      // the output buffer yet
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
    merge_histograms(temp_storage.histogram, histogram);
  }

  /**
   * Replace histogram with its own prefix sum
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void compute_bin_offsets(volatile OffsetT* histogram)
  {
    OffsetT thread_data[bins_per_thread];

    // Load global histogram (we can skip initializing oob-items to zero because they won't be stored back)
    block_load_trans_t(temp_storage.load_trans).Load(histogram, thread_data, num_buckets);
    __syncthreads();

    block_scan_t(temp_storage.scan).InclusiveSum(thread_data, thread_data);
    __syncthreads();

    block_store_trans_t(temp_storage.store_trans).Store(temp_storage.histogram, thread_data, num_buckets);
  }

  /**
   * Identify the bucket that the k-th value falls into
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  choose_bucket(Counter<key_in_t, OffsetT, OutOffsetT>* counter, const OutOffsetT k, const int pass)
  {
    // Initialize histogram bin counts to zeros
    int histo_offset = 0;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (; histo_offset + BLOCK_THREADS <= num_buckets; histo_offset += BLOCK_THREADS)
    {
      const int bin_idx = histo_offset + threadIdx.x;
      OffsetT prev      = (bin_idx == 0) ? 0 : temp_storage.histogram[bin_idx - 1];
      OffsetT cur       = temp_storage.histogram[bin_idx];

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
        int start_bit = calc_start_bit<key_in_t, BITS_PER_PASS>(pass);
        counter->kth_key_bits |= bucket << start_bit;
      }
    }
    // Finish up with guarded initialization if necessary
    if ((num_buckets % BLOCK_THREADS != 0) && (histo_offset + threadIdx.x < num_buckets))
    {
      const int bin_idx = histo_offset + threadIdx.x;
      OffsetT prev      = (bin_idx == 0) ? 0 : temp_storage.histogram[bin_idx - 1];
      OffsetT cur       = temp_storage.histogram[bin_idx];

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
        int start_bit = calc_start_bit<key_in_t, BITS_PER_PASS>(pass);
        counter->kth_key_bits |= bucket << start_bit;
      }
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void invoke_last_filter(
    key_in_t* in_buf,
    OffsetT* in_idx_buf,
    Counter<key_in_t, OffsetT, OutOffsetT>* counter,
    OffsetT* histogram,
    OutOffsetT k,
    int pass)
  {
    const bool load_from_original_input = (pass <= 1) || counter->previous_len > buffer_length;
    OffsetT current_len                 = load_from_original_input ? num_items : counter->previous_len;
    in_idx_buf = load_from_original_input ? nullptr : in_idx_buf; // ? out_idx_buf : in_idx_buf;

    if (current_len == 0)
    {
      return;
    }

    // changed in choose_bucket(); need to reload
    OffsetT num_of_kth_needed  = counter->k;
    OutOffsetT* p_out_cnt      = &counter->out_cnt;
    OutOffsetT* p_out_back_cnt = &counter->out_back_cnt;

    auto f = [this, p_out_cnt, counter, in_idx_buf, p_out_back_cnt, num_of_kth_needed, k, current_len](
               key_in_t key, OffsetT i) {
      candidate_class res = identify_candidates_op(key);
      if (res == candidate_class::selected)
      {
        OutOffsetT pos  = atomicAdd(p_out_cnt, OffsetT{1});
        d_keys_out[pos] = key;
        if constexpr (!KEYS_ONLY)
        {
          // If writing has been skipped up to this point, `in_idx_buf` is nullptr
          OffsetT index     = in_idx_buf ? in_idx_buf[i] : i;
          d_values_out[pos] = d_values_in[index];
        }
      }
      else if (res == candidate_class::candidate)
      {
        OffsetT new_idx     = in_idx_buf ? in_idx_buf[i] : i;
        OutOffsetT back_pos = atomicAdd(p_out_back_cnt, OffsetT{1});

        if (back_pos < num_of_kth_needed)
        {
          OutOffsetT pos  = k - 1 - back_pos;
          d_keys_out[pos] = key;
          if constexpr (!KEYS_ONLY)
          {
            d_values_out[pos] = d_values_in[new_idx];
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

  template <bool IsFirstPass>
  _CCCL_DEVICE _CCCL_FORCEINLINE void invoke_filter_and_histogram(
    key_in_t* in_buf,
    OffsetT* in_idx_buf,
    key_in_t* out_buf,
    OffsetT* out_idx_buf,
    Counter<key_in_t, OffsetT, OutOffsetT>* counter,
    OffsetT* histogram,
    int pass)
  {
    OutOffsetT current_k;
    OffsetT previous_len;
    OffsetT current_len;

    if constexpr (IsFirstPass)
    {
      current_k    = k;
      previous_len = num_items;
      current_len  = num_items;
    }
    else
    {
      current_k    = counter->k;
      current_len  = counter->len;
      previous_len = counter->previous_len;
    }

    // If current_len is 0, it means all the candidates have been found in previous passes.
    if (current_len == 0)
    {
      return;
    }

    // Early stop means that the bin containing the k-th element has been identified, and all
    // the elements in this bin are exactly the remaining k items we need to find. So we can
    // stop the process right here.
    const bool early_stop = ((!IsFirstPass) && current_len == static_cast<OffsetT>(current_k));

    // If previous_len > buffer_length, it means we haven't started writing candidates to out_buf yet,
    // so have to make sure to load input directly from the original input.
    // Also, unless we've had the chance to do at least one filtering pass, our input is definitely the original input
    // (this is to guard against edge cases, e.g., buffer_length=num_items=1).
    bool load_from_original_input = (pass <= 1) || previous_len > buffer_length;

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
    filter_and_histogram<IsFirstPass>(
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
      constexpr int num_passes = calc_num_passes<key_in_t, BITS_PER_PASS>();
      if (pass != num_passes - 1)
      {
        init_histograms(histogram);
      }
    }
  }
};

} // namespace detail::topk
CUB_NAMESPACE_END
