// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * @file
 *   cub::DeviceTopK provides device-wide, parallel operations for finding the K largest (or smallest) items
 * from sequences of unordered data items residing within device-accessible memory.
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

#include <cub/agent/agent_topk.cuh>
#include <cub/block/block_load.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_temporary_storage.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/cmath>

CUB_NAMESPACE_BEGIN

namespace detail::topk
{

template <class KeyT>
constexpr int calc_bits_per_pass()
{
  return sizeof(KeyT) == 1 ? 8 : sizeof(KeyT) == 2 ? 11 : sizeof(KeyT) == 4 ? 11 : sizeof(KeyT) == 8 ? 11 : 8;
}

template <class KeyInT>
struct sm90_tuning
{
  static constexpr int threads = 512; // Number of threads per block

  static constexpr int nominal_4b_items_per_thread = 4;
  static constexpr int items = ::cuda::std::max(1, (nominal_4b_items_per_thread * 4 / (int) sizeof(KeyInT)));
  // Try to load 16 Bytes per thread. (int64(items=2);int32(items=4);int16(items=8)).

  static constexpr int BITS_PER_PASS = detail::topk::calc_bits_per_pass<KeyInT>();

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_VECTORIZE;
};

template <class KeyInT, class OffsetT>
struct device_topk_policy_hub
{
  struct DefaultTuning
  {
    static constexpr int NOMINAL_4B_ITEMS_PER_THREAD = 4;
    static constexpr int ITEMS_PER_THREAD            = ::cuda::std::min(
      NOMINAL_4B_ITEMS_PER_THREAD, ::cuda::std::max(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / (int) sizeof(KeyInT))));

    static constexpr int BITS_PER_PASS = detail::topk::calc_bits_per_pass<KeyInT>();

    using topk_policy_t =
      AgentTopKPolicy<512, ITEMS_PER_THREAD, BITS_PER_PASS, BLOCK_LOAD_VECTORIZE, BLOCK_SCAN_WARP_SCANS>;
  };

  struct Policy350
      : DefaultTuning
      , ChainedPolicy<350, Policy350, Policy350>
  {};

  struct Policy900 : ChainedPolicy<900, Policy900, Policy350>
  {
    using tuning = detail::topk::sm90_tuning<KeyInT>;
    using topk_policy_t =
      AgentTopKPolicy<tuning::threads, tuning::items, tuning::BITS_PER_PASS, tuning::load_algorithm, BLOCK_SCAN_WARP_SCANS>;
  };

  using max_policy = Policy900;
};

/******************************************************************************
 * Kernel entry points
 *****************************************************************************/
/**
 * TopK kernel entry point (multi-block) for histogram collection, prefix sum and filering operations except for the
 * last round
 *
 * Find the largest (or smallest) K items from a sequence of unordered data
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
 * @tparam OffsetT
 *  Data Type for variables: num_items
 *
 * @tparam OutOffsetT
 *  Data Type for variables: k
 *
 * @tparam KeyInT
 *  Data Type for input keys
 *
 * @tparam ExtractBinOpT
 *   Operations to extract the bin from the input key value
 *
 * @tparam IdentifyCandidatesOpT
 *   Operations to filter the input key value
 *
 * @tparam SelectMin
 *   Determine whether to select the smallest (SelectMin=true) or largest (SelectMin=false) K elements.
 *
 * @param[in] d_keys_in
 *   Pointer to the input data of key data
 *
 * @param[out] d_keys_out
 *   Pointer to the K output sequence of key data
 *
 * @param[in] d_values_in
 *   Pointer to the input sequence of associated value items
 *
 * @param[out] d_values_out
 *   Pointer to the output sequence of associated value items
 *
 * @param[in] in_buf
 *   Pointer to buffer of input key data
 *
 * @param[out] out_buf
 *   Pointer to buffer of output key data
 *
 * @param[in] in_idx_buf
 *   Pointer to buffer of index of input buffer
 *
 * @param[out] out_idx_buf
 *   Pointer to buffer of index of output
 *
 * @param[in] counter
 *   Pointer to buffer of counter array
 *
 * @param[in] histogram
 *   Pointer to buffer of histogram array
 *
 * @param[in] num_items
 *   Number of items to be processed
 *
 * @param[in] k
 *   The K value. Will find K elements from num_items elements. The variable K should be smaller than the variable N.
 *
 * @param[in] buffer_length
 *   The size of the buffer for storing intermediate candidates
 *
 * @param[in] extract_bin_op
 *   Extract the bin operator
 *
 * @param[in] identify_candidates_op
 *   Extract element filter operator
 *
 * @param[in] pass
 *   The index of the passes
 */
template <typename ChainedPolicyT,
          typename KeyInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueInputIteratorT,
          typename ValueOutputIteratorT,
          typename OffsetT,
          typename OutOffsetT,
          typename KeyInT,
          typename ExtractBinOpT,
          typename IdentifyCandidatesOpT,
          bool SelectMin,
          bool IsFirstPass>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::topk_policy_t::BLOCK_THREADS))
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceTopKKernel(
    const KeyInputIteratorT d_keys_in,
    KeyOutputIteratorT d_keys_out,
    const ValueInputIteratorT d_values_in,
    ValueOutputIteratorT d_values_out,
    KeyInT* in_buf,
    OffsetT* in_idx_buf,
    KeyInT* out_buf,
    OffsetT* out_idx_buf,
    Counter<detail::it_value_t<KeyInputIteratorT>, OffsetT, OutOffsetT>* counter,
    OffsetT* histogram,
    OffsetT num_items,
    OutOffsetT k,
    OffsetT buffer_length,
    ExtractBinOpT extract_bin_op,
    IdentifyCandidatesOpT identify_candidates_op,
    int pass)
{
  using agent_topk_policy_t = typename ChainedPolicyT::ActivePolicy::topk_policy_t;
  using agent_topk_t =
    AgentTopK<agent_topk_policy_t,
              KeyInputIteratorT,
              KeyOutputIteratorT,
              ValueInputIteratorT,
              ValueOutputIteratorT,
              ExtractBinOpT,
              IdentifyCandidatesOpT,
              OffsetT,
              OutOffsetT>;

  // Shared memory storage
  __shared__ typename agent_topk_t::TempStorage temp_storage;
  agent_topk_t(
    temp_storage,
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    num_items,
    k,
    buffer_length,
    extract_bin_op,
    identify_candidates_op)
    .invoke_filter_and_histogram<IsFirstPass>(in_buf, in_idx_buf, out_buf, out_idx_buf, counter, histogram, pass);
}

/**
 * TopK kernel entry point for the last filtering step (multi-block)
 *
 * Find the largest (or smallest) K items from a sequence of unordered data
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
 * @tparam OffsetT
 *  Data Type for variables: num_items
 *
 * @tparam OutOffsetT
 *  Data Type for variables: k
 *
 * @tparam KeyInT
 *  Data Type for input keys
 *
 * @tparam IdentifyCandidatesOpT
 *   Operations to filter the input key value
 *
 * @tparam SelectMin
 *   Determine whether to select the smallest (SelectMin=true) or largest (SelectMin=false) K elements.
 *
 * @param[in] d_keys_in
 *   Pointer to the input data of key data
 *
 * @param[out] d_keys_out
 *   Pointer to the K output sequence of key data
 *
 * @param[in] d_values_in
 *   Pointer to the input sequence of associated value items
 *
 * @param[out] d_values_out
 *   Pointer to the output sequence of associated value items
 *
 * @param[in] in_buf
 *   Pointer to buffer of input key data
 *
 * @param[in] in_idx_buf
 *   Pointer to buffer of index of input buffer
 *
 * @param[in] counter
 *   Pointer to buffer of counter array
 *
 * @param[in] histogram
 *   Pointer to buffer of histogram array
 *
 * @param[in] num_items
 *   Number of items to be processed
 *
 * @param[in] k
 *   The K value. Will find K elements from num_items elements. The variable K should be smaller than the variable N.
 *
 * @param[in] buffer_length
 *   The size of the buffer for storing intermediate candidates
 *
 * @param[in] identify_candidates_op
 *   Extract element filter operator
 *
 * @param[in] pass
 *   The index of the passes
 */
template <typename ChainedPolicyT,
          typename KeyInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueInputIteratorT,
          typename ValueOutputIteratorT,
          typename OffsetT,
          typename OutOffsetT,
          typename KeyInT,
          typename IdentifyCandidatesOpT,
          bool SelectMin>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::topk_policy_t::BLOCK_THREADS))
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceTopKLastFilterKernel(
    const KeyInputIteratorT d_keys_in,
    KeyOutputIteratorT d_keys_out,
    const ValueInputIteratorT d_values_in,
    ValueOutputIteratorT d_values_out,
    KeyInT* in_buf,
    OffsetT* in_idx_buf,
    Counter<detail::it_value_t<KeyInputIteratorT>, OffsetT, OutOffsetT>* counter,
    OffsetT* histogram,
    OffsetT num_items,
    OutOffsetT k,
    OffsetT buffer_length,
    IdentifyCandidatesOpT identify_candidates_op,
    int pass)
{
  using agent_topk_policy_t = typename ChainedPolicyT::ActivePolicy::topk_policy_t;
  using extract_bin_op_t    = NullType;
  using agent_topk_t =
    AgentTopK<agent_topk_policy_t,
              KeyInputIteratorT,
              KeyOutputIteratorT,
              ValueInputIteratorT,
              ValueOutputIteratorT,
              extract_bin_op_t, // ExtractBinOp operator (not used)
              IdentifyCandidatesOpT,
              OffsetT,
              OutOffsetT>;

  // Shared memory storage
  __shared__ typename agent_topk_t::TempStorage temp_storage;
  agent_topk_t(
    temp_storage,
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    num_items,
    k,
    buffer_length,
    extract_bin_op_t{},
    identify_candidates_op)
    .invoke_last_filter(in_buf, in_idx_buf, counter, histogram, k);
}

/*
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
 *   **[inferred]** Random-access input iterator type for writing output values @iterator
 *
 * @tparam OffsetT
 *  Data Type for variables: num_items
 *
 * @tparam OutOffsetT
 *  Data Type for variables: k
 *
 * @tparam SelectMin
 *   Determine whether to select the smallest (SelectMin=true) or largest (SelectMin=false) K elements.
 */
template <typename KeyInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueInputIteratorT,
          typename ValueOutputIteratorT,
          typename OffsetT,
          typename OutOffsetT,
          bool SelectMin,
          typename SelectedPolicy = detail::topk::device_topk_policy_hub<detail::it_value_t<KeyInputIteratorT>, OffsetT>>
struct DispatchTopK : SelectedPolicy
{
  // atomicAdd does not implement overloads for all integer types, so we limit OffsetT to uint32_t or unsigned long long
  static_assert(::cuda::std::is_same_v<OffsetT, ::cuda::std::uint32_t>
                  || ::cuda::std::is_same_v<OffsetT, unsigned long long>,
                "The top-k algorithm is limited to unsigned offset types retrieved from choose_offset_t<T>.");

  // atomicAdd does not implement overloads for all integer types, so we limit OffsetT to uint32_t or unsigned long long
  static_assert(::cuda::std::is_same_v<OutOffsetT, ::cuda::std::uint32_t>
                  || ::cuda::std::is_same_v<OutOffsetT, unsigned long long>,
                "The top-k algorithm is limited to unsigned offset types retrieved from choose_offset_t<T>.");

  // TODO (elstehle): consider making this part of the env-based API
  // The algorithm allocates a double-buffer for intermediate results of size num_items/coefficient_for_candidate_buffer
  static constexpr OffsetT coefficient_for_candidate_buffer = 128;

  /// Device-accessible allocation of temporary storage.
  /// When `nullptr`, the required allocation size is written to `temp_storage_bytes` and no work is done.
  void* d_temp_storage;

  /// Reference to size in bytes of `d_temp_storage` allocation
  size_t& temp_storage_bytes;

  /// Pointer to the input sequence of data items
  KeyInputIteratorT d_keys_in;

  /// Pointer to the K output sequence of key data
  KeyOutputIteratorT d_keys_out;

  /// Pointer to the input sequence of associated value items
  ValueInputIteratorT d_values_in;

  /// Pointer to the output sequence of associated value items
  ValueOutputIteratorT d_values_out;

  /// Number of items to be processed
  OffsetT num_items;

  /// The K value. Will find K elements from num_items elements
  OutOffsetT k;

  /// CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
  cudaStream_t stream;

  int ptx_version;

  using key_in_t                  = detail::it_value_t<KeyInputIteratorT>;
  static constexpr bool keys_only = ::cuda::std::is_same<ValueInputIteratorT, NullType>::value;
  /*
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to `temp_storage_bytes` and no work is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_keys_in
   *   Pointer to the input data of key data
   *
   * @param[out] d_keys_out
   *   Pointer to the K output sequence of key data
   *
   * @param[in] d_values_in
   *   Pointer to the input sequence of associated value items
   *
   * @param[out] d_values_out
   *   Pointer to the output sequence of associated value items
   *
   * @param[in] num_items
   *   Number of items to be processed
   *
   * @param[in] k
   *   The K value. Will find K elements from num_items elements. If K exceeds `num_items`, K is capped at a maximum of
   * `num_items`.
   *
   * @param[in] stream
   *   @rst
   *   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
   *   @endrst
   */
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchTopK(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    const KeyInputIteratorT d_keys_in,
    KeyOutputIteratorT d_keys_out,
    const ValueInputIteratorT d_values_in,
    ValueOutputIteratorT d_values_out,
    OffsetT num_items,
    OutOffsetT k,
    cudaStream_t stream,
    int ptx_version)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_keys_in(d_keys_in)
      , d_keys_out(d_keys_out)
      , d_values_in(d_values_in)
      , d_values_out(d_values_out)
      , num_items(num_items)
      , k(k)
      , stream(stream)
      , ptx_version(ptx_version)
  {}

  /******************************************************************************
   * Dispatch entrypoints
   ******************************************************************************/
  template <typename TopKKernelPtrT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE int calculate_blocks_per_sm(TopKKernelPtrT topk_kernel, int block_threads)
  {
    int topk_blocks_per_sm;
    cudaError error;
    error = CubDebug(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&topk_blocks_per_sm, topk_kernel, block_threads, 0));
    if (cudaSuccess != error)
    {
      return error;
    }
    return topk_blocks_per_sm;
  }

  template <typename ActivePolicyT,
            typename TopKFirstPassKernelPtrT,
            typename TopKKernelPtrT,
            typename TopKLastFilterKernelPtrT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
  Invoke(TopKFirstPassKernelPtrT topk_first_pass_kernel,
         TopKKernelPtrT topk_kernel,
         TopKLastFilterKernelPtrT topk_last_filter_kernel)
  {
    using policy_t = typename ActivePolicyT::topk_policy_t;

    cudaError error = cudaSuccess;

    constexpr int block_threads    = policy_t::BLOCK_THREADS; // Threads per block
    constexpr int items_per_thread = policy_t::ITEMS_PER_THREAD; // Items per thread
    constexpr int tile_size        = block_threads * items_per_thread; // Items per block
    int num_tiles                  = static_cast<int>(::cuda::ceil_div(num_items, tile_size)); // Num of blocks
    constexpr int num_passes       = calc_num_passes<key_in_t, policy_t::BITS_PER_PASS>();
    constexpr int num_buckets      = 1 << policy_t::BITS_PER_PASS;

    // We are capping k at a maximum of num_items
    using common_offset_t = ::cuda::std::common_type_t<OffsetT, OutOffsetT>;
    k                     = static_cast<OutOffsetT>(
      ::cuda::std::clamp(static_cast<common_offset_t>(k), common_offset_t{k}, static_cast<common_offset_t>(num_items)));

    // Specify temporary storage allocation requirements
    size_t size_counter             = sizeof(Counter<key_in_t, OffsetT, OutOffsetT>);
    size_t size_histogram           = num_buckets * sizeof(OffsetT);
    OffsetT candidate_buffer_length = ::cuda::std::max(OffsetT{1}, num_items / coefficient_for_candidate_buffer);

    size_t allocation_sizes[6] = {
      size_counter,
      size_histogram,
      candidate_buffer_length * sizeof(key_in_t),
      candidate_buffer_length * sizeof(key_in_t),
      keys_only ? 0 : candidate_buffer_length * sizeof(OffsetT),
      keys_only ? 0 : candidate_buffer_length * sizeof(OffsetT)};

    // Compute allocation pointers into the single storage blob (or compute the necessary size of the blob)
    void* allocations[6] = {};

    error = CubDebug(detail::AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes));
    if (cudaSuccess != error)
    {
      return error;
    }

    if (d_temp_storage == nullptr)
    {
      // Return if the caller is simply requesting the size of the storage allocation
      return cudaSuccess;
    }

    // Init the buffer for descriptor and histogram
    error = CubDebug(cudaMemsetAsync(
      allocations[0], 0, static_cast<char*>(allocations[2]) - static_cast<char*>(allocations[0]), stream));
    if (cudaSuccess != error)
    {
      return error;
    }

    // Get grid size for scanning tiles
    int device  = -1;
    int num_sms = 0;

    error = CubDebug(cudaGetDevice(&device));
    if (cudaSuccess != error)
    {
      return error;
    }
    error = CubDebug(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
    if (cudaSuccess != error)
    {
      return error;
    }

    const auto main_kernel_blocks_per_sm = calculate_blocks_per_sm(topk_kernel, block_threads);
    const auto main_kernel_max_occupancy = static_cast<unsigned int>(main_kernel_blocks_per_sm * num_sms);
    const auto topk_grid_size =
      ::cuda::std::min(main_kernel_max_occupancy, static_cast<unsigned int>(::cuda::ceil_div(num_items, tile_size)));

// Log topk_kernel configuration @todo check the kernel launch
#ifdef CUB_DEBUG_LOG
    {
      // Get SM occupancy for select_if_kernel
      if (cudaSuccess != error)
      {
        return error;
      }

      _CubLog("Invoking topk_kernel<<<{%d,%d,%d}, %d, 0, "
              "%lld>>>(), %d items per thread, %d SM occupancy\n",
              topk_grid_size.x,
              topk_grid_size.y,
              topk_grid_size.z,
              block_threads,
              (long long) stream,
              items_per_thread,
              topk_blocks_per_sm);
    }
#endif // CUB_DEBUG_LOG

    // Initialize address variables
    Counter<key_in_t, OffsetT, OutOffsetT>* counter;
    counter = static_cast<decltype(counter)>(allocations[0]);
    OffsetT* histogram;
    histogram            = static_cast<decltype(histogram)>(allocations[1]);
    key_in_t* in_buf     = nullptr;
    key_in_t* out_buf    = nullptr;
    OffsetT* in_idx_buf  = nullptr;
    OffsetT* out_idx_buf = nullptr;
    int pass             = 0;
    for (; pass < num_passes; pass++)
    {
      // Set operator
      ExtractBinOp<key_in_t, !SelectMin, policy_t::BITS_PER_PASS> extract_bin_op(pass);
      IdentifyCandidatesOp<key_in_t, !SelectMin, policy_t::BITS_PER_PASS> identify_candidates_op(
        &counter->kth_key_bits, pass);

      // Initialize address variables
      in_buf  = static_cast<key_in_t*>(pass % 2 == 0 ? allocations[2] : allocations[3]);
      out_buf = pass == 0 ? nullptr : static_cast<key_in_t*>(pass % 2 == 0 ? allocations[3] : allocations[2]);
      if (!keys_only)
      {
        in_idx_buf  = pass <= 1 ? nullptr : static_cast<OffsetT*>(pass % 2 == 0 ? allocations[4] : allocations[5]);
        out_idx_buf = pass == 0 ? nullptr : static_cast<OffsetT*>(pass % 2 == 0 ? allocations[5] : allocations[4]);
      }

      // Invoke kernel
      if (pass == 0)
      {
        // Compute grid size for the histogram kernel of the first pass
        const auto first_pass_kernel_blocks_per_sm = calculate_blocks_per_sm(topk_first_pass_kernel, block_threads);
        const auto first_pass_kernel_max_occupancy =
          static_cast<unsigned int>(first_pass_kernel_blocks_per_sm * num_sms);
        const auto topk_first_pass_grid_size = ::cuda::std::min(
          first_pass_kernel_max_occupancy, static_cast<unsigned int>(::cuda::ceil_div(num_items, tile_size)));

        // Compute histogram of the first pass
        THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(topk_first_pass_grid_size, block_threads, 0, stream)
          .doit(
            topk_first_pass_kernel,
            d_keys_in,
            d_keys_out,
            d_values_in,
            d_values_out,
            in_buf,
            in_idx_buf,
            out_buf,
            out_idx_buf,
            counter,
            histogram,
            num_items,
            k,
            candidate_buffer_length,
            extract_bin_op,
            identify_candidates_op,
            pass);
      }
      else
      {
        THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(topk_grid_size, block_threads, 0, stream)
          .doit(
            topk_kernel,
            d_keys_in,
            d_keys_out,
            d_values_in,
            d_values_out,
            in_buf,
            in_idx_buf,
            out_buf,
            out_idx_buf,
            counter,
            histogram,
            num_items,
            k,
            candidate_buffer_length,
            extract_bin_op,
            identify_candidates_op,
            pass);
      }
    }

    IdentifyCandidatesOp<key_in_t, !SelectMin, policy_t::BITS_PER_PASS> identify_candidates_op(
      &counter->kth_key_bits, pass);
    const auto last_filter_kernel_blocks_per_sm = calculate_blocks_per_sm(topk_kernel, block_threads);
    const auto last_filter_kernel_max_occupancy = static_cast<unsigned int>(last_filter_kernel_blocks_per_sm * num_sms);
    const auto last_filter_grid_size            = ::cuda::std::min(
      last_filter_kernel_max_occupancy, static_cast<unsigned int>(::cuda::ceil_div(num_items, tile_size)));
    THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(last_filter_grid_size, block_threads, 0, stream)
      .doit(topk_last_filter_kernel,
            d_keys_in,
            d_keys_out,
            d_values_in,
            d_values_out,
            out_buf,
            out_idx_buf,
            counter,
            histogram,
            num_items,
            k,
            candidate_buffer_length,
            identify_candidates_op,
            pass);

    // pass==num_passes to align with the usage of identify_candidates_op in previous passes.
    return error;
  }
  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    using max_policy_t = typename SelectedPolicy::max_policy;
    return Invoke<ActivePolicyT>(
      detail::topk::DeviceTopKKernel<
        max_policy_t,
        KeyInputIteratorT,
        KeyOutputIteratorT,
        ValueInputIteratorT,
        ValueOutputIteratorT,
        OffsetT,
        OutOffsetT,
        key_in_t,
        ExtractBinOp<key_in_t, !SelectMin, ActivePolicyT::topk_policy_t::BITS_PER_PASS>,
        IdentifyCandidatesOp<key_in_t, !SelectMin, ActivePolicyT::topk_policy_t::BITS_PER_PASS>,
        SelectMin,
        /*IsFirstPass*/ true>,

      detail::topk::DeviceTopKKernel<
        max_policy_t,
        KeyInputIteratorT,
        KeyOutputIteratorT,
        ValueInputIteratorT,
        ValueOutputIteratorT,
        OffsetT,
        OutOffsetT,
        key_in_t,
        ExtractBinOp<key_in_t, !SelectMin, ActivePolicyT::topk_policy_t::BITS_PER_PASS>,
        IdentifyCandidatesOp<key_in_t, !SelectMin, ActivePolicyT::topk_policy_t::BITS_PER_PASS>,
        SelectMin,
        /*IsFirstPass*/ false>,

      detail::topk::DeviceTopKLastFilterKernel<
        max_policy_t,
        KeyInputIteratorT,
        KeyOutputIteratorT,
        ValueInputIteratorT,
        ValueOutputIteratorT,
        OffsetT,
        OutOffsetT,
        key_in_t,
        IdentifyCandidatesOp<key_in_t, !SelectMin, ActivePolicyT::topk_policy_t::BITS_PER_PASS>,
        SelectMin>);
  }

  /*
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to `temp_storage_bytes` and no work is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_keys_in
   *   Pointer to the input data of key data to find top K
   *
   * @param[out] d_keys_out
   *   Pointer to the K output sequence of key data
   *
   * @param[in] d_values_in
   *   Pointer to the input sequence of associated value items
   *
   * @param[out] d_values_out
   *   Pointer to the output sequence of associated value items
   *
   * @param[in] num_items
   *   Number of items to be processed
   *
   * @param[in] k
   *   The K value. Will find K elements from num_items elements. If K exceeds `num_items`, K is capped at a maximum of
   * `num_items`.
   *
   * @param[in] stream
   *   @rst
   *   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
   *   @endrst
   */
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    const KeyInputIteratorT d_keys_in,
    KeyOutputIteratorT d_keys_out,
    const ValueInputIteratorT d_values_in,
    ValueOutputIteratorT d_values_out,
    OffsetT num_items,
    OutOffsetT k,
    cudaStream_t stream)
  {
    using max_policy_t = typename SelectedPolicy::max_policy;

    int ptx_version = 0;
    if (cudaError_t error = CubDebug(PtxVersion(ptx_version)))
    {
      return error;
    }

    DispatchTopK dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_keys_out,
      d_values_in,
      d_values_out,
      num_items,
      k,
      stream,
      ptx_version);

    return CubDebug(max_policy_t::Invoke(ptx_version, dispatch));
  }
};

} // namespace detail::topk

CUB_NAMESPACE_END
