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

  static constexpr int BITS_PER_PASS          = detail::topk::calc_bits_per_pass<KeyInT>();
  static constexpr int COEFFICIENT_FOR_BUFFER = 128;

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

    static constexpr int BITS_PER_PASS          = detail::topk::calc_bits_per_pass<KeyInT>();
    static constexpr int COEFFICIENT_FOR_BUFFER = 128;

    using topk_policy_t =
      AgentTopKPolicy<512,
                      ITEMS_PER_THREAD,
                      BITS_PER_PASS,
                      COEFFICIENT_FOR_BUFFER,
                      BLOCK_LOAD_VECTORIZE,
                      BLOCK_SCAN_WARP_SCANS>;
  };

  struct Policy350
      : DefaultTuning
      , ChainedPolicy<350, Policy350, Policy350>
  {};

  struct Policy900 : ChainedPolicy<900, Policy900, Policy350>
  {
    using tuning = detail::topk::sm90_tuning<KeyInT>;
    using topk_policy_t =
      AgentTopKPolicy<tuning::threads,
                      tuning::items,
                      tuning::BITS_PER_PASS,
                      tuning::COEFFICIENT_FOR_BUFFER,
                      tuning::load_algorithm,
                      BLOCK_SCAN_WARP_SCANS>;
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
              OutOffsetT,
              SelectMin>;

  // Shared memory storage
  __shared__ typename agent_topk_t::TempStorage temp_storage;
  agent_topk_t(
    temp_storage, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, k, extract_bin_op, identify_candidates_op)
    .InvokeOneSweep<IsFirstPass>(in_buf, in_idx_buf, out_buf, out_idx_buf, counter, histogram, pass);
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
              OutOffsetT,
              SelectMin>;

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
    extract_bin_op_t{},
    identify_candidates_op)
    .InvokeLastFilter(in_buf, in_idx_buf, counter, histogram, k, pass);
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
 *   The K value. Will find K elements from num_items elements. The variable K should be smaller than the variable N.
 *
 * @param[in] stream
 *   @rst
 *   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
 *   @endrst
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
  /// Device-accessible allocation of temporary storage.
  /// When `nullptr`, the required allocation size is written to `temp_storage_bytes`
  /// and no work is done.
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
  static constexpr bool KEYS_ONLY = ::cuda::std::is_same<ValueInputIteratorT, NullType>::value;
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
   *   The K value. Will find K elements from num_items elements. The variable K should be smaller than the variable N.
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
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE int CalculateBlocksPerSM(TopKKernelPtrT topk_kernel, int block_threads)
  {
    int topk_blocks_per_sm;
    cudaError error;
    do
    {
      error =
        CubDebug(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&topk_blocks_per_sm, topk_kernel, block_threads, 0));
      if (cudaSuccess != error)
      {
        break;
      }
    } while (0);
    return topk_blocks_per_sm;
  }

  template <typename ActivePolicyT,
            typename TopKFirstPassKernelPtrT,
            typename TopKKernelPtrT,
            typename TopKLastFilterKernelPtrT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
  Invoke(TopKFirstPassKernelPtrT topk_firstpass_kernel,
         TopKKernelPtrT topk_kernel,
         TopKLastFilterKernelPtrT topk_lastfilter_kernel)
  {
    using policy_t = typename ActivePolicyT::topk_policy_t;

    cudaError error = cudaSuccess;

    constexpr int block_threads    = policy_t::BLOCK_THREADS; // Threads per block
    constexpr int items_per_thread = policy_t::ITEMS_PER_THREAD; // Items per thread
    constexpr int tile_size        = block_threads * items_per_thread; // Items per block
    int num_tiles                  = static_cast<int>(::cuda::ceil_div(num_items, tile_size)); // Num of blocks
    constexpr int num_passes       = calc_num_passes<key_in_t, policy_t::BITS_PER_PASS>();
    constexpr int num_buckets      = 1 << policy_t::BITS_PER_PASS;

    if (static_cast<OffsetT>(k) >= num_items)
    {
      // We only support the case where the variable K is smaller than the variable N.
      return cudaErrorInvalidValue;
    }

    // Specify temporary storage allocation requirements
    size_t size_counter   = sizeof(Counter<key_in_t, OffsetT, OutOffsetT>);
    size_t size_histogram = num_buckets * sizeof(OffsetT);
    size_t num_candidates = ::cuda::std::max((size_t) 256, (size_t) num_items / policy_t::COEFFICIENT_FOR_BUFFER);

    size_t allocation_sizes[6] = {
      size_counter,
      size_histogram,
      num_candidates * sizeof(key_in_t),
      num_candidates * sizeof(key_in_t),
      KEYS_ONLY ? 0 : num_candidates * sizeof(OffsetT),
      KEYS_ONLY ? 0 : num_candidates * sizeof(OffsetT)};

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

    dim3 topk_grid_size;
    topk_grid_size.z       = 1;
    topk_grid_size.y       = 1;
    int topk_blocks_per_sm = CalculateBlocksPerSM(topk_kernel, block_threads);
    topk_grid_size.x =
      ::cuda::std::min(static_cast<unsigned int>(topk_blocks_per_sm * num_sms),
                       static_cast<unsigned int>((num_items - 1) / (items_per_thread * block_threads) + 1));

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
        counter->kth_key_bits, pass);

      // Initialize address variables
      in_buf  = static_cast<key_in_t*>(pass % 2 == 0 ? allocations[2] : allocations[3]);
      out_buf = pass == 0 ? nullptr : static_cast<key_in_t*>(pass % 2 == 0 ? allocations[3] : allocations[2]);
      if (!KEYS_ONLY)
      {
        in_idx_buf  = pass <= 1 ? nullptr : static_cast<OffsetT*>(pass % 2 == 0 ? allocations[4] : allocations[5]);
        out_idx_buf = pass == 0 ? nullptr : static_cast<OffsetT*>(pass % 2 == 0 ? allocations[5] : allocations[4]);
      }

      // Invoke kernel
      if (pass == 0)
      {
        int topk_blocks_per_sm = CalculateBlocksPerSM(topk_firstpass_kernel, block_threads);
        dim3 topk_firstpass_grid_size;
        topk_firstpass_grid_size.z = 1;
        topk_firstpass_grid_size.y = 1;
        topk_firstpass_grid_size.x =
          ::cuda::std::min((unsigned int) topk_blocks_per_sm * num_sms,
                           (unsigned int) (num_items - 1) / (items_per_thread * block_threads) + 1);
        THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(topk_firstpass_grid_size, block_threads, 0, stream)
          .doit(
            topk_firstpass_kernel,
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
            extract_bin_op,
            identify_candidates_op,
            pass);
      }
    }
    // Set operator
    IdentifyCandidatesOp<key_in_t, !SelectMin, policy_t::BITS_PER_PASS> identify_candidates_op(
      counter->kth_key_bits, pass);
    topk_blocks_per_sm = CalculateBlocksPerSM(topk_lastfilter_kernel, block_threads);
    topk_grid_size.x   = ::cuda::std::min((unsigned int) topk_blocks_per_sm * num_sms,
                                        (unsigned int) (num_items - 1) / (items_per_thread * block_threads) + 1);
    THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(topk_grid_size, block_threads, 0, stream)
      .doit(topk_lastfilter_kernel,
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
   *   The K value. Will find K elements from num_items elements. The variable K should be smaller than the variable N.
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
