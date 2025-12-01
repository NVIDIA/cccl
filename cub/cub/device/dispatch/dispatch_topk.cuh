// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! cub::DeviceTopK provides device-wide, parallel operations for finding the K largest (or smallest) items
//! from sequences of unordered data items residing within device-accessible memory.

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
#include <cub/device/dispatch/tuning/tuning_topk.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_temporary_storage.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/cmath>

CUB_NAMESPACE_BEGIN

namespace detail::topk
{
enum class select
{
  // Select the K elements with the lowest values
  min,
  // Select the K elements with the highest values
  max
};

// Get the bin ID from the value of element
template <typename T, select SelectDirection, int BitsPerPass>
struct extract_bin_op_t
{
  int pass{};
  int start_bit;
  unsigned mask;

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE extract_bin_op_t(int pass)
      : pass(pass)
  {
    start_bit = calc_start_bit<T, BitsPerPass>(pass);
    mask      = calc_mask<T, BitsPerPass>(pass);
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int operator()(T key) const
  {
    auto bits = reinterpret_cast<typename Traits<T>::UnsignedBits&>(key);
    bits      = Traits<T>::TwiddleIn(bits);
    if constexpr (SelectDirection != select::min)
    {
      bits = ~bits;
    }
    int bucket = (bits >> start_bit) & mask;
    return bucket;
  }
};

// Check if the input element is still a candidate for the target pass.
template <typename T, select SelectDirection, int BitsPerPass>
struct identify_candidates_op_t
{
  using unsigned_bits_t = typename Traits<T>::UnsignedBits;
  unsigned_bits_t* kth_key_bits;
  int start_bit;
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE identify_candidates_op_t(unsigned_bits_t* kth_key_bits, int pass)
      : kth_key_bits(kth_key_bits)
  {
    start_bit = calc_start_bit<T, BitsPerPass>(pass - 1);
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE candidate_class operator()(T key) const
  {
    auto bits = reinterpret_cast<unsigned_bits_t&>(key);
    bits      = Traits<T>::TwiddleIn(bits);

    if constexpr (SelectDirection != select::min)
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
          bool IsFirstPass>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::topk_policy_t::block_threads))
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceTopKKernel(
    const KeyInputIteratorT d_keys_in,
    KeyOutputIteratorT d_keys_out,
    const ValueInputIteratorT d_values_in,
    ValueOutputIteratorT d_values_out,
    KeyInT* in_buf,
    OffsetT* in_idx_buf,
    KeyInT* out_buf,
    OffsetT* out_idx_buf,
    Counter<it_value_t<KeyInputIteratorT>, OffsetT, OutOffsetT>* counter,
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
    .template invoke_filter_and_histogram<IsFirstPass>(
      in_buf, in_idx_buf, out_buf, out_idx_buf, counter, histogram, pass);
}

template <typename ChainedPolicyT,
          typename KeyInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueInputIteratorT,
          typename ValueOutputIteratorT,
          typename OffsetT,
          typename OutOffsetT,
          typename KeyInT,
          typename IdentifyCandidatesOpT>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::topk_policy_t::block_threads))
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceTopKLastFilterKernel(
    const KeyInputIteratorT d_keys_in,
    KeyOutputIteratorT d_keys_out,
    const ValueInputIteratorT d_values_in,
    ValueOutputIteratorT d_values_out,
    KeyInT* in_buf,
    OffsetT* in_idx_buf,
    Counter<it_value_t<KeyInputIteratorT>, OffsetT, OutOffsetT>* counter,
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
    .invoke_last_filter(in_buf, in_idx_buf, counter, k, pass);
}

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
//!   **[inferred]** Random-access input iterator type for writing output values @iterator
//!
//! @tparam OffsetT
//!  Data Type for variables: num_items
//!
//! @tparam OutOffsetT
//!  Data Type for variables: k
//!
//! @tparam SelectDirection
//!   Determines whether to select the smallest or largest K elements.
template <typename KeyInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueInputIteratorT,
          typename ValueOutputIteratorT,
          typename OffsetT,
          typename OutOffsetT,
          select SelectDirection,
          typename SelectedPolicy = policy_hub<it_value_t<KeyInputIteratorT>, OffsetT>>
struct DispatchTopK
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

  using key_in_t                  = it_value_t<KeyInputIteratorT>;
  static constexpr bool keys_only = ::cuda::std::is_same_v<ValueInputIteratorT, NullType*>;

  template <typename ActivePolicyT,
            typename TopKFirstPassKernelPtrT,
            typename TopKKernelPtrT,
            typename TopKLastFilterKernelPtrT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t InvokePasses(
    TopKFirstPassKernelPtrT topk_first_pass_kernel,
    TopKKernelPtrT topk_kernel,
    TopKLastFilterKernelPtrT topk_last_filter_kernel)
  {
    using policy_t = typename ActivePolicyT::topk_policy_t;

    cudaError error = cudaSuccess;

    constexpr int block_threads    = policy_t::block_threads;
    constexpr int items_per_thread = policy_t::items_per_thread;
    constexpr int tile_size        = block_threads * items_per_thread;
    const auto num_tiles           = static_cast<unsigned int>(::cuda::ceil_div(num_items, tile_size));
    constexpr int num_passes       = calc_num_passes<key_in_t, policy_t::bits_per_pass>();
    constexpr int num_buckets      = 1 << policy_t::bits_per_pass;

    // Define operators
    using identify_candidates_op_t = identify_candidates_op_t<key_in_t, SelectDirection, policy_t::bits_per_pass>;
    using extract_bin_op_t         = extract_bin_op_t<key_in_t, SelectDirection, policy_t::bits_per_pass>;

    // We are capping k at a maximum of num_items
    using common_offset_t = ::cuda::std::common_type_t<OffsetT, OutOffsetT>;
    k = static_cast<OutOffsetT>(::cuda::std::min(common_offset_t{k}, static_cast<common_offset_t>(num_items)));

    // Specify temporary storage allocation requirements
    const size_t size_counter             = sizeof(Counter<key_in_t, OffsetT, OutOffsetT>);
    const size_t size_histogram           = num_buckets * sizeof(OffsetT);
    const OffsetT candidate_buffer_length = ::cuda::std::max(OffsetT{1}, num_items / coefficient_for_candidate_buffer);

    constexpr int allocations_array_size            = keys_only ? 4 : 6;
    size_t allocation_sizes[allocations_array_size] = {
      size_counter,
      size_histogram,
      candidate_buffer_length * sizeof(key_in_t),
      candidate_buffer_length * sizeof(key_in_t)};
    if constexpr (!keys_only)
    {
      allocation_sizes[4] = candidate_buffer_length * sizeof(OffsetT);
      allocation_sizes[5] = candidate_buffer_length * sizeof(OffsetT);
    }

    // Compute allocation pointers into the single storage blob (or compute the necessary size of the blob)
    void* allocations[allocations_array_size] = {};

    error = CubDebug(detail::alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes));
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

    int main_kernel_blocks_per_sm = 0;
    error                         = MaxSmOccupancy(main_kernel_blocks_per_sm, topk_kernel, block_threads);
    if (cudaSuccess != error)
    {
      return error;
    }
    const auto main_kernel_max_occupancy = static_cast<unsigned int>(main_kernel_blocks_per_sm * num_sms);
    const auto topk_grid_size            = ::cuda::std::min(main_kernel_max_occupancy, num_tiles);

// Log topk_kernel configuration @todo check the kernel launch
#ifdef CUB_DEBUG_LOG
    {
      // Get SM occupancy for select_if_kernel
      if (cudaSuccess != error)
      {
        return error;
      }

      _CubLog("Invoking topk_kernel<<<%d, %d, 0, "
              "%lld>>>(), %d items per thread, %d SM occupancy\n",
              topk_grid_size,
              block_threads,
              (long long) stream,
              items_per_thread,
              main_kernel_blocks_per_sm);
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
      extract_bin_op_t extract_bin_op(pass);
      identify_candidates_op_t identify_candidates_op(&counter->kth_key_bits, pass);

      // Initialize address variables
      in_buf  = static_cast<key_in_t*>(pass % 2 == 0 ? allocations[2] : allocations[3]);
      out_buf = pass == 0 ? nullptr : static_cast<key_in_t*>(pass % 2 == 0 ? allocations[3] : allocations[2]);
      if constexpr (!keys_only)
      {
        in_idx_buf  = pass <= 1 ? nullptr : static_cast<OffsetT*>(pass % 2 == 0 ? allocations[4] : allocations[5]);
        out_idx_buf = pass == 0 ? nullptr : static_cast<OffsetT*>(pass % 2 == 0 ? allocations[5] : allocations[4]);
      }

      // Invoke kernel
      if (pass == 0)
      {
        // Compute grid size for the histogram kernel of the first pass
        int first_pass_kernel_blocks_per_sm = 0;
        error = MaxSmOccupancy(first_pass_kernel_blocks_per_sm, topk_first_pass_kernel, block_threads);
        if (cudaSuccess != error)
        {
          return error;
        }
        const auto first_pass_kernel_max_occupancy =
          static_cast<unsigned int>(first_pass_kernel_blocks_per_sm * num_sms);
        const auto topk_first_pass_grid_size = ::cuda::std::min(first_pass_kernel_max_occupancy, num_tiles);

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

    identify_candidates_op_t identify_candidates_op(&counter->kth_key_bits, pass);
    int last_filter_kernel_blocks_per_sm = 0;
    error = MaxSmOccupancy(last_filter_kernel_blocks_per_sm, topk_last_filter_kernel, block_threads);
    if (cudaSuccess != error)
    {
      return error;
    }
    const auto last_filter_kernel_max_occupancy = static_cast<unsigned int>(last_filter_kernel_blocks_per_sm * num_sms);
    const auto last_filter_grid_size            = ::cuda::std::min(last_filter_kernel_max_occupancy, num_tiles);
    THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(last_filter_grid_size, block_threads, 0, stream)
      .doit(topk_last_filter_kernel,
            d_keys_in,
            d_keys_out,
            d_values_in,
            d_values_out,
            out_buf,
            out_idx_buf,
            counter,
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

    using identify_candidates_op_t =
      identify_candidates_op_t<key_in_t, SelectDirection, ActivePolicyT::topk_policy_t::bits_per_pass>;
    using extract_bin_op_t = extract_bin_op_t<key_in_t, SelectDirection, ActivePolicyT::topk_policy_t::bits_per_pass>;

    return InvokePasses<ActivePolicyT>(
      detail::topk::DeviceTopKKernel<
        max_policy_t,
        KeyInputIteratorT,
        KeyOutputIteratorT,
        ValueInputIteratorT,
        ValueOutputIteratorT,
        OffsetT,
        OutOffsetT,
        key_in_t,
        extract_bin_op_t,
        identify_candidates_op_t,
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
        extract_bin_op_t,
        identify_candidates_op_t,
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
        identify_candidates_op_t>);
  }

  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_keys_in
  //!   Pointer to the input data of key data to find top K
  //!
  //! @param[out] d_keys_out
  //!   Pointer to the K output sequence of key data
  //!
  //! @param[in] d_values_in
  //!   Pointer to the input sequence of associated value items
  //!
  //! @param[out] d_values_out
  //!   Pointer to the output sequence of associated value items
  //!
  //! @param[in] num_items
  //!   Number of items to be processed
  //!
  //! @param[in] k
  //!   The K value. Will find K elements from num_items elements. If K exceeds `num_items`, K is capped at a maximum of
  //! `num_items`.
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
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

    DispatchTopK dispatch{
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_keys_out,
      d_values_in,
      d_values_out,
      num_items,
      k,
      stream,
      ptx_version};

    return CubDebug(max_policy_t::Invoke(ptx_version, dispatch));
  }
};
} // namespace detail::topk

CUB_NAMESPACE_END
