// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2018, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * @file
 * cub::DeviceRadixSort provides device-wide, parallel operations for computing a radix sort across
 * a sequence of data items residing within device-accessible memory.
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

#include <cub/device/dispatch/kernels/kernel_radix_sort.cuh>
#include <cub/device/dispatch/tuning/tuning_radix_sort.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

// TODO(bgruber): included for backward compatibility, remove in CCCL 4.0
#include <cub/device/dispatch/dispatch_segmented_radix_sort.cuh>

// suppress warnings triggered by #pragma unroll:
// "warning: loop not unrolled: the optimizer was unable to perform the requested transformation; the transformation
// might be disabled or specified as part of an unsupported transformation ordering [-Wpass-failed=transform-warning]"
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wpass-failed")

CUB_NAMESPACE_BEGIN

namespace detail::radix_sort
{
template <typename MaxPolicyT, SortOrder Order, typename KeyT, typename ValueT, typename OffsetT, typename DecomposerT>
struct DeviceRadixSortKernelSource
{
  CUB_DEFINE_KERNEL_GETTER(RadixSortSingleTileKernel,
                           DeviceRadixSortSingleTileKernel<MaxPolicyT, Order, KeyT, ValueT, OffsetT, DecomposerT>);

  CUB_DEFINE_KERNEL_GETTER(RadixSortUpsweepKernel,
                           DeviceRadixSortUpsweepKernel<MaxPolicyT, false, Order, KeyT, OffsetT, DecomposerT>);

  CUB_DEFINE_KERNEL_GETTER(RadixSortAltUpsweepKernel,
                           DeviceRadixSortUpsweepKernel<MaxPolicyT, true, Order, KeyT, OffsetT, DecomposerT>);

  CUB_DEFINE_KERNEL_GETTER(DeviceRadixSortScanBinsKernel, RadixSortScanBinsKernel<MaxPolicyT, OffsetT>);

  CUB_DEFINE_KERNEL_GETTER(RadixSortDownsweepKernel,
                           DeviceRadixSortDownsweepKernel<MaxPolicyT, false, Order, KeyT, ValueT, OffsetT, DecomposerT>);

  CUB_DEFINE_KERNEL_GETTER(RadixSortAltDownsweepKernel,
                           DeviceRadixSortDownsweepKernel<MaxPolicyT, true, Order, KeyT, ValueT, OffsetT, DecomposerT>);

  CUB_DEFINE_KERNEL_GETTER(RadixSortHistogramKernel,
                           DeviceRadixSortHistogramKernel<MaxPolicyT, Order, KeyT, OffsetT, DecomposerT>);

  CUB_DEFINE_KERNEL_GETTER(RadixSortExclusiveSumKernel, DeviceRadixSortExclusiveSumKernel<MaxPolicyT, OffsetT>);

  CUB_DEFINE_KERNEL_GETTER(
    RadixSortOnesweepKernel,
    DeviceRadixSortOnesweepKernel<MaxPolicyT, Order, KeyT, ValueT, OffsetT, int, int, DecomposerT>);

  CUB_RUNTIME_FUNCTION static constexpr size_t KeySize()
  {
    return sizeof(KeyT);
  }

  CUB_RUNTIME_FUNCTION static constexpr size_t ValueSize()
  {
    return sizeof(ValueT);
  }
};
} // namespace detail::radix_sort

/******************************************************************************
 * Single-problem dispatch
 ******************************************************************************/

/**
 * Utility class for dispatching the appropriately-tuned kernels for device-wide radix sort
 *
 * @tparam SortOrder
 *   Whether to sort in ascending or descending order
 *
 * @tparam KeyT
 *   Key type
 *
 * @tparam ValueT
 *   Value type
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam DecomposerT
 *   Implementation detail, do not specify directly, requirements on the
 *   content of this type are subject to breaking change.
 */
template <SortOrder Order,
          typename KeyT,
          typename ValueT,
          typename OffsetT,
          typename DecomposerT  = detail::identity_decomposer_t,
          typename PolicyHub    = detail::radix_sort::policy_hub<KeyT, ValueT, OffsetT>,
          typename KernelSource = detail::radix_sort::
            DeviceRadixSortKernelSource<typename PolicyHub::MaxPolicy, Order, KeyT, ValueT, OffsetT, DecomposerT>,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
struct DispatchRadixSort
{
  //------------------------------------------------------------------------------
  // Constants
  //------------------------------------------------------------------------------

  // Whether this is a keys-only (or key-value) sort
  static constexpr bool KEYS_ONLY = ::cuda::std::is_same_v<ValueT, NullType>;

  //------------------------------------------------------------------------------
  // Problem state
  //------------------------------------------------------------------------------

  /// Device-accessible allocation of temporary storage.
  //  When nullptr, the required allocation size is written to `temp_storage_bytes` and no work is
  //  done.
  void* d_temp_storage;

  /// Reference to size in bytes of `d_temp_storage` allocation
  size_t& temp_storage_bytes;

  /// Double-buffer whose current buffer contains the unsorted input keys and, upon return, is
  /// updated to point to the sorted output keys
  DoubleBuffer<KeyT>& d_keys;

  /// Double-buffer whose current buffer contains the unsorted input values and, upon return, is
  /// updated to point to the sorted output values
  DoubleBuffer<ValueT>& d_values;

  /// Number of items to sort
  OffsetT num_items;

  /// The beginning (least-significant) bit index needed for key comparison
  int begin_bit;

  /// The past-the-end (most-significant) bit index needed for key comparison
  int end_bit;

  /// CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
  cudaStream_t stream;

  /// PTX version
  int ptx_version;

  /// Whether is okay to overwrite source buffers
  bool is_overwrite_okay;

  DecomposerT decomposer;

  KernelSource kernel_source;

  KernelLauncherFactory launcher_factory;

  //------------------------------------------------------------------------------
  // Constructor
  //------------------------------------------------------------------------------

  /// Constructor
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchRadixSort(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    DoubleBuffer<KeyT>& d_keys,
    DoubleBuffer<ValueT>& d_values,
    OffsetT num_items,
    int begin_bit,
    int end_bit,
    bool is_overwrite_okay,
    cudaStream_t stream,
    int ptx_version,
    DecomposerT decomposer                 = {},
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {})
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_keys(d_keys)
      , d_values(d_values)
      , num_items(num_items)
      , begin_bit(begin_bit)
      , end_bit(end_bit)
      , stream(stream)
      , ptx_version(ptx_version)
      , is_overwrite_okay(is_overwrite_okay)
      , decomposer(decomposer)
      , kernel_source(kernel_source)
      , launcher_factory(launcher_factory)
  {}

  //------------------------------------------------------------------------------
  // Small-problem (single tile) invocation
  //------------------------------------------------------------------------------

  /**
   * @brief Invoke a single block to sort in-core
   *
   * @tparam ActivePolicyT
   *   Umbrella policy active for the target device
   *
   * @tparam SingleTileKernelT
   *   Function type of cub::DeviceRadixSortSingleTileKernel
   *
   * @param[in] single_tile_kernel
   *   Kernel function pointer to parameterization of cub::DeviceRadixSortSingleTileKernel
   */
  template <typename ActivePolicyT, typename SingleTileKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t
  InvokeSingleTile(SingleTileKernelT single_tile_kernel, ActivePolicyT policy = {})
  {
    // Return if the caller is simply requesting the size of the storage allocation
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }

// Log single_tile_kernel configuration
#ifdef CUB_DEBUG_LOG
    _CubLog("Invoking single_tile_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy, current bit "
            "%d, bit_grain %d\n",
            1,
            policy.SingleTile().BlockThreads(),
            (long long) stream,
            policy.SingleTile().ItemsPerThread(),
            1,
            begin_bit,
            policy.RadixBits(policy.SingleTile()));
#endif

    // Invoke upsweep_kernel with same grid size as downsweep_kernel
    launcher_factory(1, policy.SingleTile().BlockThreads(), 0, stream)
      .doit(single_tile_kernel,
            d_keys.Current(),
            d_keys.Alternate(),
            d_values.Current(),
            d_values.Alternate(),
            num_items,
            begin_bit,
            end_bit,
            decomposer);

    // Check for failure to launch
    if (const auto error = CubDebug(cudaPeekAtLastError()))
    {
      return error;
    }

    // Sync the stream if specified to flush runtime errors
    if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
    {
      return error;
    }

    // Update selector
    d_keys.selector ^= 1;
    d_values.selector ^= 1;

    return cudaSuccess;
  }

  //------------------------------------------------------------------------------
  // Normal problem size invocation
  //------------------------------------------------------------------------------

  /**
   * Invoke a three-kernel sorting pass at the current bit.
   */
  template <typename PassConfigT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t InvokePass(
    const KeyT* d_keys_in,
    KeyT* d_keys_out,
    const ValueT* d_values_in,
    ValueT* d_values_out,
    OffsetT* d_spine,
    int /*spine_length*/,
    int& current_bit,
    PassConfigT& pass_config)
  {
    int pass_bits = ::cuda::std::min(pass_config.radix_bits, end_bit - current_bit);

// Log upsweep_kernel configuration
#ifdef CUB_DEBUG_LOG
    _CubLog("Invoking upsweep_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy, current bit %d, "
            "bit_grain %d\n",
            pass_config.even_share.grid_size,
            pass_config.upsweep_config.block_threads,
            (long long) stream,
            pass_config.upsweep_config.items_per_thread,
            pass_config.upsweep_config.sm_occupancy,
            current_bit,
            pass_bits);
#endif

    // Spine length written by the upsweep kernel in the current pass.
    int pass_spine_length = pass_config.even_share.grid_size * pass_config.radix_digits;

    // Invoke upsweep_kernel with same grid size as downsweep_kernel
    launcher_factory(pass_config.even_share.grid_size, pass_config.upsweep_config.block_threads, 0, stream)
      .doit(pass_config.upsweep_kernel,
            d_keys_in,
            d_spine,
            num_items,
            current_bit,
            pass_bits,
            pass_config.even_share,
            decomposer);

    // Check for failure to launch
    if (const auto error = CubDebug(cudaPeekAtLastError()))
    {
      return error;
    }

    // Sync the stream if specified to flush runtime errors
    if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
    {
      return error;
    }

// Log scan_kernel configuration
#ifdef CUB_DEBUG_LOG
    _CubLog("Invoking scan_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread\n",
            1,
            pass_config.scan_config.block_threads,
            (long long) stream,
            pass_config.scan_config.items_per_thread);
#endif

    // Invoke scan_kernel
    launcher_factory(1, pass_config.scan_config.block_threads, 0, stream)
      .doit(pass_config.scan_kernel, d_spine, pass_spine_length);

    // Check for failure to launch
    if (const auto error = CubDebug(cudaPeekAtLastError()))
    {
      return error;
    }

    // Sync the stream if specified to flush runtime errors
    if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
    {
      return error;
    }

// Log downsweep_kernel configuration
#ifdef CUB_DEBUG_LOG
    _CubLog("Invoking downsweep_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
            pass_config.even_share.grid_size,
            pass_config.downsweep_config.block_threads,
            (long long) stream,
            pass_config.downsweep_config.items_per_thread,
            pass_config.downsweep_config.sm_occupancy);
#endif

    // Invoke downsweep_kernel
    launcher_factory(pass_config.even_share.grid_size, pass_config.downsweep_config.block_threads, 0, stream)
      .doit(pass_config.downsweep_kernel,
            d_keys_in,
            d_keys_out,
            d_values_in,
            d_values_out,
            d_spine,
            num_items,
            current_bit,
            pass_bits,
            pass_config.even_share,
            decomposer);

    // Check for failure to launch
    if (const auto error = CubDebug(cudaPeekAtLastError()))
    {
      return error;
    }

    // Sync the stream if specified to flush runtime errors
    if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
    {
      return error;
    }

    // Update current bit
    current_bit += pass_bits;

    return cudaSuccess;
  }

  /// Pass configuration structure
  template <typename UpsweepKernelT, typename ScanKernelT, typename DownsweepKernelT>
  struct PassConfig
  {
    UpsweepKernelT upsweep_kernel;
    detail::KernelConfig upsweep_config;
    ScanKernelT scan_kernel;
    detail::KernelConfig scan_config;
    DownsweepKernelT downsweep_kernel;
    detail::KernelConfig downsweep_config;
    int radix_bits;
    int radix_digits;
    int max_downsweep_grid_size;
    GridEvenShare<OffsetT> even_share;

    /// Initialize pass configuration
    template <typename ActivePolicyT, typename UpsweepPolicyT, typename ScanPolicyT, typename DownsweepPolicyT>
    CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t InitPassConfig(
      UpsweepKernelT upsweep_kernel,
      ScanKernelT scan_kernel,
      DownsweepKernelT downsweep_kernel,
      int /*ptx_version*/,
      int sm_count,
      OffsetT num_items,
      ActivePolicyT policy                   = {},
      UpsweepPolicyT upsweep_policy          = {},
      ScanPolicyT scan_policy                = {},
      DownsweepPolicyT downsweep_policy      = {},
      KernelLauncherFactory launcher_factory = {})
    {
      this->upsweep_kernel   = upsweep_kernel;
      this->scan_kernel      = scan_kernel;
      this->downsweep_kernel = downsweep_kernel;
      radix_bits             = policy.RadixBits(downsweep_policy);
      radix_digits           = 1 << radix_bits;

      if (const auto error = CubDebug(upsweep_config.Init(upsweep_kernel, upsweep_policy, launcher_factory)))
      {
        return error;
      }

      if (const auto error = CubDebug(scan_config.Init(scan_kernel, scan_policy, launcher_factory)))
      {
        return error;
      }

      if (const auto error = CubDebug(downsweep_config.Init(downsweep_kernel, downsweep_policy, launcher_factory)))
      {
        return error;
      }

      max_downsweep_grid_size = (downsweep_config.sm_occupancy * sm_count) * detail::subscription_factor;

      even_share.DispatchInit(
        num_items, max_downsweep_grid_size, ::cuda::std::max(downsweep_config.tile_size, upsweep_config.tile_size));

      return cudaSuccess;
    }
  };

  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t InvokeOnesweep(ActivePolicyT policy = {})
  {
    // PortionOffsetT is used for offsets within a portion, and must be signed.
    using PortionOffsetT = int;
    using AtomicOffsetT  = PortionOffsetT;

    // compute temporary storage size
    const int RADIX_BITS                = policy.RadixBits(policy.Onesweep());
    const int RADIX_DIGITS              = 1 << RADIX_BITS;
    const int ONESWEEP_ITEMS_PER_THREAD = policy.Onesweep().ItemsPerThread();
    const int ONESWEEP_BLOCK_THREADS    = policy.Onesweep().BlockThreads();
    const int ONESWEEP_TILE_ITEMS       = ONESWEEP_ITEMS_PER_THREAD * ONESWEEP_BLOCK_THREADS;
    // portions handle inputs with >=2**30 elements, due to the way lookback works
    // for testing purposes, one portion is <= 2**28 elements
    const PortionOffsetT PORTION_SIZE = ((1 << 28) - 1) / ONESWEEP_TILE_ITEMS * ONESWEEP_TILE_ITEMS;
    int num_passes                    = ::cuda::ceil_div(end_bit - begin_bit, RADIX_BITS);
    OffsetT num_portions              = static_cast<OffsetT>(::cuda::ceil_div(num_items, PORTION_SIZE));
    PortionOffsetT max_num_blocks     = ::cuda::ceil_div(
      static_cast<int>(::cuda::std::min(num_items, static_cast<OffsetT>(PORTION_SIZE))), ONESWEEP_TILE_ITEMS);

    size_t value_size         = KEYS_ONLY ? 0 : kernel_source.ValueSize();
    size_t allocation_sizes[] = {
      // bins
      num_portions * num_passes * RADIX_DIGITS * sizeof(OffsetT),
      // lookback
      max_num_blocks * RADIX_DIGITS * sizeof(AtomicOffsetT),
      // extra key buffer
      is_overwrite_okay || num_passes <= 1 ? 0 : num_items * kernel_source.KeySize(),
      // extra value buffer
      is_overwrite_okay || num_passes <= 1 ? 0 : num_items * value_size,
      // counters
      num_portions * num_passes * sizeof(AtomicOffsetT),
    };
    constexpr int NUM_ALLOCATIONS      = sizeof(allocation_sizes) / sizeof(allocation_sizes[0]);
    void* allocations[NUM_ALLOCATIONS] = {};
    if (const auto error =
          detail::alias_temporaries<NUM_ALLOCATIONS>(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))
    {
      return error;
    }

    // just return if no temporary storage is provided
    if (d_temp_storage == nullptr)
    {
      return cudaSuccess;
    }

    OffsetT* d_bins           = (OffsetT*) allocations[0];
    AtomicOffsetT* d_lookback = (AtomicOffsetT*) allocations[1];
    KeyT* d_keys_tmp2         = (KeyT*) allocations[2];
    ValueT* d_values_tmp2     = (ValueT*) allocations[3];
    AtomicOffsetT* d_ctrs     = (AtomicOffsetT*) allocations[4];

    // initialization
    if (const auto error =
          CubDebug(cudaMemsetAsync(d_ctrs, 0, num_portions * num_passes * sizeof(AtomicOffsetT), stream)))
    {
      return error;
    }

    // compute num_passes histograms with RADIX_DIGITS bins each
    if (const auto error = CubDebug(cudaMemsetAsync(d_bins, 0, num_passes * RADIX_DIGITS * sizeof(OffsetT), stream)))
    {
      return error;
    }
    int device  = -1;
    int num_sms = 0;

    if (const auto error = CubDebug(cudaGetDevice(&device)))
    {
      return error;
    }

    if (const auto error = CubDebug(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device)))
    {
      return error;
    }

    const int HISTO_BLOCK_THREADS = policy.Histogram().BlockThreads();
    int histo_blocks_per_sm       = 1;
    auto histogram_kernel         = kernel_source.RadixSortHistogramKernel();

    if (const auto error =
          CubDebug(launcher_factory.MaxSmOccupancy(histo_blocks_per_sm, histogram_kernel, HISTO_BLOCK_THREADS, 0)))
    {
      return error;
    }

// log histogram_kernel configuration
#ifdef CUB_DEBUG_LOG
    _CubLog("Invoking histogram_kernel<<<%d, %d, 0, %lld>>>(), %d items per iteration, "
            "%d SM occupancy, bit_grain %d\n",
            histo_blocks_per_sm * num_sms,
            HISTO_BLOCK_THREADS,
            reinterpret_cast<long long>(stream),
            policy.Histogram().ItemsPerThread(),
            histo_blocks_per_sm,
            policy.RadixBits(policy.Histogram()));
#endif

    if (const auto error = CubDebug(
          launcher_factory(histo_blocks_per_sm * num_sms, HISTO_BLOCK_THREADS, 0, stream)
            .doit(histogram_kernel, d_bins, d_keys.Current(), num_items, begin_bit, end_bit, decomposer)))
    {
      return error;
    }

    if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
    {
      return error;
    }

    // exclusive sums to determine starts
    const int SCAN_BLOCK_THREADS = policy.BlockThreads(policy.ExclusiveSum());

// log exclusive_sum_kernel configuration
#ifdef CUB_DEBUG_LOG
    _CubLog("Invoking exclusive_sum_kernel<<<%d, %d, 0, %lld>>>(), bit_grain %d\n",
            num_passes,
            SCAN_BLOCK_THREADS,
            reinterpret_cast<long long>(stream),
            policy.RadixBits(policy.ExclusiveSum()));
#endif

    if (const auto error = CubDebug(launcher_factory(num_passes, SCAN_BLOCK_THREADS, 0, stream)
                                      .doit(kernel_source.RadixSortExclusiveSumKernel(), d_bins)))
    {
      return error;
    }

    if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
    {
      return error;
    }
    // use the other buffer if no overwrite is allowed
    KeyT* d_keys_tmp     = d_keys.Alternate();
    ValueT* d_values_tmp = d_values.Alternate();
    if (!is_overwrite_okay && num_passes % 2 == 0)
    {
      d_keys.d_buffers[1]   = d_keys_tmp2;
      d_values.d_buffers[1] = d_values_tmp2;
    }

    for (int current_bit = begin_bit, pass = 0; current_bit < end_bit; current_bit += RADIX_BITS, ++pass)
    {
      int num_bits = ::cuda::std::min(end_bit - current_bit, RADIX_BITS);
      for (OffsetT portion = 0; portion < num_portions; ++portion)
      {
        PortionOffsetT portion_num_items = static_cast<PortionOffsetT>(
          ::cuda::std::min(num_items - portion * PORTION_SIZE, static_cast<OffsetT>(PORTION_SIZE)));

        PortionOffsetT num_blocks = ::cuda::ceil_div(portion_num_items, ONESWEEP_TILE_ITEMS);

        if (const auto error =
              CubDebug(cudaMemsetAsync(d_lookback, 0, num_blocks * RADIX_DIGITS * sizeof(AtomicOffsetT), stream)))
        {
          return error;
        }

// log onesweep_kernel configuration
#ifdef CUB_DEBUG_LOG
        _CubLog("Invoking onesweep_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread, "
                "current bit %d, bit_grain %d, portion %d/%d\n",
                num_blocks,
                ONESWEEP_BLOCK_THREADS,
                reinterpret_cast<long long>(stream),
                policy.Onesweep().ItemsPerThread(),
                current_bit,
                num_bits,
                static_cast<int>(portion),
                static_cast<int>(num_portions));
#endif

        auto onesweep_kernel = kernel_source.RadixSortOnesweepKernel();

        if (const auto error = CubDebug(
              launcher_factory(num_blocks, ONESWEEP_BLOCK_THREADS, 0, stream)
                .doit(
                  onesweep_kernel,
                  d_lookback,
                  d_ctrs + portion * num_passes + pass,
                  portion < num_portions - 1 ? d_bins + ((portion + 1) * num_passes + pass) * RADIX_DIGITS : nullptr,
                  d_bins + (portion * num_passes + pass) * RADIX_DIGITS,
                  d_keys.Alternate(),
                  d_keys.Current() + portion * PORTION_SIZE,
                  d_values.Alternate(),
                  d_values.Current() + portion * PORTION_SIZE,
                  portion_num_items,
                  current_bit,
                  num_bits,
                  decomposer)))
        {
          return error;
        }

        if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
        {
          return error;
        }
      }

      // use the temporary buffers if no overwrite is allowed
      if (!is_overwrite_okay && pass == 0)
      {
        d_keys   = num_passes % 2 == 0 ? DoubleBuffer<KeyT>(d_keys_tmp, d_keys_tmp2)
                                       : DoubleBuffer<KeyT>(d_keys_tmp2, d_keys_tmp);
        d_values = num_passes % 2 == 0 ? DoubleBuffer<ValueT>(d_values_tmp, d_values_tmp2)
                                       : DoubleBuffer<ValueT>(d_values_tmp2, d_values_tmp);
      }
      d_keys.selector ^= 1;
      d_values.selector ^= 1;
    }

    return cudaSuccess;
  }

  /**
   * @brief Invocation (run multiple digit passes)
   *
   * @tparam ActivePolicyT
   *   Umbrella policy active for the target device
   *
   * @tparam UpsweepKernelT
   *   Function type of cub::DeviceRadixSortUpsweepKernel
   *
   * @tparam ScanKernelT
   *   Function type of cub::SpineScanKernel
   *
   * @tparam DownsweepKernelT
   *   Function type of cub::DeviceRadixSortDownsweepKernel
   *
   * @param[in] upsweep_kernel
   *   Kernel function pointer to parameterization of cub::DeviceRadixSortUpsweepKernel
   *
   * @param[in] alt_upsweep_kernel
   *   Alternate kernel function pointer to parameterization of cub::DeviceRadixSortUpsweepKernel
   *
   * @param[in] scan_kernel
   *   Kernel function pointer to parameterization of cub::SpineScanKernel
   *
   * @param[in] downsweep_kernel
   *   Kernel function pointer to parameterization of cub::DeviceRadixSortDownsweepKernel
   *
   * @param[in] alt_downsweep_kernel
   *   Alternate kernel function pointer to parameterization of
   *   cub::DeviceRadixSortDownsweepKernel
   */
  template <typename ActivePolicyT, typename UpsweepKernelT, typename ScanKernelT, typename DownsweepKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t InvokePasses(
    UpsweepKernelT upsweep_kernel,
    UpsweepKernelT alt_upsweep_kernel,
    ScanKernelT scan_kernel,
    DownsweepKernelT downsweep_kernel,
    DownsweepKernelT alt_downsweep_kernel,
    ActivePolicyT policy = {})
  {
    // Get device ordinal
    int device_ordinal;
    if (const auto error = CubDebug(cudaGetDevice(&device_ordinal)))
    {
      return error;
    }

    // Get SM count
    int sm_count;
    if (const auto error = CubDebug(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal)))
    {
      return error;
    }

    // Init regular and alternate-digit kernel configurations
    PassConfig<UpsweepKernelT, ScanKernelT, DownsweepKernelT> pass_config, alt_pass_config;
    if (const auto error = pass_config.InitPassConfig(
          upsweep_kernel,
          scan_kernel,
          downsweep_kernel,
          ptx_version,
          sm_count,
          num_items,
          policy,
          policy.Upsweep(),
          policy.Scan(),
          policy.Downsweep(),
          launcher_factory))
    {
      return error;
    }

    if (const auto error = alt_pass_config.InitPassConfig(
          alt_upsweep_kernel,
          scan_kernel,
          alt_downsweep_kernel,
          ptx_version,
          sm_count,
          num_items,
          policy,
          policy.AltUpsweep(),
          policy.Scan(),
          policy.AltDownsweep(),
          launcher_factory))
    {
      return error;
    }

    // Get maximum spine length
    int max_grid_size = ::cuda::std::max(pass_config.max_downsweep_grid_size, alt_pass_config.max_downsweep_grid_size);
    int spine_length  = (max_grid_size * pass_config.radix_digits) + pass_config.scan_config.tile_size;

    // Temporary storage allocation requirements
    void* allocations[3]       = {};
    size_t allocation_sizes[3] = {
      // bytes needed for privatized block digit histograms
      spine_length * sizeof(OffsetT),

      // bytes needed for 3rd keys buffer
      (is_overwrite_okay) ? 0 : num_items * kernel_source.KeySize(),

      // bytes needed for 3rd values buffer
      (is_overwrite_okay || (KEYS_ONLY)) ? 0 : num_items * kernel_source.ValueSize(),
    };

    // Alias the temporary allocations from the single storage blob (or compute the necessary size of the blob)
    if (const auto error =
          CubDebug(detail::alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes)))
    {
      return error;
    }

    // Return if the caller is simply requesting the size of the storage allocation
    if (d_temp_storage == nullptr)
    {
      return cudaSuccess;
    }

    // Pass planning.  Run passes of the alternate digit-size configuration until we have an even multiple of our
    // preferred digit size
    int num_bits           = end_bit - begin_bit;
    int num_passes         = ::cuda::ceil_div(num_bits, pass_config.radix_bits);
    bool is_num_passes_odd = num_passes & 1;
    int max_alt_passes     = (num_passes * pass_config.radix_bits) - num_bits;
    int alt_end_bit        = ::cuda::std::min(end_bit, begin_bit + (max_alt_passes * alt_pass_config.radix_bits));

    // Alias the temporary storage allocations
    OffsetT* d_spine = static_cast<OffsetT*>(allocations[0]);

    DoubleBuffer<KeyT> d_keys_remaining_passes(
      (is_overwrite_okay || is_num_passes_odd) ? d_keys.Alternate() : static_cast<KeyT*>(allocations[1]),
      (is_overwrite_okay)   ? d_keys.Current()
      : (is_num_passes_odd) ? static_cast<KeyT*>(allocations[1])
                            : d_keys.Alternate());

    DoubleBuffer<ValueT> d_values_remaining_passes(
      (is_overwrite_okay || is_num_passes_odd) ? d_values.Alternate() : static_cast<ValueT*>(allocations[2]),
      (is_overwrite_okay)   ? d_values.Current()
      : (is_num_passes_odd) ? static_cast<ValueT*>(allocations[2])
                            : d_values.Alternate());

    // Run first pass, consuming from the input's current buffers
    int current_bit = begin_bit;
    if (const auto error = CubDebug(InvokePass(
          d_keys.Current(),
          d_keys_remaining_passes.Current(),
          d_values.Current(),
          d_values_remaining_passes.Current(),
          d_spine,
          spine_length,
          current_bit,
          (current_bit < alt_end_bit) ? alt_pass_config : pass_config)))
    {
      return error;
    }

    // Run remaining passes
    while (current_bit < end_bit)
    {
      if (const auto error = CubDebug(InvokePass(
            d_keys_remaining_passes.d_buffers[d_keys_remaining_passes.selector],
            d_keys_remaining_passes.d_buffers[d_keys_remaining_passes.selector ^ 1],
            d_values_remaining_passes.d_buffers[d_keys_remaining_passes.selector],
            d_values_remaining_passes.d_buffers[d_keys_remaining_passes.selector ^ 1],
            d_spine,
            spine_length,
            current_bit,
            (current_bit < alt_end_bit) ? alt_pass_config : pass_config)))
      {
        return error;
      }

      // Invert selectors
      d_keys_remaining_passes.selector ^= 1;
      d_values_remaining_passes.selector ^= 1;
    }

    // Update selector
    if (!is_overwrite_okay)
    {
      num_passes = 1; // Sorted data always ends up in the other vector
    }

    d_keys.selector   = (d_keys.selector + num_passes) & 1;
    d_values.selector = (d_values.selector + num_passes) & 1;

    return cudaSuccess;
  }

  //------------------------------------------------------------------------------
  // Chained policy invocation
  //------------------------------------------------------------------------------

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t InvokeCopy()
  {
    // is_overwrite_okay == false here
    // Return the number of temporary bytes if requested
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }

// Copy keys
#ifdef CUB_DEBUG_LOG
    _CubLog("Invoking async copy of %lld keys on stream %lld\n", (long long) num_items, (long long) stream);
#endif
    if (const auto error = CubDebug(cudaMemcpyAsync(
          d_keys.Alternate(), d_keys.Current(), num_items * kernel_source.KeySize(), cudaMemcpyDefault, stream)))
    {
      return error;
    }

    if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
    {
      return error;
    }
    d_keys.selector ^= 1;

    // Copy values if necessary
    if (!KEYS_ONLY)
    {
#ifdef CUB_DEBUG_LOG
      _CubLog("Invoking async copy of %lld values on stream %lld\n", (long long) num_items, (long long) stream);
#endif
      if (const auto error = CubDebug(cudaMemcpyAsync(
            d_values.Alternate(), d_values.Current(), num_items * kernel_source.ValueSize(), cudaMemcpyDefault, stream)))
      {
        return error;
      }

      if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
      {
        return error;
      }
    }
    d_values.selector ^= 1;

    return cudaSuccess;
  }

  /// Invocation
  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke(ActivePolicyT policy = {})
  {
    auto wrapped_policy = detail::radix_sort::MakeRadixSortPolicyWrapper(policy);

    // Return if empty problem, or if no bits to sort and double-buffering is used
    if (num_items == 0 || (begin_bit == end_bit && is_overwrite_okay))
    {
      if (d_temp_storage == nullptr)
      {
        temp_storage_bytes = 1;
      }
      return cudaSuccess;
    }

    // Check if simple copy suffices (is_overwrite_okay == false at this point)
    if (begin_bit == end_bit)
    {
      bool has_uva = false;
      if (const auto error = detail::HasUVA(has_uva))
      {
        return error;
      }
      if (has_uva)
      {
        return InvokeCopy();
      }
    }

    // Force kernel code-generation in all compiler passes
    if (num_items <= static_cast<OffsetT>(
          wrapped_policy.SingleTile().BlockThreads() * wrapped_policy.SingleTile().ItemsPerThread()))
    {
      // Small, single tile size
      return InvokeSingleTile(kernel_source.RadixSortSingleTileKernel(), wrapped_policy);
    }

    if CUB_DETAIL_CONSTEXPR_ISH (wrapped_policy.IsOnesweep())
    {
      return InvokeOnesweep(wrapped_policy);
    }
    else
    {
      return InvokePasses(
        kernel_source.RadixSortUpsweepKernel(),
        kernel_source.RadixSortAltUpsweepKernel(),
        kernel_source.DeviceRadixSortScanBinsKernel(),
        kernel_source.RadixSortDownsweepKernel(),
        kernel_source.RadixSortAltDownsweepKernel(),
        wrapped_policy);
    }
  }

  //------------------------------------------------------------------------------
  // Dispatch entrypoints
  //------------------------------------------------------------------------------

  /**
   * @brief Internal dispatch routine
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When nullptr, the required
   *   allocation size is written to `temp_storage_bytes` and no work is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in,out] d_keys
   *   Double-buffer whose current buffer contains the unsorted input keys and,
   *   upon return, is updated to point to the sorted output keys
   *
   * @param[in,out] d_values
   *   Double-buffer whose current buffer contains the unsorted input values and,
   *   upon return, is updated to point to the sorted output values
   *
   * @param[in] num_items
   *   Number of items to sort
   *
   * @param[in] begin_bit
   *   The beginning (least-significant) bit index needed for key comparison
   *
   * @param[in] end_bit
   *   The past-the-end (most-significant) bit index needed for key comparison
   *
   * @param[in] is_overwrite_okay
   *   Whether is okay to overwrite source buffers
   *
   * @param[in] stream
   *   CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
   */
  template <typename MaxPolicyT = typename PolicyHub::MaxPolicy>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    DoubleBuffer<KeyT>& d_keys,
    DoubleBuffer<ValueT>& d_values,
    OffsetT num_items,
    int begin_bit,
    int end_bit,
    bool is_overwrite_okay,
    cudaStream_t stream,
    DecomposerT decomposer                 = {},
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {},
    MaxPolicyT max_policy                  = {})
  {
    // Get PTX version
    int ptx_version = 0;
    if (const auto error = CubDebug(launcher_factory.PtxVersion(ptx_version)))
    {
      return error;
    }

    // Create dispatch functor
    DispatchRadixSort dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_keys,
      d_values,
      num_items,
      begin_bit,
      end_bit,
      is_overwrite_okay,
      stream,
      ptx_version,
      decomposer,
      kernel_source,
      launcher_factory);

    // Dispatch to chained policy
    if (const auto error = CubDebug(max_policy.Invoke(ptx_version, dispatch)))
    {
      return error;
    }

    return cudaSuccess;
  }
};

CUB_NAMESPACE_END

_CCCL_DIAG_POP
