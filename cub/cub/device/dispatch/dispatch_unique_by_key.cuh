// SPDX-FileCopyrightText: Copyright (c), NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * @file
 * cub::DeviceSelect::UniqueByKey provides device-wide, parallel operations for selecting unique
 * items by key from sequences of data items residing within device-accessible memory.
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

#include <cub/device/dispatch/dispatch_scan.cuh>
#include <cub/device/dispatch/kernels/kernel_scan.cuh>
#include <cub/device/dispatch/kernels/kernel_unique_by_key.cuh>
#include <cub/device/dispatch/tuning/tuning_unique_by_key.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_vsmem.cuh>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>

CUB_NAMESPACE_BEGIN

namespace detail::unique_by_key
{
template <typename MaxPolicyT,
          typename KeyInputIteratorT,
          typename ValueInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueOutputIteratorT,
          typename NumSelectedIteratorT,
          typename ScanTileStateT,
          typename EqualityOpT,
          typename OffsetT>

struct DeviceUniqueByKeyKernelSource
{
  CUB_DEFINE_KERNEL_GETTER(CompactInitKernel,
                           detail::scan::DeviceCompactInitKernel<ScanTileStateT, NumSelectedIteratorT>);

  CUB_DEFINE_KERNEL_GETTER(
    UniqueByKeySweepKernel,
    DeviceUniqueByKeySweepKernel<
      MaxPolicyT,
      KeyInputIteratorT,
      ValueInputIteratorT,
      KeyOutputIteratorT,
      ValueOutputIteratorT,
      NumSelectedIteratorT,
      ScanTileStateT,
      EqualityOpT,
      OffsetT>);

  CUB_RUNTIME_FUNCTION ScanTileStateT TileState()
  {
    return ScanTileStateT();
  }
};
} // namespace detail::unique_by_key

/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * @brief Utility class for dispatching the appropriately-tuned kernels for DeviceSelect
 *
 * @tparam KeyInputIteratorT
 *   Random-access input iterator type for keys
 *
 * @tparam ValueInputIteratorT
 *   Random-access input iterator type for values
 *
 * @tparam KeyOutputIteratorT
 *   Random-access output iterator type for keys
 *
 * @tparam ValueOutputIteratorT
 *   Random-access output iterator type for values
 *
 * @tparam NumSelectedIteratorT
 *   Output iterator type for recording the number of items selected
 *
 * @tparam EqualityOpT
 *   Equality operator type
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 */
template <
  typename KeyInputIteratorT,
  typename ValueInputIteratorT,
  typename KeyOutputIteratorT,
  typename ValueOutputIteratorT,
  typename NumSelectedIteratorT,
  typename EqualityOpT,
  typename OffsetT,
  typename PolicyHub =
    detail::unique_by_key::policy_hub<detail::it_value_t<KeyInputIteratorT>, detail::it_value_t<ValueInputIteratorT>>,
  typename KernelSource = detail::unique_by_key::DeviceUniqueByKeyKernelSource<
    typename PolicyHub::MaxPolicy,
    KeyInputIteratorT,
    ValueInputIteratorT,
    KeyOutputIteratorT,
    ValueOutputIteratorT,
    NumSelectedIteratorT,
    ScanTileState<OffsetT>,
    EqualityOpT,
    OffsetT>,
  typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY,
  typename VSMemHelperT          = detail::unique_by_key::VSMemHelper,
  typename KeyT                  = detail::it_value_t<KeyInputIteratorT>,
  typename ValueT                = detail::it_value_t<ValueInputIteratorT>>
struct DispatchUniqueByKey
{
  /******************************************************************************
   * Types and constants
   ******************************************************************************/

  static constexpr int INIT_KERNEL_THREADS = 128;

  /// Device-accessible allocation of temporary storage.  When nullptr, the required allocation size
  /// is written to `temp_storage_bytes` and no work is done.
  void* d_temp_storage;

  /// Reference to size in bytes of `d_temp_storage` allocation
  size_t& temp_storage_bytes;

  /// Pointer to the input sequence of keys
  KeyInputIteratorT d_keys_in;

  /// Pointer to the input sequence of values
  ValueInputIteratorT d_values_in;

  /// Pointer to the output sequence of selected data items
  KeyOutputIteratorT d_keys_out;

  /// Pointer to the output sequence of selected data items
  ValueOutputIteratorT d_values_out;

  /// Pointer to the total number of items selected
  /// (i.e., length of @p d_keys_out or @p d_values_out)
  NumSelectedIteratorT d_num_selected_out;

  /// Equality operator
  EqualityOpT equality_op;

  /// Total number of input items (i.e., length of @p d_keys_in or @p d_values_in)
  OffsetT num_items;

  /// **[optional]** CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
  cudaStream_t stream;

  KernelSource kernel_source;

  KernelLauncherFactory launcher_factory;

  /**
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage.
   *   When nullptr, the required allocation size is written to
   *   `temp_storage_bytes` and no work is done.
   *
   * @tparam temp_storage_bytes
   *   [in,out] Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_keys_in
   *   Pointer to the input sequence of keys
   *
   * @param[in] d_values_in
   *   Pointer to the input sequence of values
   *
   * @param[out] d_keys_out
   *   Pointer to the output sequence of selected data items
   *
   * @param[out] d_values_out
   *   Pointer to the output sequence of selected data items
   *
   * @param[out] d_num_selected_out
   *   Pointer to the total number of items selected
   *   (i.e., length of @p d_keys_out or @p d_values_out)
   *
   * @param[in] equality_op
   *   Equality operator
   *
   * @param[in] num_items
   *   Total number of input items (i.e., length of @p d_keys_in or @p d_values_in)
   *
   * @param[in] stream
   *   **[optional]** CUDA stream to launch kernels within.
   *   Default is stream<sub>0</sub>.
   */
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchUniqueByKey(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyInputIteratorT d_keys_in,
    ValueInputIteratorT d_values_in,
    KeyOutputIteratorT d_keys_out,
    ValueOutputIteratorT d_values_out,
    NumSelectedIteratorT d_num_selected_out,
    EqualityOpT equality_op,
    OffsetT num_items,
    cudaStream_t stream,
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {})
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_keys_in(d_keys_in)
      , d_values_in(d_values_in)
      , d_keys_out(d_keys_out)
      , d_values_out(d_values_out)
      , d_num_selected_out(d_num_selected_out)
      , equality_op(equality_op)
      , num_items(num_items)
      , stream(stream)
      , kernel_source(kernel_source)
      , launcher_factory(launcher_factory)
  {}

  /******************************************************************************
   * Dispatch entrypoints
   ******************************************************************************/

  template <typename ActivePolicyT, typename InitKernelT, typename UniqueByKeySweepKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_HOST _CCCL_FORCEINLINE cudaError_t
  Invoke(InitKernelT init_kernel, UniqueByKeySweepKernelT sweep_kernel, ActivePolicyT policy = {})
  {
    // Number of input tiles
    const auto block_threads = VSMemHelperT::template BlockThreads<
      typename ActivePolicyT::UniqueByKeyPolicyT,
      KeyInputIteratorT,
      ValueInputIteratorT,
      KeyOutputIteratorT,
      ValueOutputIteratorT,
      EqualityOpT,
      OffsetT>(policy.UniqueByKey());
    const auto items_per_thread = VSMemHelperT::template ItemsPerThread<
      typename ActivePolicyT::UniqueByKeyPolicyT,
      KeyInputIteratorT,
      ValueInputIteratorT,
      KeyOutputIteratorT,
      ValueOutputIteratorT,
      EqualityOpT,
      OffsetT>(policy.UniqueByKey());
    int tile_size = block_threads * items_per_thread;
    int num_tiles = static_cast<int>(::cuda::ceil_div(num_items, tile_size));
    const auto vsmem_size =
      num_tiles
      * VSMemHelperT::template VSMemPerBlock<
        typename ActivePolicyT::UniqueByKeyPolicyT,
        KeyInputIteratorT,
        ValueInputIteratorT,
        KeyOutputIteratorT,
        ValueOutputIteratorT,
        EqualityOpT,
        OffsetT>(policy.UniqueByKey());

    // Specify temporary storage allocation requirements
    size_t allocation_sizes[2] = {0, vsmem_size};

    auto tile_state = kernel_source.TileState();

    // Bytes needed for tile status descriptors
    if (const auto error = CubDebug(tile_state.AllocationSize(num_tiles, allocation_sizes[0])))
    {
      return error;
    }

    // Compute allocation pointers into the single storage blob (or compute the necessary size of the blob)
    void* allocations[2] = {nullptr, nullptr};
    if (const auto error =
          CubDebug(detail::alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes)))
    {
      return error;
    }

    if (d_temp_storage == nullptr)
    {
      // Return if the caller is simply requesting the size of the storage allocation
      return cudaSuccess;
    }

    if (const auto error = CubDebug(tile_state.Init(num_tiles, allocations[0], allocation_sizes[0])))
    {
      return error;
    }

    // Log init_kernel configuration
    num_tiles                = ::cuda::std::max(1, num_tiles);
    const int init_grid_size = ::cuda::ceil_div(num_tiles, INIT_KERNEL_THREADS);

#ifdef CUB_DEBUG_LOG
    _CubLog("Invoking init_kernel<<<%d, %d, 0, %lld>>>()\n", init_grid_size, INIT_KERNEL_THREADS, (long long) stream);
#endif // CUB_DEBUG_LOG

    // Invoke init_kernel to initialize tile descriptors
    launcher_factory(init_grid_size, INIT_KERNEL_THREADS, 0, stream)
      .doit(init_kernel, tile_state, num_tiles, d_num_selected_out);

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

    // Return if empty problem
    if (num_items == 0)
    {
      return cudaSuccess;
    }

    constexpr int max_dim_x = INT_MAX; // Since sm30

    // Get grid size for scanning tiles
    dim3 scan_grid_size;
    scan_grid_size.z = 1;
    scan_grid_size.y = ::cuda::ceil_div(num_tiles, max_dim_x);
    scan_grid_size.x = ::cuda::std::min(num_tiles, max_dim_x);

    // Log select_if_kernel configuration
#ifdef CUB_DEBUG_LOG
    {
      // Get SM occupancy for unique_by_key_kernel
      int sweep_sm_occupancy;
      if (const auto error = CubDebug(launcher_factory.MaxSmOccupancy(
            sweep_sm_occupancy, // out
            sweep_kernel,
            block_threads)))
      {
        return error;
      }

      _CubLog("Invoking unique_by_key_kernel<<<{%d,%d,%d}, %d, 0, "
              "%lld>>>(), %d items per thread, %d SM occupancy\n",
              scan_grid_size.x,
              scan_grid_size.y,
              scan_grid_size.z,
              block_threads,
              (long long) stream,
              items_per_thread,
              sweep_sm_occupancy);
    }
#endif // CUB_DEBUG_LOG

    // Invoke select_if_kernel
    if (const auto error = CubDebug(
          launcher_factory(scan_grid_size, block_threads, 0, stream)
            .doit(sweep_kernel,
                  d_keys_in,
                  d_values_in,
                  d_keys_out,
                  d_values_out,
                  d_num_selected_out,
                  tile_state,
                  equality_op,
                  num_items,
                  num_tiles,
                  cub::detail::vsmem_t{allocations[1]})))
    {
      return error;
    }

    // Sync the stream if specified to flush runtime errors
    return CubDebug(detail::DebugSyncStream(stream));
  }

  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_HOST _CCCL_FORCEINLINE cudaError_t Invoke(ActivePolicyT active_policy = {})
  {
    auto wrapped_policy = detail::unique_by_key::MakeUniqueByKeyPolicyWrapper(active_policy);

    return Invoke(kernel_source.CompactInitKernel(), kernel_source.UniqueByKeySweepKernel(), wrapped_policy);
  }

  /**
   * @brief Internal dispatch routine
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage.
   *   When nullptr, the required allocation size is written to
   *   `temp_storage_bytes` and no work is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_keys_in
   *   Pointer to the input sequence of keys
   *
   * @param[in] d_values_in
   *   Pointer to the input sequence of values
   *
   * @param[out] d_keys_out
   *   Pointer to the output sequence of selected data items
   *
   * @param[out] d_values_out
   *   Pointer to the output sequence of selected data items
   *
   * @param[out] d_num_selected_out
   *   Pointer to the total number of items selected
   *   (i.e., length of @p d_keys_out or @p d_values_out)
   *
   * @param[in] equality_op
   *   Equality operator
   *
   * @param[in] num_items
   *   Total number of input items (i.e., the length of @p d_in)
   *
   * @param[in] stream
   *   **[optional]** CUDA stream to launch kernels within.
   *   Default is stream<sub>0</sub>.
   */
  template <typename MaxPolicyT = typename PolicyHub::MaxPolicy>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyInputIteratorT d_keys_in,
    ValueInputIteratorT d_values_in,
    KeyOutputIteratorT d_keys_out,
    ValueOutputIteratorT d_values_out,
    NumSelectedIteratorT d_num_selected_out,
    EqualityOpT equality_op,
    OffsetT num_items,
    cudaStream_t stream,
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
    DispatchUniqueByKey dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_values_in,
      d_keys_out,
      d_values_out,
      d_num_selected_out,
      equality_op,
      num_items,
      stream,
      kernel_source,
      launcher_factory);

    // Dispatch to chained policy
    return CubDebug(max_policy.Invoke(ptx_version, dispatch));
  }
};

CUB_NAMESPACE_END
