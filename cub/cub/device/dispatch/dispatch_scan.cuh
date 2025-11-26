// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * @file cub::DeviceScan provides device-wide, parallel operations for
 *       computing a prefix scan across a sequence of data items residing
 *       within device-accessible memory.
 */

#pragma once

#include <cub/config.cuh>

#include <cub/util_namespace.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_scan.cuh>
#include <cub/detail/launcher/cuda_runtime.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/device/dispatch/kernels/kernel_scan.cuh>
#include <cub/device/dispatch/tuning/tuning_scan.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_unsigned.h>

CUB_NAMESPACE_BEGIN

namespace detail::scan
{
template <typename MaxPolicyT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename ScanOpT,
          typename InitValueT,
          typename OffsetT,
          typename AccumT,
          ForceInclusive EnforceInclusive>
struct DeviceScanKernelSource
{
  using ScanTileStateT = typename cub::ScanTileState<AccumT>;

  CUB_DEFINE_KERNEL_GETTER(InitKernel, DeviceScanInitKernel<ScanTileStateT>)

  CUB_DEFINE_KERNEL_GETTER(
    ScanKernel,
    DeviceScanKernel<MaxPolicyT,
                     InputIteratorT,
                     OutputIteratorT,
                     ScanTileStateT,
                     ScanOpT,
                     InitValueT,
                     OffsetT,
                     AccumT,
                     EnforceInclusive == ForceInclusive::Yes>)

  CUB_RUNTIME_FUNCTION static constexpr size_t AccumSize()
  {
    return sizeof(AccumT);
  }

  CUB_RUNTIME_FUNCTION ScanTileStateT TileState()
  {
    return ScanTileStateT();
  }
};
} // namespace detail::scan

/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * @brief Utility class for dispatching the appropriately-tuned kernels for
 *        DeviceScan
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading scan inputs @iterator
 *
 * @tparam OutputIteratorT
 *   Random-access output iterator type for writing scan outputs @iterator
 *
 * @tparam ScanOpT
 *   Binary scan functor type having member
 *   `auto operator()(const T &a, const U &b)`
 *
 * @tparam InitValueT
 *   The init_value element type for ScanOpT (cub::NullType for inclusive scans)
 *
 * @tparam OffsetT
 *   Unsigned integer type for global offsets
 *
 * @tparam EnforceInclusive
 *   Enum flag to specify whether to enforce inclusive scan.
 *
 */
template <
  typename InputIteratorT,
  typename OutputIteratorT,
  typename ScanOpT,
  typename InitValueT,
  typename OffsetT,
  typename AccumT                 = ::cuda::std::__accumulator_t<ScanOpT,
                                                                 cub::detail::it_value_t<InputIteratorT>,
                                                                 ::cuda::std::_If<::cuda::std::is_same_v<InitValueT, NullType>,
                                                                                  cub::detail::it_value_t<InputIteratorT>,
                                                                                  typename InitValueT::value_type>>,
  ForceInclusive EnforceInclusive = ForceInclusive::No,
  typename PolicyHub              = detail::scan::
    policy_hub<detail::it_value_t<InputIteratorT>, detail::it_value_t<OutputIteratorT>, AccumT, OffsetT, ScanOpT>,
  typename KernelSource = detail::scan::DeviceScanKernelSource<
    typename PolicyHub::MaxPolicy,
    InputIteratorT,
    OutputIteratorT,
    ScanOpT,
    InitValueT,
    OffsetT,
    AccumT,
    EnforceInclusive>,
  typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
struct DispatchScan
{
  static_assert(::cuda::std::is_unsigned_v<OffsetT> && sizeof(OffsetT) >= 4,
                "DispatchScan only supports unsigned offset types of at least 4-bytes");

  //---------------------------------------------------------------------
  // Constants and Types
  //---------------------------------------------------------------------

  static constexpr int INIT_KERNEL_THREADS = 128;

  /// Device-accessible allocation of temporary storage.  When nullptr, the
  /// required allocation size is written to \p temp_storage_bytes and no work
  /// is done.
  void* d_temp_storage;

  /// Reference to size in bytes of \p d_temp_storage allocation
  size_t& temp_storage_bytes;

  /// Iterator to the input sequence of data items
  InputIteratorT d_in;

  /// Iterator to the output sequence of data items
  OutputIteratorT d_out;

  /// Binary scan functor
  ScanOpT scan_op;

  /// Initial value to seed the exclusive scan
  InitValueT init_value;

  /// Total number of input items (i.e., the length of \p d_in)
  OffsetT num_items;

  /// CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
  cudaStream_t stream;

  int ptx_version;

  KernelSource kernel_source;

  KernelLauncherFactory launcher_factory;

  /**
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to `temp_storage_bytes` and no
   *   work is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_in
   *   Iterator to the input sequence of data items
   *
   * @param[out] d_out
   *   Iterator to the output sequence of data items
   *
   * @param[in] num_items
   *   Total number of input items (i.e., the length of `d_in`)
   *
   * @param[in] scan_op
   *   Binary scan functor
   *
   * @param[in] init_value
   *   Initial value to seed the exclusive scan
   *
   * @param[in] stream
   *   **[optional]** CUDA stream to launch kernels within.
   *   Default is stream<sub>0</sub>.
   *
   * @param[in] kernel_source
   *   Object specifying implementation kernels
   *
   * @param[in] launcher_factory
   *   Object to execute implementation kernels on the given stream
   *
   * @param[in] max_policy
   *   Struct encoding chain of algorithm tuning policies
   */
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchScan(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    OffsetT num_items,
    ScanOpT scan_op,
    InitValueT init_value,
    cudaStream_t stream,
    int ptx_version,
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {})
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_in(d_in)
      , d_out(d_out)
      , scan_op(scan_op)
      , init_value(init_value)
      , num_items(num_items)
      , stream(stream)
      , ptx_version(ptx_version)
      , kernel_source(kernel_source)
      , launcher_factory(launcher_factory)
  {}

  template <typename ActivePolicyT, typename InitKernelT, typename ScanKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_HOST _CCCL_FORCEINLINE cudaError_t
  Invoke(InitKernelT init_kernel, ScanKernelT scan_kernel, ActivePolicyT policy = {})
  {
    // `LOAD_LDG` makes in-place execution UB and doesn't lead to better
    // performance.
    policy.CheckLoadModifier();

    // Number of input tiles
    const int tile_size = policy.Scan().BlockThreads() * policy.Scan().ItemsPerThread();
    const int num_tiles = static_cast<int>(::cuda::ceil_div(num_items, tile_size));

    auto tile_state = kernel_source.TileState();

    // Specify temporary storage allocation requirements
    size_t allocation_sizes[1];
    if (const auto error = CubDebug(tile_state.AllocationSize(num_tiles, allocation_sizes[0])))
    {
      return error; // bytes needed for tile status descriptors
    }

    // Compute allocation pointers into the single storage blob (or compute
    // the necessary size of the blob)
    void* allocations[1] = {};
    if (const auto error =
          CubDebug(detail::alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes)))
    {
      return error;
    }

    // Return if the caller is simply requesting the size of the storage allocation, or the problem is empty
    if (d_temp_storage == nullptr || num_items == 0)
    {
      return cudaSuccess;
    }

    // Construct the tile status interface
    if (const auto error = CubDebug(tile_state.Init(num_tiles, allocations[0], allocation_sizes[0])))
    {
      return error;
    }

    // Log init_kernel configuration
    const int init_grid_size = ::cuda::ceil_div(num_tiles, INIT_KERNEL_THREADS);

#ifdef CUB_DEBUG_LOG
    _CubLog("Invoking init_kernel<<<%d, %d, 0, %lld>>>()\n", init_grid_size, INIT_KERNEL_THREADS, (long long) stream);
#endif // CUB_DEBUG_LOG

    // Invoke init_kernel to initialize tile descriptors
    launcher_factory(init_grid_size, INIT_KERNEL_THREADS, 0, stream, /* use_pdl */ true)
      .doit(init_kernel, tile_state, num_tiles);

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

    // Get SM occupancy for scan_kernel
    int scan_sm_occupancy;
    if (const auto error =
          CubDebug(launcher_factory.MaxSmOccupancy(scan_sm_occupancy, scan_kernel, policy.Scan().BlockThreads())))
    {
      return error;
    }

    // Get max x-dimension of grid
    int max_dim_x;
    if (const auto error = CubDebug(launcher_factory.MaxGridDimX(max_dim_x)))
    {
      return error;
    }

    // Run grids in epochs (in case number of tiles exceeds max x-dimension
    const int scan_grid_size = ::cuda::std::min(num_tiles, max_dim_x);
    for (int start_tile = 0; start_tile < num_tiles; start_tile += scan_grid_size)
    {
// Log scan_kernel configuration
#ifdef CUB_DEBUG_LOG
      _CubLog("Invoking %d scan_kernel<<<%d, %d, 0, %lld>>>(), %d items "
              "per thread, %d SM occupancy\n",
              start_tile,
              scan_grid_size,
              policy.Scan().BlockThreads(),
              (long long) stream,
              policy.Scan().ItemsPerThread(),
              scan_sm_occupancy);
#endif // CUB_DEBUG_LOG

      // Invoke scan_kernel
      launcher_factory(scan_grid_size, policy.Scan().BlockThreads(), 0, stream, /* use_pdl */ true)
        .doit(scan_kernel, d_in, d_out, tile_state, start_tile, scan_op, init_value, num_items);

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
    }

    return cudaSuccess;
  }

  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_HOST _CCCL_FORCEINLINE cudaError_t Invoke(ActivePolicyT active_policy = {})
  {
    auto wrapped_policy = detail::scan::MakeScanPolicyWrapper(active_policy);
    // Ensure kernels are instantiated.
    return Invoke(kernel_source.InitKernel(), kernel_source.ScanKernel(), wrapped_policy);
  }

  /**
   * @brief Internal dispatch routine
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to `temp_storage_bytes` and no
   *   work is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_in
   *   Iterator to the input sequence of data items
   *
   * @param[out] d_out
   *   Iterator to the output sequence of data items
   *
   * @param[in] scan_op
   *   Binary scan functor
   *
   * @param[in] init_value
   *   Initial value to seed the exclusive scan
   *
   * @param[in] num_items
   *   Total number of input items (i.e., the length of `d_in`)
   *
   * @param[in] stream
   *   **[optional]** CUDA stream to launch kernels within.
   *   Default is stream<sub>0</sub>.
   *
   * @param[in] kernel_source
   *   Object specifying implementation kernels
   *
   * @param[in] launcher_factory
   *   Object to execute implementation kernels on the given stream
   *
   * @param[in] max_policy
   *   Struct encoding chain of algorithm tuning policies
   */
  template <typename MaxPolicyT = typename PolicyHub::MaxPolicy>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ScanOpT scan_op,
    InitValueT init_value,
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
    DispatchScan dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      num_items,
      scan_op,
      init_value,
      stream,
      ptx_version,
      kernel_source,
      launcher_factory);

    // Dispatch to chained policy
    return CubDebug(max_policy.Invoke(ptx_version, dispatch));
  }
};

CUB_NAMESPACE_END
