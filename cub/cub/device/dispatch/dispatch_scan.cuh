// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2026, NVIDIA CORPORATION. All rights reserved.
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
#include <cub/detail/arch_dispatch.cuh>
#include <cub/detail/launcher/cuda_runtime.cuh>
#include <cub/detail/warpspeed/warpspeed.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/device/dispatch/kernels/kernel_scan.cuh>
#include <cub/device/dispatch/tuning/tuning_scan.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>
#include <thrust/type_traits/unwrap_contiguous_iterator.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__iterator/readable_traits.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/move.h>

#include <cuda_runtime_api.h>
#include <cudaTypedefs.h>

CUB_NAMESPACE_BEGIN

namespace detail::scan
{
template <typename PolicySelector,
          typename UnwrappedInputIteratorT,
          typename UnwrappedOutputIteratorT,
          typename ScanOpT,
          typename InitValueT,
          typename OffsetT,
          typename AccumT,
          ForceInclusive EnforceInclusive>
struct DeviceScanKernelSource
{
  using ScanTileStateT = ScanTileState<AccumT>;

  CUB_DEFINE_KERNEL_GETTER(
    InitKernel,
    DeviceScanInitKernel<PolicySelector, UnwrappedInputIteratorT, UnwrappedOutputIteratorT, ScanTileStateT, AccumT>)

  CUB_DEFINE_KERNEL_GETTER(
    ScanKernel,
    DeviceScanKernel<PolicySelector,
                     UnwrappedInputIteratorT,
                     UnwrappedOutputIteratorT,
                     ScanTileStateT,
                     ScanOpT,
                     InitValueT,
                     OffsetT,
                     AccumT,
                     EnforceInclusive == ForceInclusive::Yes>)

  CUB_RUNTIME_FUNCTION static constexpr ::cuda::std::size_t AccumSize()
  {
    return sizeof(AccumT);
  }

  CUB_RUNTIME_FUNCTION static ScanTileStateT TileState()
  {
    return {};
  }

  CUB_RUNTIME_FUNCTION static constexpr ::cuda::std::size_t look_ahead_tile_state_size()
  {
    return sizeof(warpspeed::tile_state_t<AccumT>);
  }

  CUB_RUNTIME_FUNCTION static constexpr ::cuda::std::size_t look_ahead_tile_state_alignment()
  {
    return alignof(warpspeed::tile_state_t<AccumT>);
  }

  CUB_RUNTIME_FUNCTION static constexpr auto make_tile_state_kernel_arg(ScanTileStateT ts)
  {
    tile_state_kernel_arg_t<ScanTileStateT, AccumT> arg;
    ::cuda::std::__construct_at(&arg.lookback, ::cuda::std::move(ts));
    return arg;
  }

  CUB_RUNTIME_FUNCTION static constexpr auto look_ahead_make_tile_state_kernel_arg(void* ts)
  {
    tile_state_kernel_arg_t<ScanTileStateT, AccumT> arg;
    ::cuda::std::__construct_at(&arg.lookahead, static_cast<warpspeed::tile_state_t<AccumT>*>(ts));
    return arg;
  }

  CUB_RUNTIME_FUNCTION static constexpr bool use_warpspeed(const scan_policy& policy)
  {
#if _CCCL_CUDACC_AT_LEAST(12, 8)
    if (policy.warpspeed)
    {
      return detail::scan::use_warpspeed<UnwrappedInputIteratorT, UnwrappedOutputIteratorT, AccumT>(*policy.warpspeed);
    }
#else
    (void) policy;
#endif
    return false;
  }
};

// TODO(griwes): remove in CCCL 4.0 when we drop the scan dispatcher after publishing the tuning API
template <typename T, typename = void>
struct has_warpspeed_policy : ::cuda::std::false_type
{};

template <typename T>
struct has_warpspeed_policy<T, ::cuda::std::void_t<typename T::WarpspeedPolicy>> : ::cuda::std::true_type
{};

template <typename LegacyActivePolicy>
_CCCL_API constexpr auto convert_warpspeed_policy() -> ::cuda::std::optional<scan_warpspeed_policy>
{
#if _CCCL_CUDACC_AT_LEAST(12, 8)
  if constexpr (has_warpspeed_policy<LegacyActivePolicy>::value)
  {
    return make_scan_warpspeed_policy<typename LegacyActivePolicy::WarpspeedPolicy>();
  }
#endif // _CCCL_CUDACC_AT_LEAST(12, 8)
  return ::cuda::std::nullopt;
}

// TODO(griwes): remove in CCCL 4.0 when we drop the scan dispatcher after publishing the tuning API
template <typename LegacyActivePolicy>
_CCCL_API constexpr auto convert_policy() -> scan_policy
{
  using scan_policy_t = typename LegacyActivePolicy::ScanPolicyT;
  return scan_policy{
    scan_policy_t::BLOCK_THREADS,
    scan_policy_t::ITEMS_PER_THREAD,
    scan_policy_t::LOAD_ALGORITHM,
    scan_policy_t::LOAD_MODIFIER,
    scan_policy_t::STORE_ALGORITHM,
    scan_policy_t::SCAN_ALGORITHM,
    detail::delay_constructor_policy_from_type<typename scan_policy_t::detail::delay_constructor_t>,
    convert_warpspeed_policy<LegacyActivePolicy>()};
}

// TODO(griwes): remove in CCCL 4.0 when we drop the scan dispatcher after publishing the tuning API
template <typename PolicyHub>
struct policy_selector_from_hub
{
  // this is only called in device code
  _CCCL_DEVICE_API constexpr auto operator()(::cuda::arch_id /*arch*/) const -> scan_policy
  {
    return convert_policy<typename PolicyHub::MaxPolicy::ActivePolicy>();
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
// TODO(griwes): deprecate when we make the tuning API public and remove in CCCL 4.0
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
    detail::scan::policy_selector_from_hub<PolicyHub>,
    THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator_t<InputIteratorT>,
    THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator_t<OutputIteratorT>,
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
  // TODO(griwes): deprecate when we make the tuning API public and remove in CCCL 4.0
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
    if (const auto error = CubDebug(
          launcher_factory(init_grid_size, INIT_KERNEL_THREADS, 0, stream, /* use_pdl */ true)
            .doit(init_kernel, kernel_source.make_tile_state_kernel_arg(tile_state), num_tiles)))
    {
      return error;
    }

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
      if (const auto error = CubDebug(
            launcher_factory(scan_grid_size, policy.Scan().BlockThreads(), 0, stream, /* use_pdl */ true)
              .doit(scan_kernel,
                    THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(d_in),
                    THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(d_out),
                    kernel_source.make_tile_state_kernel_arg(tile_state),
                    start_tile,
                    scan_op,
                    init_value,
                    num_items,
                    /* num_stages, unused */ 1)))
      {
        return error;
      }

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

#if _CCCL_CUDACC_AT_LEAST(12, 8)
  template <typename PolicySelectorT>
  CUB_RUNTIME_FUNCTION _CCCL_HOST _CCCL_FORCEINLINE cudaError_t __invoke_lookahead_algorithm(
    const detail::scan::scan_warpspeed_policy& warpspeed_policy, const PolicySelectorT& policy_selector)
  {
    const int grid_dim =
      static_cast<int>(::cuda::ceil_div(num_items, static_cast<OffsetT>(warpspeed_policy.tile_size)));

    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = static_cast<size_t>(grid_dim) * kernel_source.look_ahead_tile_state_size();
      return cudaSuccess;
    }

    if (num_items == 0)
    {
      return cudaSuccess;
    }

    int sm_count = 0;
    if (const auto error = CubDebug(launcher_factory.MultiProcessorCount(sm_count)))
    {
      return error;
    }
    // number of stages to have an even workload across all SMs (improves small problem sizes), assuming 1 CTA per SM
    // +1 since it tends to improve performance
    // TODO(bgruber): make the +1 a tuning parameter
    const int max_stages_for_even_workload =
      static_cast<int>(::cuda::ceil_div(num_items, static_cast<OffsetT>(sm_count * warpspeed_policy.tile_size)) + 1);

    // Maximum dynamic shared memory size that we can use for temporary storage.
    int max_dynamic_smem_size{};
    if (const auto error =
          CubDebug(launcher_factory.max_dynamic_smem_size_for(max_dynamic_smem_size, kernel_source.ScanKernel())))
    {
      return error;
    }

    // TODO(bgruber): we probably need to ensure alignment of d_temp_storage
    _CCCL_ASSERT(::cuda::is_aligned(d_temp_storage, kernel_source.look_ahead_tile_state_alignment()), "");

    auto scan_kernel = kernel_source.ScanKernel();
    int num_stages   = 1;
    int smem_size    = detail::scan::smem_for_stages(
      warpspeed_policy,
      num_stages,
      policy_selector.input_value_size,
      policy_selector.input_value_alignment,
      policy_selector.output_value_size,
      policy_selector.output_value_alignment,
      policy_selector.accum_size,
      policy_selector.accum_alignment);

    // When launched from the host, maximize the number of stages that we can fit inside the shared memory.
    NV_IF_TARGET(NV_IS_HOST, ({
                   while (num_stages <= max_stages_for_even_workload)
                   {
                     const auto next_smem_size = detail::scan::smem_for_stages(
                       warpspeed_policy,
                       num_stages + 1,
                       policy_selector.input_value_size,
                       policy_selector.input_value_alignment,
                       policy_selector.output_value_size,
                       policy_selector.output_value_alignment,
                       policy_selector.accum_size,
                       policy_selector.accum_alignment);
                     if (next_smem_size > max_dynamic_smem_size)
                     {
                       // This number of stages failed, so stay at the current settings
                       break;
                     }

                     smem_size = next_smem_size;
                     ++num_stages;
                   }

                   if (const auto error = launcher_factory.set_max_dynamic_smem_size_for(scan_kernel, smem_size))
                   {
                     return error;
                   }
                 }))

    // Invoke init kernel
    {
      constexpr auto init_kernel_threads = 128;
      const auto init_grid_size          = ::cuda::ceil_div(grid_dim, init_kernel_threads);

#  ifdef CUB_DEBUG_LOG
      _CubLog("Invoking DeviceScanInitKernel<<<%d, %d, 0, , %lld>>>()\n",
              init_grid_size,
              init_kernel_threads,
              (long long) stream);
#  endif // CUB_DEBUG_LOG

      if (const auto error = CubDebug(
            launcher_factory(init_grid_size, init_kernel_threads, 0, stream, /* use_pdl */ true)
              .doit(kernel_source.InitKernel(),
                    kernel_source.look_ahead_make_tile_state_kernel_arg(d_temp_storage),
                    grid_dim)))
      {
        return error;
      }

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

    // Invoke scan kernel
    {
      const int block_dim = warpspeed_policy.num_total_threads;

#  ifdef CUB_DEBUG_LOG
      _CubLog("Invoking DeviceScanKernel<<<%d, %d, %d, %lld>>>()\n", grid_dim, block_dim, smem_size, (long long) stream);
#  endif // CUB_DEBUG_LOG

      if (const auto error = CubDebug(
            launcher_factory(grid_dim, block_dim, smem_size, stream, /* use_pdl */ true)
              .doit(scan_kernel,
                    THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(d_in),
                    THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(d_out),
                    kernel_source.look_ahead_make_tile_state_kernel_arg(d_temp_storage),
                    /* start_tile, unused */ 0,
                    ::cuda::std::move(scan_op),
                    init_value,
                    num_items,
                    num_stages)))
      {
        return error;
      }

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
#endif // _CCCL_CUDACC_AT_LEAST(12, 8)

  template <typename PolicyGetter, typename PolicySelectorT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
  __invoke(PolicyGetter policy_getter, const PolicySelectorT& policy_selector)
  {
    CUB_DETAIL_CONSTEXPR_ISH auto active_policy = policy_getter();

    CUB_DETAIL_STATIC_ISH_ASSERT(active_policy.load_modifier != CacheLoadModifier::LOAD_LDG,
                                 "The memory consistency model does not apply to texture accesses");

#if !_CCCL_CUDACC_AT_LEAST(12, 8)
    (void) policy_selector;
#endif // !_CCCL_CUDACC_AT_LEAST(12, 8)

#if _CCCL_CUDACC_AT_LEAST(12, 8)
    if (kernel_source.use_warpspeed(active_policy))
    {
      return __invoke_lookahead_algorithm(*active_policy.warpspeed, policy_selector);
    }
#endif // _CCCL_CUDACC_AT_LEAST(12, 8)

    // Number of input tiles
    const int tile_size = active_policy.block_threads * active_policy.items_per_thread;
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
    constexpr int init_kernel_threads = 128;
    const int init_grid_size          = ::cuda::ceil_div(num_tiles, init_kernel_threads);

#ifdef CUB_DEBUG_LOG
    _CubLog("Invoking init_kernel<<<%d, %d, 0, %lld>>>()\n", init_grid_size, init_kernel_threads, (long long) stream);
#endif // CUB_DEBUG_LOG

    // Invoke init_kernel to initialize tile descriptors
    if (const auto error = CubDebug(
          launcher_factory(init_grid_size, init_kernel_threads, 0, stream, /* use_pdl */ true)
            .doit(kernel_source.InitKernel(), kernel_source.make_tile_state_kernel_arg(tile_state), num_tiles)))
    {
      return error;
    }

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
    if (const auto error = CubDebug(
          launcher_factory.MaxSmOccupancy(scan_sm_occupancy, kernel_source.ScanKernel(), active_policy.block_threads)))
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
              active_policy.block_threads,
              (long long) stream,
              active_policy.items_per_thread,
              scan_sm_occupancy);
#endif // CUB_DEBUG_LOG

      // Invoke scan_kernel
      if (const auto error = CubDebug(
            launcher_factory(scan_grid_size, active_policy.block_threads, 0, stream, /* use_pdl */ true)
              .doit(kernel_source.ScanKernel(),
                    THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(d_in),
                    THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(d_out),
                    kernel_source.make_tile_state_kernel_arg(tile_state),
                    start_tile,
                    scan_op,
                    init_value,
                    num_items,
                    /* num_stages, unused */ 1)))
      {
        return error;
      }

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
  CUB_RUNTIME_FUNCTION _CCCL_HOST _CCCL_FORCEINLINE cudaError_t Invoke(ActivePolicyT = {})
  {
    struct policy_getter
    {
      _CCCL_API _CCCL_FORCEINLINE constexpr auto operator()() const
      {
        return detail::scan::convert_policy<ActivePolicyT>();
      }
    };

    using policy_selector_t = detail::scan::policy_selector_from_types<
      detail::it_value_t<InputIteratorT>,
      detail::it_value_t<OutputIteratorT>,
      AccumT,
      OffsetT,
      ScanOpT>;
    return __invoke(policy_getter{}, policy_selector_t{});
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
  // TODO(griwes): deprecate when we make the tuning API public and remove in CCCL 4.0
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

namespace detail::scan
{
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
  typename PolicySelector         = policy_selector_from_types<detail::it_value_t<InputIteratorT>,
                                                               detail::it_value_t<OutputIteratorT>,
                                                               AccumT,
                                                               OffsetT,
                                                               ScanOpT>,
  typename KernelSource =
    DeviceScanKernelSource<PolicySelector, InputIteratorT, OutputIteratorT, ScanOpT, InitValueT, OffsetT, AccumT, EnforceInclusive>,
  typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
#if _CCCL_HAS_CONCEPTS()
  requires scan_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE auto dispatch(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  InputIteratorT d_in,
  OutputIteratorT d_out,
  ScanOpT scan_op,
  InitValueT init_value,
  OffsetT num_items,
  cudaStream_t stream,
  PolicySelector policy_selector         = {},
  KernelSource kernel_source             = {},
  KernelLauncherFactory launcher_factory = {}) -> cudaError_t
{
  static_assert(::cuda::std::is_unsigned_v<OffsetT> && sizeof(OffsetT) >= 4,
                "DispatchScan only supports unsigned offset types of at least 4-bytes");

  ::cuda::arch_id arch_id{};
  if (const auto error = CubDebug(launcher_factory.PtxArchId(arch_id)))
  {
    return error;
  }

#if !_CCCL_COMPILER(NVRTC) && defined(CUB_DEBUG_LOG)
  NV_IF_TARGET(NV_IS_HOST,
               (std::stringstream ss; ss << policy_selector(arch_id);
                _CubLog("Dispatching DeviceScan to arch %d with tuning: %s\n", (int) arch_id, ss.str().c_str());))
#endif // !_CCCL_COMPILER(NVRTC) && defined(CUB_DEBUG_LOG)

  struct fake_policy
  {
    using MaxPolicy = void;
  };

  return dispatch_arch(policy_selector, arch_id, [&](auto policy_getter) {
    return DispatchScan<InputIteratorT,
                        OutputIteratorT,
                        ScanOpT,
                        InitValueT,
                        OffsetT,
                        AccumT,
                        EnforceInclusive,
                        fake_policy,
                        KernelSource,
                        KernelLauncherFactory>{
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      num_items,
      scan_op,
      init_value,
      stream,
      -1 /* ptx_version, not used actually */,
      kernel_source,
      launcher_factory}
      .__invoke(policy_getter, policy_selector);
  });
}

template <
  typename AccumT,
  typename InputIteratorT,
  typename OutputIteratorT,
  typename ScanOpT,
  typename InitValueT,
  typename OffsetT,
  ForceInclusive EnforceInclusive = ForceInclusive::No,
  typename PolicySelector         = policy_selector_from_types<detail::it_value_t<InputIteratorT>,
                                                               detail::it_value_t<OutputIteratorT>,
                                                               AccumT,
                                                               OffsetT,
                                                               ScanOpT>,
  typename KernelSource =
    DeviceScanKernelSource<PolicySelector, InputIteratorT, OutputIteratorT, ScanOpT, InitValueT, OffsetT, AccumT, EnforceInclusive>,
  typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE auto dispatch_with_accum(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  InputIteratorT d_in,
  OutputIteratorT d_out,
  ScanOpT scan_op,
  InitValueT init_value,
  OffsetT num_items,
  cudaStream_t stream,
  PolicySelector policy_selector         = {},
  KernelSource kernel_source             = {},
  KernelLauncherFactory launcher_factory = {}) -> cudaError_t
{
  return dispatch<InputIteratorT, OutputIteratorT, ScanOpT, InitValueT, OffsetT, AccumT>(
    d_temp_storage,
    temp_storage_bytes,
    d_in,
    d_out,
    scan_op,
    init_value,
    num_items,
    stream,
    policy_selector,
    kernel_source,
    launcher_factory);
}
} // namespace detail::scan

CUB_NAMESPACE_END
