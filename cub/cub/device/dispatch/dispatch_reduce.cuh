// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * @file
 * @brief cub::DeviceReduce provides device-wide, parallel operations for
 *        computing a reduction across a sequence of data items residing within
 *        device-accessible memory.
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

#include <cub/detail/cc_dispatch.cuh>
#include <cub/detail/launcher/cuda_runtime.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/device/dispatch/kernels/kernel_reduce.cuh>
#include <cub/device/dispatch/tuning/tuning_reduce.cuh>
#include <cub/grid/grid_even_share.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_temporary_storage.cuh>
#include <cub/util_type.cuh> // for cub::detail::non_void_value_t, cub::detail::it_value_t

#include <cuda/std/__functional/identity.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__host_stdlib/sstream>
#include <cuda/std/cstdint>

// TODO(bgruber): included to not break users when moving DeviceSegmentedReduce to its own file. Remove in CCCL 4.0.
#include <cub/device/dispatch/dispatch_segmented_reduce.cuh>

CUB_NAMESPACE_BEGIN

namespace detail::reduce
{
template <typename PolicySelector,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ReductionOpT,
          typename InitValueT,
          typename AccumT,
          typename TransformOpT,
          bool StableReductionOrder = true>
struct DeviceReduceKernelSource
{
  // PolicySelector must be stateless, so we can pass the type to the kernel
  static_assert(::cuda::std::is_empty_v<PolicySelector>);

  CUB_DEFINE_KERNEL_GETTER(
    SingleTileKernel,
    DeviceReduceSingleTileKernel<PolicySelector,
                                 InputIteratorT,
                                 OutputIteratorT,
                                 OffsetT,
                                 ReductionOpT,
                                 InitValueT,
                                 AccumT,
                                 TransformOpT>)

  // The atomic code path finishes in one kernel, the two-phase code path writes to an intermediate buffer of
  // accumulators
  using reduce_kernel_output_t = ::cuda::std::conditional_t<StableReductionOrder, AccumT*, OutputIteratorT>;
  CUB_DEFINE_KERNEL_GETTER(
    ReductionKernel,
    DeviceReduceKernel<PolicySelector,
                       StableReductionOrder,
                       InputIteratorT,
                       reduce_kernel_output_t,
                       OffsetT,
                       ReductionOpT,
                       AccumT,
                       InitValueT,
                       TransformOpT>)

  CUB_DEFINE_KERNEL_GETTER(
    SingleTileSecondKernel,
    DeviceReduceSingleTileKernel<PolicySelector,
                                 AccumT*,
                                 OutputIteratorT,
                                 int, // Always used with int offsets
                                 ReductionOpT,
                                 InitValueT,
                                 AccumT>)

  CUB_RUNTIME_FUNCTION static constexpr size_t AccumSize()
  {
    return sizeof(AccumT);
  }

  CUB_RUNTIME_FUNCTION static constexpr size_t InitSize()
  {
    return sizeof(InitValueT);
  }
};

// TODO(bgruber): remove in CCCL 4.0
template <typename PolicyHub>
struct policy_selector_from_hub
{
  // this is only called in device code, so we can ignore the arch parameter
  _CCCL_DEVICE_API constexpr auto operator()(::cuda::compute_capability) const -> reduce_policy
  {
    using ap             = typename PolicyHub::MaxPolicy::ActivePolicy;
    using ap_reduce      = typename ap::ReducePolicy;
    using ap_single_tile = typename ap::SingleTilePolicy;
    return reduce_policy{
      agent_reduce_policy{
        ap_reduce::BLOCK_THREADS,
        ap_reduce::ITEMS_PER_THREAD,
        ap_reduce::VECTOR_LOAD_LENGTH,
        ap_reduce::BLOCK_ALGORITHM,
        ap_reduce::LOAD_MODIFIER,
      },
      agent_reduce_policy{
        ap_single_tile::BLOCK_THREADS,
        ap_single_tile::ITEMS_PER_THREAD,
        ap_single_tile::VECTOR_LOAD_LENGTH,
        ap_single_tile::BLOCK_ALGORITHM,
        ap_single_tile::LOAD_MODIFIER,
      }};
  }
};
} // namespace detail::reduce

/******************************************************************************
 * Single-problem dispatch
 *****************************************************************************/

// TODO(bgruber): deprecate once we publish the tuning API
/**
 * @brief Utility class for dispatching the appropriately-tuned kernels for
 *        device-wide reduction
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading input items @iterator
 *
 * @tparam OutputIteratorT
 *   Output iterator type for recording the reduced aggregate @iterator
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam ReductionOpT
 *   Binary reduction functor type having member
 *   `auto operator()(const T &a, const U &b)`
 *
 * @tparam InitValueT
 *   Initial value type
 */
template <
  typename InputIteratorT,
  typename OutputIteratorT,
  typename OffsetT,
  typename ReductionOpT,
  typename InitValueT = cub::detail::non_void_value_t<OutputIteratorT, cub::detail::it_value_t<InputIteratorT>>,
  typename AccumT     = ::cuda::std::__accumulator_t<ReductionOpT, cub::detail::it_value_t<InputIteratorT>, InitValueT>,
  typename TransformOpT = ::cuda::std::identity,
  typename PolicyHub    = detail::reduce::policy_hub<AccumT, OffsetT, ReductionOpT>,
  typename KernelSource = detail::reduce::DeviceReduceKernelSource<
    detail::reduce::policy_selector_from_hub<PolicyHub>,
    InputIteratorT,
    OutputIteratorT,
    OffsetT,
    ReductionOpT,
    InitValueT,
    AccumT,
    TransformOpT>,
  typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
struct DispatchReduce
{
  //---------------------------------------------------------------------------
  // Problem state
  //---------------------------------------------------------------------------

  /// Device-accessible allocation of temporary storage. When `nullptr`, the
  /// required allocation size is written to `temp_storage_bytes` and no work
  /// is done.
  void* d_temp_storage;

  /// Reference to size in bytes of `d_temp_storage` allocation
  size_t& temp_storage_bytes;

  /// Pointer to the input sequence of data items
  InputIteratorT d_in;

  /// Pointer to the output aggregate
  OutputIteratorT d_out;

  /// Total number of input items (i.e., length of `d_in`)
  OffsetT num_items;

  /// Binary reduction functor
  ReductionOpT reduction_op;

  /// The initial value of the reduction
  InitValueT init;

  /// CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
  cudaStream_t stream;

  int ptx_version;

  TransformOpT transform_op;

  KernelSource kernel_source;

  KernelLauncherFactory launcher_factory;

  //---------------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------------

  /// Constructor
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchReduce(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    OffsetT num_items,
    ReductionOpT reduction_op,
    InitValueT init,
    cudaStream_t stream,
    int ptx_version,
    TransformOpT transform_op              = {},
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {})
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_in(d_in)
      , d_out(d_out)
      , num_items(num_items)
      , reduction_op(reduction_op)
      , init(init)
      , stream(stream)
      , ptx_version(ptx_version)
      , transform_op(transform_op)
      , kernel_source(kernel_source)
      , launcher_factory(launcher_factory)
  {}

  //---------------------------------------------------------------------------
  // Small-problem (single tile) invocation
  //---------------------------------------------------------------------------

  /**
   * @brief Invoke a single block block to reduce in-core
   *
   * @tparam ActivePolicyT
   *   Umbrella policy active for the target device
   *
   * @tparam SingleTileKernelT
   *   Function type of cub::DeviceReduceSingleTileKernel
   *
   * @param[in] single_tile_kernel
   *   Kernel function pointer to parameterization of
   *   cub::DeviceReduceSingleTileKernel
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

// Log single_reduce_sweep_kernel configuration
#ifdef CUB_DEBUG_LOG
    _CubLog("Invoking DeviceReduceSingleTileKernel<<<1, %d, 0, %lld>>>(), "
            "%d items per thread\n",
            policy.SingleTile().ThreadsPerBlock(),
            (long long) stream,
            policy.SingleTile().ItemsPerThread());
#endif // CUB_DEBUG_LOG

    // Invoke single_reduce_sweep_kernel
    launcher_factory(1, policy.SingleTile().ThreadsPerBlock(), 0, stream)
      .doit(single_tile_kernel, d_in, d_out, num_items, reduction_op, init, transform_op);

    // Check for failure to launch
    if (const auto error = CubDebug(cudaPeekAtLastError()))
    {
      return error;
    }

    // Sync the stream if specified to flush runtime errors
    return CubDebug(detail::DebugSyncStream(stream));
  }

  //---------------------------------------------------------------------------
  // Normal problem size invocation (two-pass)
  //---------------------------------------------------------------------------

  /**
   * @brief Invoke two-passes to reduce
   * @tparam ActivePolicyT
   *   Umbrella policy active for the target device
   *
   * @tparam ReduceKernelT
   *   Function type of cub::DeviceReduceKernel
   *
   * @tparam SingleTileKernelT
   *   Function type of cub::DeviceReduceSingleTileKernel
   *
   * @param[in] reduce_kernel
   *   Kernel function pointer to parameterization of cub::DeviceReduceKernel
   *
   * @param[in] single_tile_kernel
   *   Kernel function pointer to parameterization of
   *   cub::DeviceReduceSingleTileKernel
   */
  template <typename ActivePolicyT, typename ReduceKernelT, typename SingleTileKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t
  InvokePasses(ReduceKernelT reduce_kernel, SingleTileKernelT single_tile_kernel, ActivePolicyT active_policy = {})
  {
    // Get SM count
    int sm_count;
    if (const auto error = CubDebug(launcher_factory.MultiProcessorCount(sm_count)))
    {
      return error;
    }

    // Init regular kernel configuration
    detail::KernelConfig reduce_config;
    if (const auto error = CubDebug(reduce_config.Init(reduce_kernel, active_policy.Reduce(), launcher_factory)))
    {
      return error;
    }

    int reduce_device_occupancy = reduce_config.sm_occupancy * sm_count;

    // Even-share work distribution
    int max_blocks = reduce_device_occupancy * detail::subscription_factor;
    GridEvenShare<OffsetT> even_share;
    even_share.DispatchInit(num_items, max_blocks, reduce_config.tile_size);

    // Temporary storage allocation requirements
    void* allocations[1]       = {};
    size_t allocation_sizes[1] = {
      max_blocks * kernel_source.AccumSize() // bytes needed for privatized block
                                             // reductions
    };

    // Alias the temporary allocations from the single storage blob (or
    // compute the necessary size of the blob)
    if (const auto error =
          CubDebug(detail::alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes)))
    {
      return error;
    }

    if (d_temp_storage == nullptr)
    {
      // Return if the caller is simply requesting the size of the storage
      // allocation
      return cudaSuccess;
    }

    // Alias the allocation for the privatized per-block reductions
    AccumT* d_block_reductions = static_cast<AccumT*>(allocations[0]);

    // Get grid size for device_reduce_sweep_kernel
    int reduce_grid_size = even_share.grid_size;

// Log device_reduce_sweep_kernel configuration
#ifdef CUB_DEBUG_LOG
    _CubLog("Invoking DeviceReduceKernel<<<%lu, %d, 0, %lld>>>(), %d items "
            "per thread, %d SM occupancy\n",
            (unsigned long) reduce_grid_size,
            active_policy.Reduce().ThreadsPerBlock(),
            (long long) stream,
            active_policy.Reduce().ItemsPerThread(),
            reduce_config.sm_occupancy);
#endif // CUB_DEBUG_LOG

    // Invoke DeviceReduceKernel
    launcher_factory(reduce_grid_size, active_policy.Reduce().ThreadsPerBlock(), 0, stream)
      .doit(reduce_kernel, d_in, d_block_reductions, num_items, even_share, reduction_op, init, transform_op);

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

// Log single_reduce_sweep_kernel configuration
#ifdef CUB_DEBUG_LOG
    _CubLog("Invoking DeviceReduceSingleTileKernel<<<1, %d, 0, %lld>>>(), "
            "%d items per thread\n",
            active_policy.SingleTile().ThreadsPerBlock(),
            (long long) stream,
            active_policy.SingleTile().ItemsPerThread());
#endif // CUB_DEBUG_LOG

    // Invoke DeviceReduceSingleTileKernel
    launcher_factory(1, active_policy.SingleTile().ThreadsPerBlock(), 0, stream)
      .doit(
        single_tile_kernel, d_block_reductions, d_out, reduce_grid_size, reduction_op, init, ::cuda::std::identity{});

    // Check for failure to launch
    if (const auto error = CubDebug(cudaPeekAtLastError()))
    {
      return error;
    }

    // Sync the stream if specified to flush runtime errors
    return CubDebug(detail::DebugSyncStream(stream));
  }

  //---------------------------------------------------------------------------
  // Chained policy invocation
  //---------------------------------------------------------------------------

  /// Invocation
  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke(ActivePolicyT active_policy = {})
  {
    auto wrapped_policy = detail::reduce::MakeReducePolicyWrapper(active_policy);
    if (num_items <= static_cast<OffsetT>(
          wrapped_policy.SingleTile().ThreadsPerBlock() * wrapped_policy.SingleTile().ItemsPerThread()))
    {
      // Small, single tile size
      return InvokeSingleTile(kernel_source.SingleTileKernel(), wrapped_policy);
    }
    else
    {
      // Regular size
      return InvokePasses(kernel_source.ReductionKernel(), kernel_source.SingleTileSecondKernel(), wrapped_policy);
    }
  }

  //---------------------------------------------------------------------------
  // Dispatch entrypoints
  //---------------------------------------------------------------------------

  /**
   * @brief Internal dispatch routine for computing a device-wide reduction
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to `temp_storage_bytes` and no work
   *   is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_in
   *   Pointer to the input sequence of data items
   *
   * @param[out] d_out
   *   Pointer to the output aggregate
   *
   * @param[in] num_items
   *   Total number of input items (i.e., length of `d_in`)
   *
   * @param[in] reduction_op
   *   Binary reduction functor
   *
   * @param[in] init
   *   The initial value of the reduction
   *
   * @param[in] stream
   *   **[optional]** CUDA stream to launch kernels within.
   *   Default is stream<sub>0</sub>.
   */
  template <typename MaxPolicyT = typename PolicyHub::MaxPolicy>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    OffsetT num_items,
    ReductionOpT reduction_op,
    InitValueT init,
    cudaStream_t stream,
    TransformOpT transform_op              = {},
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
    DispatchReduce dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      num_items,
      reduction_op,
      init,
      stream,
      ptx_version,
      transform_op,
      kernel_source,
      launcher_factory);

    // Ignore Wmaybe-uninitialized to work around a GCC 13 issue:
    // https://github.com/NVIDIA/cccl/issues/4053
    _CCCL_DIAG_PUSH
    _CCCL_DIAG_SUPPRESS_GCC("-Wmaybe-uninitialized")
    // Dispatch to chained policy
    return CubDebug(max_policy.Invoke(ptx_version, dispatch));
    _CCCL_DIAG_POP
  }
};

// TODO(bgruber): deprecate once we publish the tuning API and drop in CCCL 4.0
/**
 * @brief Utility class for dispatching the appropriately-tuned kernels for
 *        device-wide transform reduce
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading input items @iterator
 *
 * @tparam OutputIteratorT
 *   Output iterator type for recording the reduced aggregate @iterator
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam ReductionOpT
 *   Binary reduction functor type having member
 *   `auto operator()(const T &a, const U &b)`
 *
 * @tparam TransformOpT
 *   Unary transform functor type having member
 *   `auto operator()(const T &a)`
 *
 * @tparam InitValueT
 *   Initial value type
 */
template <
  typename InputIteratorT,
  typename OutputIteratorT,
  typename OffsetT,
  typename ReductionOpT,
  typename TransformOpT,
  typename InitValueT,
  typename AccumT =
    ::cuda::std::__accumulator_t<ReductionOpT,
                                 ::cuda::std::invoke_result_t<TransformOpT, ::cuda::std::iter_value_t<InputIteratorT>>,
                                 InitValueT>,
  typename PolicyHub    = detail::reduce::policy_hub<AccumT, OffsetT, ReductionOpT>,
  typename KernelSource = detail::reduce::DeviceReduceKernelSource<
    typename PolicyHub::MaxPolicy,
    InputIteratorT,
    OutputIteratorT,
    OffsetT,
    ReductionOpT,
    InitValueT,
    AccumT,
    TransformOpT>,
  typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
using DispatchTransformReduce =
  DispatchReduce<InputIteratorT,
                 OutputIteratorT,
                 OffsetT,
                 ReductionOpT,
                 InitValueT,
                 AccumT,
                 TransformOpT,
                 PolicyHub,
                 KernelSource,
                 KernelLauncherFactory>;

namespace detail::reduce
{
// Retrieves a device pointer from a pointer-to-pointer.
//
// For CCCL.C's indirect_arg_t: ptr holds the address of the device pointer (&it.state).
// For regular C++ pointers: the caller passes &device_ptr directly.
// In both cases, dereferencing yields the actual device pointer.
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE void* get_device_ptr(void* ptr)
{
  return *reinterpret_cast<void**>(ptr);
}

template <bool StableReductionOrder,
          typename AccumT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ReductionOpT,
          typename InitValueT,
          typename TransformOpT,
          typename KernelSource,
          typename KernelLauncherFactory>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t invoke_regular_size_reduce(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  InputIteratorT d_in,
  OutputIteratorT d_out,
  OffsetT num_items,
  ReductionOpT reduction_op,
  InitValueT init,
  cudaStream_t stream,
  TransformOpT transform_op,
  reduce_policy active_policy,
  KernelSource kernel_source,
  KernelLauncherFactory launcher_factory)
{
  // Get SM count
  int sm_count = 0;
  if (const auto error = CubDebug(launcher_factory.MultiProcessorCount(sm_count)))
  {
    return error;
  }

  // Init regular kernel configuration
  const auto tile_size = active_policy.reduce.threads_per_block * active_policy.reduce.items_per_thread;
  int sm_occupancy     = 0;
  if (const auto error = CubDebug(launcher_factory.MaxSmOccupancy(
        sm_occupancy, kernel_source.ReductionKernel(), active_policy.reduce.threads_per_block)))
  {
    return error;
  }

  const int reduce_device_occupancy = sm_occupancy * sm_count;

  // Even-share work distribution
  const int max_blocks = reduce_device_occupancy * detail::subscription_factor;
  GridEvenShare<OffsetT> even_share;
  even_share.DispatchInit(num_items, max_blocks, tile_size);

  AccumT* d_block_reductions = nullptr; // buffer for per-block aggregates for the two-phase code path
  if constexpr (!StableReductionOrder)
  {
    if (const auto error =
          CubDebug(launcher_factory.MemsetAsync(get_device_ptr(&d_out), 0, kernel_source.InitSize(), stream)))
    {
      return error;
    }
  }
  else
  {
    // Temporary storage allocation requirements
    void* allocations[1]       = {};
    size_t allocation_sizes[1] = {
      max_blocks * kernel_source.AccumSize() // bytes needed for privatized block reductions
    };

    // Alias the temporary allocations from the single storage blob (or
    // compute the necessary size of the blob)
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

    // Alias the allocation for the privatized per-block reductions
    d_block_reductions = static_cast<AccumT*>(allocations[0]);
  }

  // The grid size for DeviceReduceKernel can be zero if the input size is zero. Since the atomic code path does not run
  // a second kernel, we need to handle the empty grid in first kernel already
  int reduce_grid_size = even_share.grid_size;
  if constexpr (!StableReductionOrder)
  {
    reduce_grid_size = ::cuda::std::max(1, reduce_grid_size);
  }

// Log device_reduce_sweep_kernel configuration
#ifdef CUB_DEBUG_LOG
  _CubLog("Invoking DeviceReduceKernel<<<%lu, %d, 0, %lld>>>(), %d items "
          "per thread, %d SM occupancy\n",
          (unsigned long) reduce_grid_size,
          active_policy.reduce.threads_per_block,
          (long long) stream,
          active_policy.reduce.items_per_thread,
          sm_occupancy);
#endif // CUB_DEBUG_LOG

  // Invoke DeviceReduceKernel
  auto reduce_kernel_output = [&] {
    if constexpr (!StableReductionOrder)
    {
      return d_out;
    }
    else
    {
      return d_block_reductions;
    }
  }();
  if (const auto error = CubDebug(
        launcher_factory(reduce_grid_size, active_policy.reduce.threads_per_block, 0, stream)
          .doit(kernel_source.ReductionKernel(),
                d_in,
                reduce_kernel_output,
                num_items,
                even_share,
                reduction_op,
                init,
                transform_op)))
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

  if constexpr (StableReductionOrder)
  {
    // Log single_reduce_sweep_kernel configuration
#ifdef CUB_DEBUG_LOG
    _CubLog("Invoking DeviceReduceSingleTileKernel<<<1, %d, 0, %lld>>>(), "
            "%d items per thread\n",
            active_policy.single_tile.threads_per_block,
            (long long) stream,
            active_policy.single_tile.items_per_thread);
#endif // CUB_DEBUG_LOG

    // Invoke DeviceReduceSingleTileKernel
    if (const auto error = CubDebug(
          launcher_factory(1, active_policy.single_tile.threads_per_block, 0, stream)
            .doit(kernel_source.SingleTileSecondKernel(),
                  d_block_reductions,
                  d_out,
                  reduce_grid_size,
                  reduction_op,
                  init,
                  ::cuda::std::identity{})))
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

// select the accumulator type using an overload set, so __accumulator_t and invoke_result_t are not instantiated when
// an overriding accumulator type is present. This is needed by CCCL.C, which uses void as accumulator type.
template <typename InputIteratorT,
          typename InitValueT,
          typename ReductionOpT,
          typename TransformOpT,
          typename InputT = ::cuda::std::invoke_result_t<TransformOpT, ::cuda::std::iter_value_t<InputIteratorT>>>
_CCCL_HOST_DEVICE_API auto select_accum_t(use_default*) -> ::cuda::std::__accumulator_t<
  ReductionOpT,
  InputT,
  ::cuda::std::conditional_t<::cuda::std::is_same_v<InitValueT, no_init_t>, InputT, InitValueT>>;

template <typename InputIteratorT,
          typename InitValueT,
          typename ReductionOpT,
          typename TransformOpT,
          typename OverrideAccumT,
          ::cuda::std::enable_if_t<!::cuda::std::is_same_v<OverrideAccumT, use_default>, int> = 0>
_CCCL_HOST_DEVICE_API auto select_accum_t(OverrideAccumT*) -> OverrideAccumT;

template <typename OverrideAccumT   = use_default,
          bool StableReductionOrder = true,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ReductionOpT,
          typename InitValueT     = non_void_value_t<OutputIteratorT, it_value_t<InputIteratorT>>,
          typename TransformOpT   = ::cuda::std::identity,
          typename AccumT         = decltype(select_accum_t<InputIteratorT, InitValueT, ReductionOpT, TransformOpT>(
            static_cast<OverrideAccumT*>(nullptr))),
          typename PolicySelector = policy_selector_from_types<AccumT, OffsetT, ReductionOpT, StableReductionOrder>,
          typename KernelSource   = DeviceReduceKernelSource<
              PolicySelector,
              InputIteratorT,
              OutputIteratorT,
              OffsetT,
              ReductionOpT,
              InitValueT,
              AccumT,
              TransformOpT,
              StableReductionOrder>,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
#if _CCCL_HAS_CONCEPTS()
  requires reduce_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE auto dispatch(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  InputIteratorT d_in,
  OutputIteratorT d_out,
  OffsetT num_items,
  ReductionOpT reduction_op,
  InitValueT init,
  cudaStream_t stream,
  TransformOpT transform_op              = {},
  PolicySelector policy_selector         = {},
  KernelSource kernel_source             = {},
  KernelLauncherFactory launcher_factory = {})
{
  ::cuda::compute_capability cc{};
  if (const auto error = CubDebug(launcher_factory.PtxComputeCap(cc)))
  {
    return error;
  }

  return dispatch_compute_cap(policy_selector, cc, [&](auto policy_getter) {
    CUB_DETAIL_CONSTEXPR_ISH const reduce_policy active_policy = policy_getter();

    // known operators for integers are stable, even when using a non-deterministic reduction order
    if constexpr (StableReductionOrder
                  && (!::cuda::std::is_integral_v<AccumT> || !is_cuda_binary_operator<ReductionOpT>) )
    {
      CUB_DETAIL_STATIC_ISH_ASSERT(
        active_policy.reduce.block_algorithm != BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC,
        "A run-to-run deterministic reduction must not use a non-deterministic block_algorithm");
      CUB_DETAIL_STATIC_ISH_ASSERT(
        active_policy.single_tile.block_algorithm != BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC,
        "A run-to-run deterministic reduction must not use a non-deterministic block_algorithm");
    }

#if _CCCL_HOSTED() && defined(CUB_DEBUG_LOG)
    NV_IF_TARGET(NV_IS_HOST, ({
                   std::stringstream ss;
                   ss << active_policy;
                   _CubLog("Dispatching DeviceReduce to compute capability %d.%d with tuning: %s\n",
                           cc.major_cap(),
                           cc.minor_cap(),
                           ss.str().c_str());
                 }))
#endif // _CCCL_HOSTED() && defined(CUB_DEBUG_LOG)

    const bool single_tile_problem =
      num_items
      <= static_cast<OffsetT>(active_policy.single_tile.threads_per_block * active_policy.single_tile.items_per_thread);

    // If we use the atomic code path or have a problem size that fits into a single tile, we don't need temp storage
    if (!StableReductionOrder || single_tile_problem)
    {
      // Return if the caller is simply requesting the size of the storage allocation
      if (d_temp_storage == nullptr)
      {
        temp_storage_bytes = 1;
        return cudaSuccess;
      }
    }

    if constexpr (StableReductionOrder)
    {
      // if the problem is small enough to fit into a single tile, just handle it and return early
      if (single_tile_problem)
      {
#ifdef CUB_DEBUG_LOG
        _CubLog("Invoking DeviceReduceSingleTileKernel<<<1, %d, 0, %lld>>>(), "
                "%d items per thread\n",
                active_policy.single_tile.threads_per_block,
                (long long) stream,
                active_policy.single_tile.items_per_thread);
#endif // CUB_DEBUG_LOG

        // Invoke single_reduce_sweep_kernel
        if (const auto error = CubDebug(
              launcher_factory(1, active_policy.single_tile.threads_per_block, 0, stream)
                .doit(kernel_source.SingleTileKernel(), d_in, d_out, num_items, reduction_op, init, transform_op)))
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

        return cudaSuccess;
      }
    }

    // Regular size
    return invoke_regular_size_reduce<StableReductionOrder, AccumT>(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      num_items,
      reduction_op,
      init,
      stream,
      transform_op,
      active_policy,
      kernel_source,
      launcher_factory);
  });
}
} // namespace detail::reduce

CUB_NAMESPACE_END
