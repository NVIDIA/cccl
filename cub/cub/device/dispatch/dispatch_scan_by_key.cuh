// SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * @file
 * @brief DeviceScan provides device-wide, parallel operations for computing a
 *        prefix scan across a sequence of data items residing within
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

#include <cub/agent/agent_scan_by_key.cuh>
#include <cub/device/dispatch/dispatch_scan.cuh>
#include <cub/device/dispatch/tuning/tuning_scan_by_key.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__host_stdlib/sstream>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_unsigned.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wattributes") // __visibility__ attribute ignored
_CCCL_DIAG_SUPPRESS_NVHPC(attribute_requires_external_linkage)

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

namespace detail::scan_by_key
{
/**
 * @brief Scan by key kernel entry point (multi-block)
 *
 * @tparam PolicySelector
 *   Policy selector type
 *
 * @tparam KeysInputIteratorT
 *   Random-access input iterator type
 *
 * @tparam ValuesInputIteratorT
 *   Random-access input iterator type
 *
 * @tparam ValuesOutputIteratorT
 *   Random-access output iterator type
 *
 * @tparam ScanByKeyTileStateT
 *   Tile status interface type
 *
 * @tparam EqualityOp
 *   Equality functor type
 *
 * @tparam ScanOpT
 *   Scan functor type
 *
 * @tparam InitValueT
 *   The init_value element for ScanOpT type (cub::NullType for inclusive scan)
 *
 * @tparam OffsetT
 *   Unsigned integer type for global offsets
 *
 * @param d_keys_in
 *   Input keys data
 *
 * @param d_keys_prev_in
 *   Predecessor items for each tile
 *
 * @param d_values_in
 *   Input values data
 *
 * @param d_values_out
 *   Output values data
 *
 * @param tile_state
 *   Tile status interface
 *
 * @param start_tile
 *   The starting tile for the current grid
 *
 * @param equality_op
 *   Binary equality functor
 *
 * @param scan_op
 *   Binary scan functor
 *
 * @param init_value
 *   Initial value to seed the exclusive scan
 *
 * @param num_items
 *   Total number of scan items for the entire problem
 */
template <typename PolicySelector,
          typename KeysInputIteratorT,
          typename ValuesInputIteratorT,
          typename ValuesOutputIteratorT,
          typename ScanByKeyTileStateT,
          typename EqualityOp,
          typename ScanOpT,
          typename InitValueT,
          typename OffsetT,
          typename AccumT,
          typename KeyT = cub::detail::it_value_t<KeysInputIteratorT>>
__launch_bounds__(int(current_policy<PolicySelector>().block_threads))
  _CCCL_KERNEL_ATTRIBUTES void DeviceScanByKeyKernel(
    _CCCL_GRID_CONSTANT const KeysInputIteratorT d_keys_in,
    _CCCL_GRID_CONSTANT KeyT* const d_keys_prev_in,
    _CCCL_GRID_CONSTANT const ValuesInputIteratorT d_values_in,
    _CCCL_GRID_CONSTANT const ValuesOutputIteratorT d_values_out,
    ScanByKeyTileStateT tile_state,
    _CCCL_GRID_CONSTANT const int start_tile,
    EqualityOp equality_op,
    _CCCL_GRID_CONSTANT const ScanOpT scan_op,
    _CCCL_GRID_CONSTANT const InitValueT init_value,
    _CCCL_GRID_CONSTANT const OffsetT num_items)
{
  static constexpr scan_by_key_policy policy = current_policy<PolicySelector>();

  using scan_by_key_policy_t = AgentScanByKeyPolicy<
    policy.block_threads,
    policy.items_per_thread,
    policy.load_algorithm,
    policy.load_modifier,
    policy.scan_algorithm,
    policy.store_algorithm,
    delay_constructor_t<policy.delay_constructor.kind,
                        policy.delay_constructor.delay,
                        policy.delay_constructor.l2_write_latency>>;

  // Thread block type for scanning input tiles
  using AgentScanByKeyT = detail::scan_by_key::AgentScanByKey<
    scan_by_key_policy_t,
    KeysInputIteratorT,
    ValuesInputIteratorT,
    ValuesOutputIteratorT,
    EqualityOp,
    ScanOpT,
    InitValueT,
    OffsetT,
    AccumT>;

  // Shared memory for AgentScanByKey
  __shared__ typename AgentScanByKeyT::TempStorage temp_storage;

  // Process tiles
  AgentScanByKeyT(temp_storage, d_keys_in, d_keys_prev_in, d_values_in, d_values_out, equality_op, scan_op, init_value)
    .ConsumeRange(num_items, tile_state, start_tile);
}

template <typename ScanTileStateT, typename KeysInputIteratorT, typename OffsetT>
_CCCL_KERNEL_ATTRIBUTES void DeviceScanByKeyInitKernel(
  ScanTileStateT tile_state,
  _CCCL_GRID_CONSTANT const KeysInputIteratorT d_keys_in,
  cub::detail::it_value_t<KeysInputIteratorT>* d_keys_prev_in,
  _CCCL_GRID_CONSTANT const OffsetT items_per_tile,
  _CCCL_GRID_CONSTANT const int num_tiles)
{
  // Initialize tile status
  tile_state.InitializeStatus(num_tiles);

  const int tid           = blockDim.x * blockIdx.x + threadIdx.x;
  const OffsetT tile_base = static_cast<OffsetT>(tid) * items_per_tile;
  if (tid > 0 && tid < num_tiles)
  {
    d_keys_prev_in[tid] = d_keys_in[tile_base - 1];
  }
}

template <typename PolicySelector,
          typename KeysInputIteratorT,
          typename ValuesInputIteratorT,
          typename ValuesOutputIteratorT,
          typename EqualityOp,
          typename ScanOpT,
          typename InitValueT,
          typename OffsetT,
          typename AccumT>
struct DeviceScanByKeyKernelSource
{
  using ScanByKeyTileStateT = ReduceByKeyScanTileState<AccumT, int>;

  CUB_DEFINE_KERNEL_GETTER(InitKernel, DeviceScanByKeyInitKernel<ScanByKeyTileStateT, KeysInputIteratorT, OffsetT>)

  CUB_DEFINE_KERNEL_GETTER(
    ScanKernel,
    DeviceScanByKeyKernel<PolicySelector,
                          KeysInputIteratorT,
                          ValuesInputIteratorT,
                          ValuesOutputIteratorT,
                          ScanByKeyTileStateT,
                          EqualityOp,
                          ScanOpT,
                          InitValueT,
                          OffsetT,
                          AccumT>)

  CUB_RUNTIME_FUNCTION static ScanByKeyTileStateT TileState()
  {
    return {};
  }
};
} // namespace detail::scan_by_key

/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * @brief Utility class for dispatching the appropriately-tuned kernels
 *        for DeviceScan
 *
 * @tparam KeysInputIteratorT
 *   Random-access input iterator type
 *
 * @tparam ValuesInputIteratorT
 *   Random-access input iterator type
 *
 * @tparam ValuesOutputIteratorT
 *   Random-access output iterator type
 *
 * @tparam EqualityOp
 *   Equality functor type
 *
 * @tparam ScanOpT
 *   Scan functor type
 *
 * @tparam InitValueT
 *   The init_value element for ScanOpT type (cub::NullType for inclusive scan)
 *
 * @tparam OffsetT
 *   Unsigned integer type for global offsets
 *
 */
// TODO(griwes): deprecate when we make the tuning API public and remove in CCCL 4.0
template <
  typename KeysInputIteratorT,
  typename ValuesInputIteratorT,
  typename ValuesOutputIteratorT,
  typename EqualityOp,
  typename ScanOpT,
  typename InitValueT,
  typename OffsetT,
  typename AccumT = ::cuda::std::__accumulator_t<
    ScanOpT,
    cub::detail::it_value_t<ValuesInputIteratorT>,
    ::cuda::std::
      _If<::cuda::std::is_same_v<InitValueT, NullType>, cub::detail::it_value_t<ValuesInputIteratorT>, InitValueT>>,
  typename PolicyHub =
    detail::scan_by_key::policy_hub<KeysInputIteratorT, AccumT, cub::detail::it_value_t<ValuesInputIteratorT>, ScanOpT>,
  typename PolicySelector = detail::scan_by_key::policy_selector_from_hub<PolicyHub>,
  typename KernelSource   = detail::scan_by_key::DeviceScanByKeyKernelSource<
      PolicySelector,
      KeysInputIteratorT,
      ValuesInputIteratorT,
      ValuesOutputIteratorT,
      EqualityOp,
      ScanOpT,
      InitValueT,
      OffsetT,
      AccumT>,
  typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
struct DispatchScanByKey
{
  static_assert(::cuda::std::is_unsigned_v<OffsetT> && sizeof(OffsetT) >= 4,
                "DispatchScan only supports unsigned offset types of at least 4-bytes");

  //---------------------------------------------------------------------
  // Constants and Types
  //---------------------------------------------------------------------

  static constexpr int INIT_KERNEL_THREADS = 128;

  // The input key type
  using KeyT = cub::detail::it_value_t<KeysInputIteratorT>;

  // The input value type
  using InputT = cub::detail::it_value_t<ValuesInputIteratorT>;

  // Tile state used for the decoupled look-back
  using ScanByKeyTileStateT = typename KernelSource::ScanByKeyTileStateT;

  /// Device-accessible allocation of temporary storage. When `nullptr`, the
  /// required allocation size is written to `temp_storage_bytes` and no work
  /// is done.
  void* d_temp_storage;

  /// Reference to size in bytes of `d_temp_storage` allocation
  size_t& temp_storage_bytes;

  /// Iterator to the input sequence of key items
  KeysInputIteratorT d_keys_in;

  /// Iterator to the input sequence of value items
  ValuesInputIteratorT d_values_in;

  /// Iterator to the input sequence of value items
  ValuesOutputIteratorT d_values_out;

  /// Binary equality functor
  EqualityOp equality_op;

  /// Binary scan functor
  ScanOpT scan_op;

  /// Initial value to seed the exclusive scan
  InitValueT init_value;

  /// Total number of input items (i.e., the length of `d_in`)
  OffsetT num_items;

  /// CUDA stream to launch kernels within.
  cudaStream_t stream;
  int ptx_version;
  KernelSource kernel_source;
  KernelLauncherFactory launcher_factory;

  /**
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to `temp_storage_bytes` and no
   *   work is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_keys_in
   *   Iterator to the input sequence of key items
   *
   * @param[in] d_values_in
   *   Iterator to the input sequence of value items
   *
   * @param[out] d_values_out
   *   Iterator to the output sequence of value items
   *
   * @param[in] equality_op
   *   Binary equality functor
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
   *   CUDA stream to launch kernels within.
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
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchScanByKey(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeysInputIteratorT d_keys_in,
    ValuesInputIteratorT d_values_in,
    ValuesOutputIteratorT d_values_out,
    EqualityOp equality_op,
    ScanOpT scan_op,
    InitValueT init_value,
    OffsetT num_items,
    cudaStream_t stream,
    int ptx_version,
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {})
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_keys_in(d_keys_in)
      , d_values_in(d_values_in)
      , d_values_out(d_values_out)
      , equality_op(equality_op)
      , scan_op(scan_op)
      , init_value(init_value)
      , num_items(num_items)
      , stream(stream)
      , ptx_version(ptx_version)
      , kernel_source(kernel_source)
      , launcher_factory(launcher_factory)
  {}

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t __invoke(detail::scan_by_key::scan_by_key_policy active_policy)
  {
    // Get device ordinal
    int device_ordinal;
    if (const auto error = CubDebug(cudaGetDevice(&device_ordinal)))
    {
      return error;
    }

    // Number of input tiles
    const int tile_size = active_policy.block_threads * active_policy.items_per_thread;
    const int num_tiles = static_cast<int>(::cuda::ceil_div(num_items, tile_size));

    auto tile_state = kernel_source.TileState();

    // Specify temporary storage allocation requirements
    size_t allocation_sizes[2];
    if (const auto error = CubDebug(tile_state.AllocationSize(num_tiles, allocation_sizes[0])))
    {
      return error; // bytes needed for tile status descriptors
    }

    allocation_sizes[1] = sizeof(KeyT) * (num_tiles + 1);

    // Compute allocation pointers into the single storage blob (or compute
    // the necessary size of the blob)
    void* allocations[2] = {};
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

    KeyT* d_keys_prev_in = static_cast<KeyT*>(allocations[1]);

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
          launcher_factory(init_grid_size, INIT_KERNEL_THREADS, 0, stream)
            .doit(kernel_source.InitKernel(),
                  tile_state,
                  d_keys_in,
                  d_keys_prev_in,
                  static_cast<OffsetT>(tile_size),
                  num_tiles)))
    {
      return error;
    }

    if (const auto error = CubDebug(cudaPeekAtLastError()))
    {
      return error;
    }

    // Sync the stream if specified to flush runtime errors
    if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
    {
      return error;
    }

    // Get max x-dimension of grid
    int max_dim_x;
    if (const auto error = CubDebug(cudaDeviceGetAttribute(&max_dim_x, cudaDevAttrMaxGridDimX, device_ordinal)))
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
              "per thread\n",
              start_tile,
              scan_grid_size,
              active_policy.block_threads,
              (long long) stream,
              active_policy.items_per_thread);
#endif // CUB_DEBUG_LOG

      // Invoke scan_kernel
      if (const auto error = CubDebug(
            launcher_factory(scan_grid_size, active_policy.block_threads, 0, stream)
              .doit(kernel_source.ScanKernel(),
                    d_keys_in,
                    d_keys_prev_in,
                    d_values_in,
                    d_values_out,
                    tile_state,
                    start_tile,
                    equality_op,
                    scan_op,
                    init_value,
                    num_items)))
      {
        return error;
      }

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
    return __invoke(detail::scan_by_key::convert_policy<ActivePolicyT>());
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
   * @param[in] d_keys_in
   *   Iterator to the input sequence of key items
   *
   * @param[in] d_values_in
   *   Iterator to the input sequence of value items
   *
   * @param[out] d_values_out
   *   Iterator to the output sequence of value items
   *
   * @param[in] equality_op
   *   Binary equality functor
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
   *   CUDA stream to launch kernels within.
   */
  // TODO(griwes): deprecate when we make the tuning API public and remove in CCCL 4.0
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeysInputIteratorT d_keys_in,
    ValuesInputIteratorT d_values_in,
    ValuesOutputIteratorT d_values_out,
    EqualityOp equality_op,
    ScanOpT scan_op,
    InitValueT init_value,
    OffsetT num_items,
    cudaStream_t stream,
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {})
  {
    int ptx_version = 0;
    if (const auto error = CubDebug(launcher_factory.PtxVersion(ptx_version)))
    {
      return error;
    }

    DispatchScanByKey dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_values_in,
      d_values_out,
      equality_op,
      scan_op,
      init_value,
      num_items,
      stream,
      ptx_version,
      kernel_source,
      launcher_factory);

    return CubDebug(typename PolicyHub::MaxPolicy{}.Invoke(ptx_version, dispatch));
  }

  template <typename PolicySelectorT, typename KernelSourceT = KernelSource>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeysInputIteratorT d_keys_in,
    ValuesInputIteratorT d_values_in,
    ValuesOutputIteratorT d_values_out,
    EqualityOp equality_op,
    ScanOpT scan_op,
    InitValueT init_value,
    OffsetT num_items,
    cudaStream_t stream,
    PolicySelectorT policy_selector,
    KernelSourceT kernel_source            = {},
    KernelLauncherFactory launcher_factory = {})
  {
    ::cuda::arch_id arch_id{};
    if (const auto error = CubDebug(launcher_factory.PtxArchId(arch_id)))
    {
      return error;
    }

#if !_CCCL_COMPILER(NVRTC) && defined(CUB_DEBUG_LOG)
    NV_IF_TARGET(NV_IS_HOST, ({
                   ::std::stringstream ss;
                   ss << policy_selector(arch_id);
                   _CubLog("Dispatching DeviceScanByKey to arch %d with tuning: %s\n",
                           static_cast<int>(arch_id),
                           ss.str().c_str());
                 }))
#endif

    const detail::scan_by_key::scan_by_key_policy active_policy = policy_selector(arch_id);

    return DispatchScanByKey<KeysInputIteratorT,
                             ValuesInputIteratorT,
                             ValuesOutputIteratorT,
                             EqualityOp,
                             ScanOpT,
                             InitValueT,
                             OffsetT,
                             AccumT,
                             PolicyHub,
                             PolicySelectorT,
                             KernelSourceT,
                             KernelLauncherFactory>(
             d_temp_storage,
             temp_storage_bytes,
             d_keys_in,
             d_values_in,
             d_values_out,
             equality_op,
             scan_op,
             init_value,
             num_items,
             stream,
             -1,
             kernel_source,
             launcher_factory)
      .__invoke(active_policy);
  }
};

namespace detail::scan_by_key
{
template <
  typename KeysInputIteratorT,
  typename ValuesInputIteratorT,
  typename ValuesOutputIteratorT,
  typename EqualityOp,
  typename ScanOpT,
  typename InitValueT,
  typename OffsetT,
  typename AccumT = ::cuda::std::__accumulator_t<
    ScanOpT,
    cub::detail::it_value_t<ValuesInputIteratorT>,
    ::cuda::std::
      _If<::cuda::std::is_same_v<InitValueT, NullType>, cub::detail::it_value_t<ValuesInputIteratorT>, InitValueT>>,
  typename PolicySelector = policy_selector_from_types<cub::detail::it_value_t<KeysInputIteratorT>,
                                                       AccumT,
                                                       cub::detail::it_value_t<ValuesInputIteratorT>,
                                                       ScanOpT>,
  typename KernelSource   = DeviceScanByKeyKernelSource<
      PolicySelector,
      KeysInputIteratorT,
      ValuesInputIteratorT,
      ValuesOutputIteratorT,
      EqualityOp,
      ScanOpT,
      InitValueT,
      OffsetT,
      AccumT>,
  typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE auto dispatch(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  KeysInputIteratorT d_keys_in,
  ValuesInputIteratorT d_values_in,
  ValuesOutputIteratorT d_values_out,
  EqualityOp equality_op,
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
  NV_IF_TARGET(
    NV_IS_HOST, ({
      ::std::stringstream ss;
      ss << policy_selector(arch_id);
      _CubLog("Dispatching DeviceScanByKey to arch %d with tuning: %s\n", static_cast<int>(arch_id), ss.str().c_str());
    }))
#endif // !_CCCL_COMPILER(NVRTC) && defined(CUB_DEBUG_LOG)

  struct fake_policy
  {
    using MaxPolicy = void;
  };

  const detail::scan_by_key::scan_by_key_policy active_policy = policy_selector(arch_id);

  return DispatchScanByKey<KeysInputIteratorT,
                           ValuesInputIteratorT,
                           ValuesOutputIteratorT,
                           EqualityOp,
                           ScanOpT,
                           InitValueT,
                           OffsetT,
                           AccumT,
                           fake_policy,
                           PolicySelector,
                           KernelSource,
                           KernelLauncherFactory>{
    d_temp_storage,
    temp_storage_bytes,
    d_keys_in,
    d_values_in,
    d_values_out,
    equality_op,
    scan_op,
    init_value,
    num_items,
    stream,
    -1,
    kernel_source,
    launcher_factory}
    .__invoke(active_policy);
}
} // namespace detail::scan_by_key

CUB_NAMESPACE_END

_CCCL_DIAG_POP
