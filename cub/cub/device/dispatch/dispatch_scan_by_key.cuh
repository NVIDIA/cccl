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
#include <cub/detail/cc_dispatch.cuh>
#include <cub/detail/logging.cuh>
#include <cub/device/dispatch/dispatch_scan.cuh>
#include <cub/device/dispatch/tuning/tuning_scan_by_key.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_vsmem.cuh>

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
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto
with_threads_and_items(ScanByKeyPolicy policy, int threads_per_block, int items_per_thread) -> ScanByKeyPolicy
{
  policy.lookback.threads_per_block = threads_per_block;
  policy.lookback.items_per_thread  = items_per_thread;
  return policy;
}

template <typename PolicyGetter, typename... AgentParamsT>
struct scan_by_key_vsmem_helper
{
  static constexpr ScanByKeyPolicy selected_policy = PolicyGetter{}();

  using type = vsmem_helper_default_fallback_policy_t<
    agent_scan_by_key_policy<selected_policy.lookback.threads_per_block,
                             selected_policy.lookback.items_per_thread,
                             selected_policy.lookback.load_algorithm,
                             selected_policy.lookback.load_modifier,
                             selected_policy.lookback.scan_algorithm,
                             selected_policy.lookback.store_algorithm,
                             delay_constructor_t<selected_policy.lookback.lookback_delay.kind,
                                                 selected_policy.lookback.lookback_delay.delay,
                                                 selected_policy.lookback.lookback_delay.l2_write_latency>>,
    AgentScanByKey,
    AgentParamsT...>;

  static constexpr ScanByKeyPolicy policy = with_threads_and_items(
    selected_policy, type::agent_policy_t::BLOCK_THREADS, type::agent_policy_t::ITEMS_PER_THREAD);
  static constexpr ::cuda::std::size_t vsmem_per_block = type::vsmem_per_block;
};

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
 *
 * @param vsmem
 *   Memory to support virtual shared memory
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
__launch_bounds__(
  scan_by_key_vsmem_helper<device_policy_getter<PolicySelector, current_tuning_cc().get()>,
                           KeysInputIteratorT,
                           ValuesInputIteratorT,
                           ValuesOutputIteratorT,
                           EqualityOp,
                           ScanOpT,
                           InitValueT,
                           OffsetT,
                           AccumT>::policy.lookback.threads_per_block)
  _CCCL_KERNEL_ATTRIBUTES void DeviceScanByKeyKernel(
    const KeysInputIteratorT d_keys_in,
    KeyT* const d_keys_prev_in,
    const ValuesInputIteratorT d_values_in,
    const ValuesOutputIteratorT d_values_out,
    ScanByKeyTileStateT tile_state,
    const int start_tile,
    EqualityOp equality_op,
    const ScanOpT scan_op,
    const InitValueT init_value,
    const OffsetT num_items,
    vsmem_t vsmem)
{
  using vsmem_helper_t = typename scan_by_key_vsmem_helper<
    device_policy_getter<PolicySelector, current_tuning_cc().get()>,
    KeysInputIteratorT,
    ValuesInputIteratorT,
    ValuesOutputIteratorT,
    EqualityOp,
    ScanOpT,
    InitValueT,
    OffsetT,
    AccumT>::type;

  // Thread block type for scanning input tiles
  using agent_scan_by_key_t = typename vsmem_helper_t::agent_t;

  __shared__ typename vsmem_helper_t::static_temp_storage_t static_temp_storage;
  typename agent_scan_by_key_t::TempStorage& temp_storage =
    vsmem_helper_t::get_temp_storage(static_temp_storage, vsmem);

  // Process tiles
  agent_scan_by_key_t(
    temp_storage, d_keys_in, d_keys_prev_in, d_values_in, d_values_out, equality_op, scan_op, init_value)
    .ConsumeRange(num_items, tile_state, start_tile);

  vsmem_helper_t::discard_temp_storage(temp_storage);
}

template <typename ScanTileStateT, typename KeysInputIteratorT, typename OffsetT>
_CCCL_KERNEL_ATTRIBUTES void DeviceScanByKeyInitKernel(
  ScanTileStateT tile_state,
  const KeysInputIteratorT d_keys_in,
  cub::detail::it_value_t<KeysInputIteratorT>* d_keys_prev_in,
  const OffsetT items_per_tile,
  const int num_tiles)
{
  // Initialize tile status
  tile_state.InitializeStatus(num_tiles);

  const int tid           = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
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
  typename PolicyHub = policy_hub<KeysInputIteratorT, AccumT, cub::detail::it_value_t<ValuesInputIteratorT>, ScanOpT>,
  typename PolicySelector = policy_selector_from_hub<PolicyHub>,
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
struct dispatch_scan_by_key
{
  static_assert(::cuda::std::is_unsigned_v<OffsetT> && sizeof(OffsetT) >= 4,
                "dispatch_scan_by_key only supports unsigned offset types of at least 4-bytes");

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
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE dispatch_scan_by_key(
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

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
  __invoke(ScanByKeyPolicy active_policy, ::cuda::std::size_t vsmem_per_block)
  {
    // Get device ordinal
    int device_ordinal;
    if (const auto error = CubDebug(cudaGetDevice(&device_ordinal)))
    {
      return error;
    }

    // Number of input tiles
    const int tile_size = active_policy.lookback.threads_per_block * active_policy.lookback.items_per_thread;
    const int num_tiles = static_cast<int>(::cuda::ceil_div(num_items, tile_size));

    auto tile_state = kernel_source.TileState();

    // Specify temporary storage allocation requirements
    size_t allocation_sizes[3];
    if (const auto error = CubDebug(tile_state.AllocationSize(num_tiles, allocation_sizes[0])))
    {
      return error; // bytes needed for tile status descriptors
    }

    allocation_sizes[1] = sizeof(KeyT) * (num_tiles + 1);
    allocation_sizes[2] = num_tiles * vsmem_per_block;

    // Compute allocation pointers into the single storage blob (or compute
    // the necessary size of the blob)
    void* allocations[3] = {};
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

    KeyT* d_keys_prev_in = static_cast<KeyT*>(allocations[1]); // NOLINT(misc-const-correctness)

    // Construct the tile status interface
    if (const auto error = CubDebug(tile_state.Init(num_tiles, allocations[0], allocation_sizes[0])))
    {
      return error;
    }

    // Log init_kernel configuration
    const int init_grid_size = ::cuda::ceil_div(num_tiles, INIT_KERNEL_THREADS);
    _CUB_LOG_KERNEL_LAUNCH("init_kernel", init_grid_size, 1, 1, INIT_KERNEL_THREADS, 0, stream, "");

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
      _CUB_LOG_KERNEL_LAUNCH(
        "scan_kernel",
        scan_grid_size,
        1,
        1,
        active_policy.lookback.threads_per_block,
        0,
        stream,
        ", epoch: %d",
        start_tile);

      // Invoke scan_kernel
      if (const auto error = CubDebug(
            launcher_factory(scan_grid_size, active_policy.lookback.threads_per_block, 0, stream)
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
                    num_items,
                    vsmem_t{allocations[2]})))
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

  template <typename PolicyGetter>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t __invoke(PolicyGetter)
  {
    using vsmem_helper_t = scan_by_key_vsmem_helper<
      PolicyGetter,
      KeysInputIteratorT,
      ValuesInputIteratorT,
      ValuesOutputIteratorT,
      EqualityOp,
      ScanOpT,
      InitValueT,
      OffsetT,
      AccumT>;
    constexpr ScanByKeyPolicy active_policy       = vsmem_helper_t::policy;
    constexpr ::cuda::std::size_t vsmem_per_block = vsmem_helper_t::vsmem_per_block;
    return __invoke(active_policy, vsmem_per_block);
  }

  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_HOST _CCCL_FORCEINLINE cudaError_t Invoke(ActivePolicyT = {})
  {
    struct policy_getter
    {
      _CCCL_HOST_DEVICE constexpr auto operator()() const -> ScanByKeyPolicy
      {
        return detail::scan_by_key::convert_policy<ActivePolicyT>();
      }
    };
    return __invoke(policy_getter{});
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

    dispatch_scan_by_key dispatch(
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
    ::cuda::compute_capability cc{};
    if (const auto error = CubDebug(launcher_factory.PtxComputeCap(cc)))
    {
      return error;
    }

    return detail::dispatch_compute_cap(policy_selector, cc, [&](auto policy_getter) {
      detail::log_dispatch("DeviceScanByKey", cc, policy_getter());

      return dispatch_scan_by_key<
               KeysInputIteratorT,
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
        .__invoke(policy_getter);
    });
  }
};

template <
  typename OverrideAccumT = use_default,
  typename KeysInputIteratorT,
  typename ValuesInputIteratorT,
  typename ValuesOutputIteratorT,
  typename EqualityOp,
  typename ScanOpT,
  typename InitValueT,
  typename OffsetT,
  typename AccumT = ::cuda::std::conditional_t<
    !::cuda::std::is_same_v<OverrideAccumT, use_default>,
    OverrideAccumT,
    ::cuda::std::__accumulator_t<
      ScanOpT,
      cub::detail::it_value_t<ValuesInputIteratorT>,
      ::cuda::std::
        _If<::cuda::std::is_same_v<InitValueT, NullType>, cub::detail::it_value_t<ValuesInputIteratorT>, InitValueT>>>,
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
#if _CCCL_HAS_CONCEPTS()
  requires scan_by_key_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
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
                "scan_by_key::dispatch only supports unsigned offset types of at least 4-bytes");

  using KeyT = cub::detail::it_value_t<KeysInputIteratorT>;

  static constexpr int INIT_KERNEL_THREADS = 128;

  ::cuda::compute_capability cc{};
  if (const auto error = CubDebug(launcher_factory.PtxComputeCap(cc)))
  {
    return error;
  }

  ScanByKeyPolicy active_policy{};
  ::cuda::std::size_t vsmem_per_block = 0;
  if (const auto error = CubDebug(detail::dispatch_compute_cap(policy_selector, cc, [&](auto policy_getter) {
        detail::log_dispatch("DeviceScanByKey", cc, policy_getter());

        using vsmem_helper_t = scan_by_key_vsmem_helper<
          decltype(policy_getter),
          KeysInputIteratorT,
          ValuesInputIteratorT,
          ValuesOutputIteratorT,
          EqualityOp,
          ScanOpT,
          InitValueT,
          OffsetT,
          AccumT>;
        constexpr ScanByKeyPolicy policy = vsmem_helper_t::policy;
        active_policy                    = policy;
        vsmem_per_block                  = vsmem_helper_t::vsmem_per_block;
        return cudaSuccess;
      })))
  {
    return error;
  }

  // Get device ordinal
  int device_ordinal;
  if (const auto error = CubDebug(cudaGetDevice(&device_ordinal)))
  {
    return error;
  }

  // Number of input tiles
  const int tile_size = active_policy.lookback.threads_per_block * active_policy.lookback.items_per_thread;
  const int num_tiles = static_cast<int>(::cuda::ceil_div(num_items, tile_size));

  auto tile_state = kernel_source.TileState();

  // Specify temporary storage allocation requirements
  size_t allocation_sizes[3];
  if (const auto error = CubDebug(tile_state.AllocationSize(num_tiles, allocation_sizes[0])))
  {
    return error; // bytes needed for tile status descriptors
  }

  allocation_sizes[1] = sizeof(KeyT) * (num_tiles + 1);
  allocation_sizes[2] = num_tiles * vsmem_per_block;

  // Compute allocation pointers into the single storage blob (or compute
  // the necessary size of the blob)
  void* allocations[3] = {};
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

  KeyT* d_keys_prev_in = static_cast<KeyT*>(allocations[1]); // NOLINT(misc-const-correctness)

  // Construct the tile status interface
  if (const auto error = CubDebug(tile_state.Init(num_tiles, allocations[0], allocation_sizes[0])))
  {
    return error;
  }

  // Log init_kernel configuration
  const int init_grid_size = ::cuda::ceil_div(num_tiles, INIT_KERNEL_THREADS);
  _CUB_LOG_KERNEL_LAUNCH("init_kernel", init_grid_size, 1, 1, INIT_KERNEL_THREADS, 0, stream, "");

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
    _CUB_LOG_KERNEL_LAUNCH(
      "scan_kernel",
      scan_grid_size,
      1,
      1,
      active_policy.lookback.threads_per_block,
      0,
      stream,
      ", epoch: %d",
      start_tile);

    // Invoke scan_kernel
    if (const auto error = CubDebug(
          launcher_factory(scan_grid_size, active_policy.lookback.threads_per_block, 0, stream)
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
                  num_items,
                  vsmem_t{allocations[2]})))
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
} // namespace detail::scan_by_key

// TODO(griwes): remove in CCCL 4.0
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
using DispatchScanByKey
  CCCL_DEPRECATED_BECAUSE("Use the tuning API for DeviceScan") = detail::scan_by_key::dispatch_scan_by_key<
    KeysInputIteratorT,
    ValuesInputIteratorT,
    ValuesOutputIteratorT,
    EqualityOp,
    ScanOpT,
    InitValueT,
    OffsetT,
    AccumT,
    PolicyHub,
    PolicySelector,
    KernelSource,
    KernelLauncherFactory>;

CUB_NAMESPACE_END

_CCCL_DIAG_POP
