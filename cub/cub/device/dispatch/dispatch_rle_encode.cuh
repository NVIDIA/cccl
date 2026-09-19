// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
#include <cub/detail/logging.cuh>
#include <cub/device/dispatch/dispatch_streaming_reduce_by_key.cuh>
#include <cub/device/dispatch/kernels/kernel_rle_encode_lookahead.cuh>
#include <cub/device/dispatch/tuning/tuning_rle_encode.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_temporary_storage.cuh>

#include <thrust/type_traits/is_contiguous_iterator.h>
#include <thrust/type_traits/unwrap_contiguous_iterator.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__device/compute_capability.h>
#include <cuda/__memory/align_up.h>
#include <cuda/__type_traits/is_trivially_copyable.h>
#include <cuda/iterator>
#include <cuda/std/__host_stdlib/sstream>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/limits>

CUB_NAMESPACE_BEGIN

namespace detail::rle::encode
{
// DeviceRunLengthEncode::Encode's lookback path dispatches to ReduceByKey, which has its own tuning policy,
// so the policy selector is adapted to convert the tuning policy
template <typename PolicySelector>
#if _CCCL_HAS_CONCEPTS()
  requires rle_encode_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
struct policy_selector_adapter
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const
    -> ReduceByKeyPolicy
  {
    const RleEncodePolicy policy = PolicySelector{}(cc);
    return ReduceByKeyPolicy{
      ReduceByKeyAlgorithm::lookback,
      {policy.lookback.threads_per_block,
       policy.lookback.items_per_thread,
       policy.lookback.load_algorithm,
       policy.lookback.load_modifier,
       policy.lookback.scan_algorithm,
       policy.lookback.lookback_delay}};
  }
};

template <typename PolicySelector,
          typename KeysInputIteratorT,
          typename UniqueOutputIteratorT,
          typename ValuesInputIteratorT,
          typename AggregatesOutputIteratorT,
          typename NumRunsOutputIteratorT,
          typename ScanTileStateT,
          typename EqualityOpT,
          typename ReductionOpT,
          typename OffsetT,
          typename AccumT,
          typename StreamingContextT>
#if _CCCL_HAS_CONCEPTS()
  requires rle_encode_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(int(current_policy<PolicySelector>().lookback.threads_per_block))
  _CCCL_KERNEL_ATTRIBUTES void DeviceRleEncodeStreamingKernel(
    const KeysInputIteratorT d_keys_in,
    const UniqueOutputIteratorT d_unique_out,
    const ValuesInputIteratorT d_values_in,
    const AggregatesOutputIteratorT d_aggregates_out,
    const NumRunsOutputIteratorT d_num_runs_out,
    ScanTileStateT tile_state,
    const int start_tile,
    EqualityOpT equality_op,
    ReductionOpT reduction_op,
    const OffsetT num_items,
    const StreamingContextT streaming_context,
    vsmem_t vsmem)
{
  static constexpr RleEncodePolicy policy = current_policy<PolicySelector>();
  // this kernel is launched only from the call path whose lookahead branch is compile-time viable, so on
  // architectures whose policy selects lookahead it can never run and compiles to an empty stub
  if constexpr (policy.algorithm != RleAlgorithm::lookback)
  {
    return;
  }
  else
  {
    using AgentReduceByKeyPolicyT = agent_reduce_by_key_policy<
      policy.lookback.threads_per_block,
      policy.lookback.items_per_thread,
      policy.lookback.load_algorithm,
      policy.lookback.load_modifier,
      policy.lookback.scan_algorithm,
      delay_constructor_t<policy.lookback.lookback_delay.kind,
                          policy.lookback.lookback_delay.delay,
                          policy.lookback.lookback_delay.l2_write_latency>>;

    using vsmem_helper_t = vsmem_helper_default_fallback_policy_t<
      AgentReduceByKeyPolicyT,
      reduce_by_key::AgentReduceByKey,
      KeysInputIteratorT,
      UniqueOutputIteratorT,
      ValuesInputIteratorT,
      AggregatesOutputIteratorT,
      NumRunsOutputIteratorT,
      EqualityOpT,
      ReductionOpT,
      OffsetT,
      AccumT,
      StreamingContextT>;

    using agent_reduce_by_key_t = typename vsmem_helper_t::agent_t;

    __shared__ typename vsmem_helper_t::static_temp_storage_t static_temp_storage;

    typename agent_reduce_by_key_t::TempStorage& temp_storage =
      vsmem_helper_t::get_temp_storage(static_temp_storage, vsmem);

    agent_reduce_by_key_t(
      temp_storage,
      d_keys_in,
      d_unique_out,
      d_values_in,
      d_aggregates_out,
      d_num_runs_out,
      equality_op,
      reduction_op,
      streaming_context)
      .ConsumeRange(num_items, tile_state, start_tile);

    vsmem_helper_t::discard_temp_storage(temp_storage);
  }
}

template <typename PolicySelector>
struct streaming_kernel_source
{
  template <typename, typename... KernelArgTs>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static constexpr auto reduce_by_key_kernel()
  {
    return &DeviceRleEncodeStreamingKernel<PolicySelector, KernelArgTs...>;
  }
};

template <class PolicySelector,
          class KernelSource = reduce_by_key::reduce_by_key_kernel_source,
          class InputIteratorT,
          class UniqueOutputIteratorT,
          class LengthsOutputIteratorT,
          class NumRunsOutputIteratorT,
          class OffsetT>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t invoke_streaming(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  InputIteratorT d_in,
  UniqueOutputIteratorT d_unique_out,
  LengthsOutputIteratorT d_counts_out,
  NumRunsOutputIteratorT d_num_runs_out,
  OffsetT num_items,
  cudaStream_t stream)
{
  using length_t                 = cub::detail::non_void_value_t<LengthsOutputIteratorT, OffsetT>;
  using lengths_input_iterator_t = ::cuda::constant_iterator<length_t, OffsetT>;
  return detail::reduce_by_key::dispatch_streaming(
    d_temp_storage,
    temp_storage_bytes,
    d_in,
    d_unique_out,
    lengths_input_iterator_t(length_t{1}),
    d_counts_out,
    d_num_runs_out,
    ::cuda::std::equal_to<>{},
    ::cuda::std::plus<>{},
    num_items,
    stream,
    policy_selector_adapter<PolicySelector>{},
    KernelSource{});
}

template <typename PolicySelector, typename KeyT, typename LengthT, typename NumRunsT, typename OffsetT>
struct DeviceRleEncodeKernelSource
{
#if __cccl_ptx_isa >= 920
  CUB_DEFINE_KERNEL_GETTER(InitKernel, DeviceRleEncodeLookaheadInitKernel<PolicySelector, tile_partial_state_t>)

  CUB_DEFINE_KERNEL_GETTER(LookaheadKernel,
                           DeviceRleEncodeLookaheadKernel<PolicySelector, KeyT, LengthT, NumRunsT, OffsetT>)
#endif // __cccl_ptx_isa >= 920
};

// the lookahead kernel only exists from PTX ISA 9.2 (CUDA 13.2); below that, dispatch is streaming-only
#if __cccl_ptx_isa >= 920
// compile-time half of the lookahead viability
template <class InputIteratorT,
          class UniqueOutputIteratorT,
          class LengthsOutputIteratorT,
          class NumRunsOutputIteratorT,
          class OffsetT>
inline constexpr bool lookahead_instantiable =
  THRUST_NS_QUALIFIER::is_contiguous_iterator_v<InputIteratorT>
  && THRUST_NS_QUALIFIER::is_contiguous_iterator_v<UniqueOutputIteratorT>
  && THRUST_NS_QUALIFIER::is_contiguous_iterator_v<LengthsOutputIteratorT>
  && THRUST_NS_QUALIFIER::is_contiguous_iterator_v<NumRunsOutputIteratorT>
  && ::cuda::is_trivially_copyable_v<it_value_t<InputIteratorT>>
  && ::cuda::std::is_same_v<it_value_t<InputIteratorT>, it_value_t<UniqueOutputIteratorT>>
  && (16 % sizeof(it_value_t<InputIteratorT>) == 0)
  && (alignof(it_value_t<InputIteratorT>) == sizeof(it_value_t<InputIteratorT>))
  && (::cuda::std::is_same_v<OffsetT, ::cuda::std::int32_t> || ::cuda::std::is_same_v<OffsetT, ::cuda::std::int64_t>);

// Launches the lookahead init + main kernels. Callable from host and device: the host arm queries the
// device's opt-in shared memory and picks the tuned staged configuration when it fits, else the unstaged
// floor; device-side (CDP) callers cannot raise the dynamic shared memory limit, so they always launch
// the floor, which fits the default limit on every device. Kernels come from the kernel_source so this
// helper stays independent of the kernel instantiation (same shape as scan).
template <class KernelSource,
          class InputIteratorT,
          class UniqueOutputIteratorT,
          class LengthsOutputIteratorT,
          class NumRunsOutputIteratorT,
          class OffsetT,
          class LauncherFactory>
CUB_RUNTIME_FUNCTION cudaError_t invoke_lookahead(
  KernelSource kernel_source,
  const RleLookaheadPolicy& lookahead_policy,
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  InputIteratorT d_in,
  UniqueOutputIteratorT d_unique_out,
  LengthsOutputIteratorT d_counts_out,
  NumRunsOutputIteratorT d_num_runs_out,
  OffsetT num_items,
  cudaStream_t stream,
  LauncherFactory launcher_factory)
{
  using key_t      = it_value_t<InputIteratorT>;
  using num_runs_t = it_value_t<NumRunsOutputIteratorT>;

  if (num_items <= 0)
  {
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1; // just fulfill the contract that CUB always requires some temporary storage
      return cudaSuccess;
    }
    return CubDebug(cudaMemsetAsync(
      THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(d_num_runs_out), 0, sizeof(num_runs_t), stream));
  }

  const OffsetT num_tiles_wide = ::cuda::ceil_div(num_items, static_cast<OffsetT>(lookahead_policy.tile_size()));
  if constexpr (sizeof(OffsetT) > sizeof(int))
  {
    // one CTA per tile: the x-grid limit caps the supported input size (>= 16 TiB of keys at the smallest tile)
    if (num_tiles_wide > static_cast<OffsetT>(::cuda::std::numeric_limits<int>::max()))
    {
      return CubDebug(cudaErrorInvalidValue);
    }
  }
  const int num_tiles = static_cast<int>(num_tiles_wide);

  void* allocations[1]       = {};
  size_t allocation_sizes[1] = {static_cast<size_t>(num_tiles) * sizeof(tile_partial_state_t)};
  if (const auto error =
        CubDebug(detail::alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes)))
  {
    return error;
  }
  if (d_temp_storage == nullptr)
  {
    return cudaSuccess;
  }
  auto* tile_partial_states = static_cast<tile_partial_state_t*>(allocations[0]);

  int key_ring_stages   = lookahead_policy.key_ring_stages;
  int pos_ring_stages   = lookahead_policy.pos_ring_stages;
  bool keys_staged      = true;
  size_t dyn_smem_bytes = lookahead_policy.dyn_smem_bytes(int{sizeof(key_t)}, int{alignof(key_t)});
  NV_IF_TARGET(NV_IS_HOST,
               ({
                 int device         = 0;
                 int max_optin_smem = 0;
                 if (const auto error = CubDebug(cudaGetDevice(&device)))
                 {
                   return error;
                 }
                 if (const auto error = CubDebug(
                       cudaDeviceGetAttribute(&max_optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device)))
                 {
                   return error;
                 }
                 if (dyn_smem_bytes + RleLookaheadPolicy::static_smem_budget <= static_cast<size_t>(max_optin_smem))
                 {
                   if (const auto error = CubDebug(launcher_factory.set_max_dynamic_smem_size_for(
                         kernel_source.LookaheadKernel(), static_cast<int>(dyn_smem_bytes))))
                   {
                     return error;
                   }
                 }
                 else
                 {
                   keys_staged = false;
                 }
               }),
               ({ keys_staged = false; }))
  if (!keys_staged)
  {
    // vvv regressed case: CDP callers always land here. Fits under 48KB SMEM vvv
    key_ring_stages = lookahead_policy.floor_key_ring_stages();
    pos_ring_stages = lookahead_policy.floor_pos_ring_stages();
    dyn_smem_bytes  = lookahead_policy.floor_dyn_smem_bytes();
    // ^^^ regressed case ^^^
  }

  {
    constexpr int init_kernel_threads = 128;
    const auto init_grid_size         = ::cuda::ceil_div(num_tiles, init_kernel_threads);
#  ifdef CUB_DEBUG_LOG
    _CubLog("Invoking DeviceRleEncodeLookaheadInitKernel<<<%d, %d, 0, %lld>>>()\n",
            init_grid_size,
            init_kernel_threads,
            (long long) stream);
#  else // CUB_DEBUG_LOG
    log("Invoking DeviceRleEncodeLookaheadInitKernel<<<%d, %d, 0, %lld>>>()\n",
        init_grid_size,
        init_kernel_threads,
        (long long) stream);
#  endif // CUB_DEBUG_LOG
    if (const auto error = CubDebug(
          launcher_factory(init_grid_size, init_kernel_threads, 0, stream, /* dependent_launch */ false)
            .doit(kernel_source.InitKernel(), tile_partial_states, static_cast<::cuda::std::int64_t>(num_tiles))))
    {
      return error;
    }
    if (const auto error = CubDebug(cudaPeekAtLastError()))
    {
      return error;
    }
    if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
    {
      return error;
    }
  }
  {
    const int block_dim = num_total_threads(lookahead_policy);
#  ifdef CUB_DEBUG_LOG
    _CubLog("Invoking DeviceRleEncodeLookaheadKernel<<<%d, %d, %zu, %lld>>>()\n",
            num_tiles,
            block_dim,
            dyn_smem_bytes,
            (long long) stream);
#  else // CUB_DEBUG_LOG
    log("Invoking DeviceRleEncodeLookaheadKernel<<<%d, %d, %zu, %lld>>>()\n",
        num_tiles,
        block_dim,
        dyn_smem_bytes,
        (long long) stream);
#  endif // CUB_DEBUG_LOG
    if (const auto error = CubDebug(
          launcher_factory(num_tiles,
                           block_dim,
                           static_cast<int>(dyn_smem_bytes),
                           stream,
                           /* dependent_launch */ false)
            .doit(kernel_source.LookaheadKernel(),
                  THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(d_in),
                  THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(d_unique_out),
                  THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(d_counts_out),
                  THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(d_num_runs_out),
                  tile_partial_states,
                  num_items,
                  num_tiles,
                  key_ring_stages,
                  pos_ring_stages,
                  keys_staged)))
    {
      return error;
    }
    if (const auto error = CubDebug(cudaPeekAtLastError()))
    {
      return error;
    }
    if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
    {
      return error;
    }
  }
  return cudaSuccess;
}
#endif // __cccl_ptx_isa >= 920

// Dispatches DeviceRunLengthEncode::Encode: the lookahead implementation when the tuning policy selects
// it (host-side callers on viable types), the streaming reduce-by-key implementation otherwise (lookback
// policies, non-viable types, and device-side callers).
template <class PolicySelector,
          class InputIteratorT,
          class UniqueOutputIteratorT,
          class LengthsOutputIteratorT,
          class NumRunsOutputIteratorT,
          class OffsetT,
          class KernelSource    = DeviceRleEncodeKernelSource<PolicySelector,
                                                              it_value_t<InputIteratorT>,
                                                              it_value_t<LengthsOutputIteratorT>,
                                                              it_value_t<NumRunsOutputIteratorT>,
                                                              OffsetT>,
          class LauncherFactory = detail::TripleChevronFactory>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  InputIteratorT d_in,
  UniqueOutputIteratorT d_unique_out,
  LengthsOutputIteratorT d_counts_out,
  NumRunsOutputIteratorT d_num_runs_out,
  OffsetT num_items,
  cudaStream_t stream,
  [[maybe_unused]] PolicySelector policy_selector   = {},
  [[maybe_unused]] KernelSource kernel_source       = {},
  [[maybe_unused]] LauncherFactory launcher_factory = {})
{
#if __cccl_ptx_isa >= 920
  if constexpr (lookahead_instantiable<InputIteratorT,
                                       UniqueOutputIteratorT,
                                       LengthsOutputIteratorT,
                                       NumRunsOutputIteratorT,
                                       OffsetT>)
  {
    ::cuda::compute_capability cc{};
    if (const auto error = CubDebug(launcher_factory.PtxComputeCap(cc)))
    {
      return error;
    }
    return detail::dispatch_compute_cap(policy_selector, cc, [&](auto policy_getter) -> cudaError_t {
#  if _CCCL_HOSTED() && defined(CUB_DEBUG_LOG)
      NV_IF_TARGET(NV_IS_HOST, ({
                     ::std::stringstream ss;
                     ss << policy_getter();
                     _CubLog("Dispatching DeviceRunLengthEncode::Encode to compute capability %d.%d with tuning: %s\n",
                             cc.major_cap(),
                             cc.minor_cap(),
                             ss.str().c_str());
                   }))
#  else // _CCCL_HOSTED() && defined(CUB_DEBUG_LOG)
      log_dispatch("DeviceRunLengthEncode::Encode", cc, policy_getter());
#  endif // _CCCL_HOSTED() && defined(CUB_DEBUG_LOG)

      if CUB_DETAIL_CONSTEXPR_ISH (policy_getter().algorithm == RleAlgorithm::lookahead)
      {
        return invoke_lookahead(
          kernel_source,
          policy_getter().lookahead,
          d_temp_storage,
          temp_storage_bytes,
          d_in,
          d_unique_out,
          d_counts_out,
          d_num_runs_out,
          num_items,
          stream,
          launcher_factory);
      }
      else
      {
        return invoke_streaming<PolicySelector, streaming_kernel_source<PolicySelector>>(
          d_temp_storage, temp_storage_bytes, d_in, d_unique_out, d_counts_out, d_num_runs_out, num_items, stream);
      }
    });
  }
  else
#endif // __cccl_ptx_isa >= 920
  {
    return invoke_streaming<PolicySelector>(
      d_temp_storage, temp_storage_bytes, d_in, d_unique_out, d_counts_out, d_num_runs_out, num_items, stream);
  }
}
} // namespace detail::rle::encode

CUB_NAMESPACE_END
