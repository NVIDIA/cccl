// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

#include <cub/detail/cc_dispatch.cuh>
#include <cub/detail/choose_offset.cuh>
#include <cub/detail/it_traits.cuh>
#include <cub/detail/launcher/cuda_runtime.cuh>
#include <cub/detail/logging.cuh>
#include <cub/detail/null_type.cuh>
#include <cub/detail/type_traits.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/device/dispatch/kernels/kernel_segmented_scan.cuh>
#include <cub/device/dispatch/kernels/kernel_segmented_scan_load_balanced.cuh>
#include <cub/device/dispatch/tuning/tuning_segmented_scan.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__host_stdlib/sstream>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/cstdint>
#include <cuda/std/optional>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_scan
{
enum class worker
{
  block
};

template <typename PolicySelector,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename BeginOffsetIteratorInputT,
          typename EndOffsetIteratorInputT,
          typename BeginOffsetIteratorOutputT,
          typename OffsetT,
          typename ScanOpT,
          typename InitValueT,
          typename AccumT,
          ForceInclusive EnforceInclusive>
struct device_segmented_scan_kernel_source
{
  static_assert(::cuda::std::is_empty_v<PolicySelector>);

  CUB_DEFINE_KERNEL_GETTER(
    segmented_scan_kernel,
    device_segmented_scan_kernel<
      PolicySelector,
      InputIteratorT,
      OutputIteratorT,
      BeginOffsetIteratorInputT,
      EndOffsetIteratorInputT,
      BeginOffsetIteratorOutputT,
      OffsetT,
      ScanOpT,
      InitValueT,
      AccumT,
      EnforceInclusive == ForceInclusive::Yes,
      false>);

  CUB_DEFINE_KERNEL_GETTER(
    segmented_scan_multi_segment_kernel,
    device_segmented_scan_kernel<
      PolicySelector,
      InputIteratorT,
      OutputIteratorT,
      BeginOffsetIteratorInputT,
      EndOffsetIteratorInputT,
      BeginOffsetIteratorOutputT,
      OffsetT,
      ScanOpT,
      InitValueT,
      AccumT,
      EnforceInclusive == ForceInclusive::Yes,
      true>);
};

template <typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename OutputBeginOffsetIteratorT = reuse_input_begin>
struct segment_offsets
{
  BeginOffsetIteratorT begin;
  EndOffsetIteratorT end;
  OutputBeginOffsetIteratorT output_begin;
};

template <typename BeginOffsetIteratorT, typename EndOffsetIteratorT, typename OutputBeginOffsetIteratorT>
using segment_offset_t = common_iterator_value_t<
  BeginOffsetIteratorT,
  EndOffsetIteratorT,
  ::cuda::std::conditional_t<::cuda::std::is_same_v<OutputBeginOffsetIteratorT, reuse_input_begin>,
                             BeginOffsetIteratorT,
                             OutputBeginOffsetIteratorT>>;

template <typename BeginOffsetIteratorT, typename EndOffsetIteratorT, typename OutputBeginOffsetIteratorT>
[[nodiscard]] CUB_RUNTIME_FUNCTION auto
output_begin_offsets(segment_offsets<BeginOffsetIteratorT, EndOffsetIteratorT, OutputBeginOffsetIteratorT> segments)
{
  if constexpr (::cuda::std::is_same_v<OutputBeginOffsetIteratorT, reuse_input_begin>)
  {
    return segments.begin;
  }
  else
  {
    return segments.output_begin;
  }
}

template <typename LoadBalancedPolicySelector,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename OutputBeginOffsetIteratorT,
          typename OffsetT,
          typename WorkT,
          typename ScanOpT,
          typename InitValueT,
          typename AccumT,
          ForceInclusive EnforceInclusive>
struct device_segmented_scan_load_balanced_kernel_source
{
  static_assert(::cuda::std::is_empty_v<LoadBalancedPolicySelector>);

  using work_t                  = WorkT;
  using work_index_scan_state_t = ScanTileState<contiguity_work_index_value<WorkT>>;
  using scan_tile_state_t       = ReduceByKeyScanTileState<AccumT, int>;
  using addressing_t            = segment_addressing<BeginOffsetIteratorT, OutputBeginOffsetIteratorT>;

  CUB_DEFINE_KERNEL_GETTER(segmented_scan_load_balanced_init_kernel,
                           device_segmented_scan_load_balanced_init_kernel<work_index_scan_state_t>);

  CUB_DEFINE_KERNEL_GETTER(
    segmented_scan_work_index_kernel,
    device_segmented_scan_work_index_kernel<BeginOffsetIteratorT,
                                            EndOffsetIteratorT,
                                            OutputBeginOffsetIteratorT,
                                            OffsetT,
                                            WorkT,
                                            scan_tile_state_t>);

  CUB_DEFINE_KERNEL_GETTER(
    segmented_scan_load_balanced_kernel,
    device_segmented_scan_load_balanced_kernel<
      LoadBalancedPolicySelector,
      InputIteratorT,
      OutputIteratorT,
      OffsetT,
      WorkT,
      ScanOpT,
      InitValueT,
      AccumT,
      EnforceInclusive == ForceInclusive::Yes,
      addressing_t>);
};

// CUB's default load-balanced policy does not depend on the compute capability, so the host can evaluate it.
template <typename AccumT>
struct default_load_balanced_policy_getter
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()() const
  {
    return load_balanced_policy_selector_from_types<AccumT>{}(::cuda::compute_capability{});
  }
};

// Whether the load-balanced kernel fits max_smem_per_block under CUB's default policy. A call that would exceed it
// scans one block per segment instead.
template <typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename WorkT,
          typename ScanOpT,
          typename InitValueT,
          typename AccumT,
          bool ForceInclusive,
          typename AddressingT>
inline constexpr bool default_load_balanced_policy_fits =
  load_balanced_kernel_smem_bytes<
    default_load_balanced_policy_getter<AccumT>,
    InputIteratorT,
    OutputIteratorT,
    OffsetT,
    WorkT,
    ScanOpT,
    InitValueT,
    AccumT,
    ForceInclusive,
    AddressingT>
  <= max_smem_per_block;

// Most tiles a tile state holds: its allocation adds one warp of padding to an int tile count.
inline constexpr int max_tile_state_tiles = ::cuda::std::numeric_limits<int>::max() - detail::warp_threads;

// Most segments the work index holds, one scan tile per work_index_tile_items segments.
inline constexpr ::cuda::std::int64_t max_work_index_segments =
  ::cuda::std::int64_t{max_tile_state_tiles} * work_index_tile_items;

struct work_index_plan
{
  int scan_tiles;
  size_t index_bytes;
  size_t scan_state_bytes;
};

template <typename KernelSource>
CUB_RUNTIME_FUNCTION cudaError_t plan_work_index(::cuda::std::int64_t num_segments, work_index_plan& plan)
{
  using work_t    = typename KernelSource::work_t;
  plan            = {};
  plan.scan_tiles = static_cast<int>(::cuda::ceil_div(num_segments, ::cuda::std::int64_t{work_index_tile_items}));
  // num_segments + 1 prefix sums of the segment sizes, then the contiguity flags
  plan.index_bytes = static_cast<size_t>(num_segments + 2) * sizeof(work_t);
  if (plan.scan_tiles > 1)
  {
    using scan_state_t = typename KernelSource::work_index_scan_state_t;
    return CubDebug(scan_state_t::AllocationSize(plan.scan_tiles, plan.scan_state_bytes));
  }
  return cudaSuccess;
}

template <typename OffsetT, typename SegmentsT, typename MainTileStateT, typename KernelSource, typename FactoryT>
CUB_RUNTIME_FUNCTION cudaError_t launch_work_index(
  const work_index_plan& plan,
  SegmentsT segments,
  ::cuda::std::int64_t num_segments,
  void* index_storage,
  void* scan_state_storage,
  MainTileStateT main_state,
  int main_tiles,
  cudaStream_t stream,
  KernelSource kernel_source,
  FactoryT& launcher_factory)
{
  using scan_state_t = typename KernelSource::work_index_scan_state_t;
  scan_state_t scan_state;
  if (plan.scan_tiles > 1)
  {
    if (const auto error = CubDebug(scan_state.Init(plan.scan_tiles, scan_state_storage, plan.scan_state_bytes)))
    {
      return error;
    }
    if (const auto error = CubDebug(
          launcher_factory(::cuda::ceil_div(plan.scan_tiles, work_index_threads), work_index_threads, 0, stream)
            .doit(kernel_source.segmented_scan_load_balanced_init_kernel(), scan_state, plan.scan_tiles)))
    {
      return error;
    }
    if (const auto error = CubDebug(DebugSyncStream(stream)))
    {
      return error;
    }
  }

  const int grid = (::cuda::std::max) (plan.scan_tiles, ::cuda::ceil_div(main_tiles, work_index_threads));
  if (const auto error = CubDebug(
        launcher_factory(grid, work_index_threads, 0, stream)
          .doit(kernel_source.segmented_scan_work_index_kernel(),
                segments.begin,
                segments.end,
                segments.output_begin,
                static_cast<OffsetT>(num_segments),
                static_cast<typename KernelSource::work_t*>(index_storage),
                scan_state,
                plan.scan_tiles,
                main_state,
                main_tiles)))
  {
    return error;
  }
  return CubDebug(DebugSyncStream(stream));
}

template <typename ScanOpT, typename InitValueT, typename InputValueT>
using deduced_accum_t = ::cuda::std::__accumulator_t<
  ScanOpT,
  InputValueT,
  ::cuda::std::conditional_t<::cuda::std::is_same_v<InitValueT, NullType>, InputValueT, typename InitValueT::value_type>>;

// Launches kernel so that block b of each launch scans the segments_per_block consecutive segments starting at
// b * segments_per_block, in launches of at most INT32_MAX segments. num_segments must be positive.
template <typename OffsetT,
          typename KernelT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename BeginOffsetIteratorInputT,
          typename EndOffsetIteratorInputT,
          typename BeginOffsetIteratorOutputT,
          typename ScanOpT,
          typename InitValueT,
          typename KernelLauncherFactory>
CUB_RUNTIME_FUNCTION cudaError_t launch_blocks(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  KernelT kernel,
  int segments_per_block,
  ::cuda::compute_capability cc,
  const SegmentedScanPolicy& active_policy,
  InputIteratorT d_in,
  OutputIteratorT d_out,
  ::cuda::std::int64_t num_segments,
  BeginOffsetIteratorInputT input_begin_offsets,
  EndOffsetIteratorInputT input_end_offsets,
  BeginOffsetIteratorOutputT output_begin_offsets,
  ScanOpT scan_op,
  InitValueT init_value,
  cudaStream_t stream,
  KernelLauncherFactory& launcher_factory)
{
  detail::log_dispatch("DeviceSegmentedScan", cc, active_policy);

  if (d_temp_storage == nullptr)
  {
    temp_storage_bytes = 1;
    return cudaSuccess;
  }

  _CCCL_ASSERT((active_policy.block.load_modifier != CacheLoadModifier::LOAD_LDG),
               "The memory consistency model does not apply to texture accesses");
  _CCCL_ASSERT(active_policy.block.threads_per_block > 0, "Policy value for threads_per_block is not positive");
  _CCCL_ASSERT(active_policy.block.items_per_thread > 0, "Policy value for items_per_thread is not positive");
  _CCCL_ASSERT(segments_per_block > 0, "Number of segments to be processed by block must be positive");
  _CCCL_ASSERT(segments_per_block <= active_policy.block.max_segments,
               "Number of segments per block exceeds maximum value");

  static constexpr auto int32_max                       = ::cuda::std::numeric_limits<::cuda::std::int32_t>::max();
  static constexpr auto max_num_segments_per_invocation = static_cast<::cuda::std::int64_t>(int32_max);

  const ::cuda::std::int64_t num_invocations = ::cuda::ceil_div(num_segments, max_num_segments_per_invocation);

  for (::cuda::std::int64_t invocation_index = 0; invocation_index < num_invocations; invocation_index++)
  {
    const auto current_seg_offset          = invocation_index * max_num_segments_per_invocation;
    const auto next_seg_offset             = current_seg_offset + max_num_segments_per_invocation;
    const auto num_segments_per_invocation = ::cuda::std::min(next_seg_offset, num_segments) - current_seg_offset;

    _CCCL_ASSERT(num_segments_per_invocation <= max_num_segments_per_invocation,
                 "data loss during narrowing: num_segments_per_invocation exceeds int32_t range");

    const auto grid_size = ::cuda::ceil_div(static_cast<int>(num_segments_per_invocation), segments_per_block);

    auto launcher = launcher_factory(grid_size, active_policy.block.threads_per_block, 0, stream);

    // Cast is safe, since OffsetT is integral with sizeof(OffsetT) >= 4, and num_segments_per_invocation
    // fits in int32_t by construction
    const auto segment_count = static_cast<OffsetT>(num_segments_per_invocation);

    if (const auto error = CubDebug(launcher.doit(
          kernel,
          d_in,
          d_out,
          input_begin_offsets,
          input_end_offsets,
          output_begin_offsets,
          segment_count,
          scan_op,
          init_value,
          segments_per_block));
        cudaSuccess != error)
    {
      return error;
    }

    if (const auto error = CubDebug(cudaPeekAtLastError()); cudaSuccess != error)
    {
      return error;
    }

    if (invocation_index + 1 < num_invocations)
    {
      input_begin_offsets += num_segments_per_invocation;
      input_end_offsets += num_segments_per_invocation;
      output_begin_offsets += num_segments_per_invocation;
    }

    if (const auto error = CubDebug(DebugSyncStream(stream)); cudaSuccess != error)
    {
      return error;
    }
  }

  return cudaSuccess;
}

template <
  ForceInclusive EnforceInclusive = ForceInclusive::No,
  typename InputIteratorT,
  typename OutputIteratorT,
  typename BeginOffsetIteratorInputT,
  typename EndOffsetIteratorInputT,
  typename BeginOffsetIteratorOutputT,
  typename ScanOpT,
  typename InitValueT,
  typename AccumT = deduced_accum_t<ScanOpT, InitValueT, it_value_t<InputIteratorT>>,
  typename OffsetT =
    common_iterator_value_t<BeginOffsetIteratorInputT, EndOffsetIteratorInputT, BeginOffsetIteratorOutputT>,
  typename PolicySelector = policy_selector_from_types<AccumT>,
  typename KernelSource   = device_segmented_scan_kernel_source<
    PolicySelector,
    InputIteratorT,
    OutputIteratorT,
    BeginOffsetIteratorInputT,
    EndOffsetIteratorInputT,
    BeginOffsetIteratorOutputT,
    OffsetT,
    ScanOpT,
    InitValueT,
    AccumT,
    EnforceInclusive>,
  typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
#if _CCCL_HAS_CONCEPTS()
  requires segmented_scan_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE auto dispatch(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  InputIteratorT d_in,
  OutputIteratorT d_out,
  ::cuda::std::int64_t num_segments,
  BeginOffsetIteratorInputT input_begin_offsets,
  EndOffsetIteratorInputT input_end_offsets,
  BeginOffsetIteratorOutputT output_begin_offsets,
  ScanOpT scan_op,
  InitValueT init_value,
  worker worker_choice,
  cudaStream_t stream,
  PolicySelector policy_selector         = {},
  KernelSource kernel_source             = {},
  KernelLauncherFactory launcher_factory = {})
{
  static_assert(::cuda::std::is_integral_v<OffsetT> && sizeof(OffsetT) >= 4 && sizeof(OffsetT) <= 8,
                "dispatch_segmented_scan only supports integral offset types of 4- or 8-bytes");

  if (num_segments <= 0)
  {
    return (num_segments == 0) ? cudaSuccess : cudaErrorUnknown;
  }

  ::cuda::compute_capability cc{};
  if (const auto error = CubDebug(launcher_factory.PtxComputeCap(cc)))
  {
    return error;
  }

  const SegmentedScanPolicy active_policy = policy_selector(cc);

  switch (worker_choice)
  {
    case worker::block:
      return launch_blocks<OffsetT>(
        d_temp_storage,
        temp_storage_bytes,
        kernel_source.segmented_scan_kernel(),
        1,
        cc,
        active_policy,
        d_in,
        d_out,
        num_segments,
        input_begin_offsets,
        input_end_offsets,
        output_begin_offsets,
        scan_op,
        init_value,
        stream,
        launcher_factory);
    default:
      _CCCL_UNREACHABLE();
  }
  _CCCL_UNREACHABLE();
}

// Segments per block: two tiles of items divided by the mean segment length, at most max_segments. Scan several
// segments only when at least four fit.
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int segments_per_block_for_mean(
  int tile_items, int max_segments, ::cuda::std::int64_t num_segments, ::cuda::std::int64_t num_items) noexcept
{
  if (num_segments <= 0 || num_items <= 0)
  {
    return 1;
  }

  // A mean below one item, with more segments than items, counts as one.
  const auto mean_items = (::cuda::std::max) (num_items / num_segments, ::cuda::std::int64_t{1});
  const auto segments   = 2 * ::cuda::std::int64_t{tile_items} / mean_items;
  return segments >= 4 ? static_cast<int>((::cuda::std::min) (segments, ::cuda::std::int64_t{max_segments})) : 1;
}

// The one-segment-per-block dispatch with block b taking segments [b * n, (b + 1) * n), where n is selected from
// num_items, the exact total item count.
template <ForceInclusive EnforceInclusive = ForceInclusive::No,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename BeginOffsetIteratorInputT,
          typename EndOffsetIteratorInputT,
          typename BeginOffsetIteratorOutputT,
          typename ScanOpT,
          typename InitValueT,
          typename AccumT = deduced_accum_t<ScanOpT, InitValueT, it_value_t<InputIteratorT>>,
          typename OffsetT =
            common_iterator_value_t<BeginOffsetIteratorInputT, EndOffsetIteratorInputT, BeginOffsetIteratorOutputT>,
          typename PolicySelector        = policy_selector_from_types<AccumT>,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
#if _CCCL_HAS_CONCEPTS()
  requires segmented_scan_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch_with_num_items(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  InputIteratorT d_in,
  OutputIteratorT d_out,
  ::cuda::std::int64_t num_segments,
  BeginOffsetIteratorInputT input_begin_offsets,
  EndOffsetIteratorInputT input_end_offsets,
  BeginOffsetIteratorOutputT output_begin_offsets,
  ScanOpT scan_op,
  InitValueT init_value,
  ::cuda::std::int64_t num_items,
  cudaStream_t stream,
  PolicySelector policy_selector         = {},
  KernelLauncherFactory launcher_factory = {})
{
  if (num_items < 0)
  {
    return cudaErrorInvalidValue;
  }

  const device_segmented_scan_kernel_source<
    PolicySelector,
    InputIteratorT,
    OutputIteratorT,
    BeginOffsetIteratorInputT,
    EndOffsetIteratorInputT,
    BeginOffsetIteratorOutputT,
    OffsetT,
    ScanOpT,
    InitValueT,
    AccumT,
    EnforceInclusive>
    kernel_source{};

  if (num_segments > 0)
  {
    ::cuda::compute_capability cc{};
    if (const auto error = CubDebug(launcher_factory.PtxComputeCap(cc)))
    {
      return error;
    }
    const SegmentedScanPolicy active_policy = policy_selector(cc);
    const int tile_items = active_policy.block.threads_per_block * active_policy.block.items_per_thread;
    int segments_per_block =
      segments_per_block_for_mean(tile_items, active_policy.block.max_segments, num_segments, num_items);
    // A block of several segments places its items in make_unsigned_t<OffsetT>, and its last tile must not wrap.
    using logical_offset_t = ::cuda::std::make_unsigned_t<OffsetT>;
    if (::cuda::std::cmp_greater(num_items, ::cuda::std::numeric_limits<logical_offset_t>::max() - tile_items))
    {
      segments_per_block = 1;
    }

    if (segments_per_block > 1)
    {
      return launch_blocks<OffsetT>(
        d_temp_storage,
        temp_storage_bytes,
        kernel_source.segmented_scan_multi_segment_kernel(),
        segments_per_block,
        cc,
        active_policy,
        d_in,
        d_out,
        num_segments,
        input_begin_offsets,
        input_end_offsets,
        output_begin_offsets,
        scan_op,
        init_value,
        stream,
        launcher_factory);
    }
  }

  return dispatch<EnforceInclusive>(
    d_temp_storage,
    temp_storage_bytes,
    d_in,
    d_out,
    num_segments,
    input_begin_offsets,
    input_end_offsets,
    output_begin_offsets,
    scan_op,
    init_value,
    worker::block,
    stream,
    policy_selector,
    kernel_source,
    launcher_factory);
}

// num_items is the exact total item count, or ::cuda::std::nullopt when the caller does not supply it.
template <
  ForceInclusive EnforceInclusive = ForceInclusive::No,
  typename InputIteratorT,
  typename OutputIteratorT,
  typename BeginOffsetIteratorT,
  typename EndOffsetIteratorT,
  typename OutputBeginOffsetIteratorT,
  typename ScanOpT,
  typename InitValueT,
  typename NumItemsT,
  typename AccumT         = deduced_accum_t<ScanOpT, InitValueT, it_value_t<InputIteratorT>>,
  typename OffsetT        = segment_offset_t<BeginOffsetIteratorT, EndOffsetIteratorT, OutputBeginOffsetIteratorT>,
  typename PolicySelector = policy_selector_from_types<AccumT>,
  typename LoadBalancedPolicySelector = load_balanced_policy_selector_from_types<AccumT>,
  typename KernelSource               = device_segmented_scan_load_balanced_kernel_source<
    LoadBalancedPolicySelector,
    InputIteratorT,
    OutputIteratorT,
    BeginOffsetIteratorT,
    EndOffsetIteratorT,
    OutputBeginOffsetIteratorT,
    OffsetT,
    load_balanced_work_t<OffsetT, !::cuda::std::is_same_v<NumItemsT, ::cuda::std::nullopt_t>>,
    ScanOpT,
    InitValueT,
    AccumT,
    EnforceInclusive>,
  typename WideKernelSource = device_segmented_scan_load_balanced_kernel_source<
    LoadBalancedPolicySelector,
    InputIteratorT,
    OutputIteratorT,
    BeginOffsetIteratorT,
    EndOffsetIteratorT,
    OutputBeginOffsetIteratorT,
    OffsetT,
    wide_load_balanced_work_t,
    ScanOpT,
    InitValueT,
    AccumT,
    EnforceInclusive>,
  typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
#if _CCCL_HAS_CONCEPTS()
  requires segmented_scan_policy_selector<PolicySelector>
        && segmented_scan_load_balanced_policy_selector<LoadBalancedPolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch_load_balanced(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  InputIteratorT d_in,
  OutputIteratorT d_out,
  ::cuda::std::int64_t num_segments,
  segment_offsets<BeginOffsetIteratorT, EndOffsetIteratorT, OutputBeginOffsetIteratorT> segments,
  ScanOpT scan_op,
  InitValueT init_value,
  [[maybe_unused]] NumItemsT num_items,
  cudaStream_t stream,
  PolicySelector policy_selector                                            = {},
  [[maybe_unused]] LoadBalancedPolicySelector load_balanced_policy_selector = {},
  KernelSource kernel_source                                                = {},
  [[maybe_unused]] WideKernelSource wide_kernel_source                      = {},
  KernelLauncherFactory launcher_factory                                    = {})
{
  static_assert(::cuda::std::is_integral_v<OffsetT> && sizeof(OffsetT) >= 4 && sizeof(OffsetT) <= 8,
                "dispatch_load_balanced only supports integral offset types of 4- or 8-bytes");
  constexpr bool has_num_items = !::cuda::std::is_same_v<NumItemsT, ::cuda::std::nullopt_t>;
  static_assert(!has_num_items || ::cuda::std::is_same_v<NumItemsT, ::cuda::std::int64_t>,
                "num_items is a ::cuda::std::int64_t or ::cuda::std::nullopt");

  if constexpr (has_num_items)
  {
    if (num_items < 0)
    {
      return cudaErrorInvalidValue;
    }
  }
  if (num_segments <= 0)
  {
    return num_segments == 0 ? cudaSuccess : cudaErrorUnknown;
  }

  const auto dispatch_blocks = [&] {
    using fallback_output_begin_t = decltype(output_begin_offsets(segments));
    return dispatch<EnforceInclusive,
                    InputIteratorT,
                    OutputIteratorT,
                    BeginOffsetIteratorT,
                    EndOffsetIteratorT,
                    fallback_output_begin_t,
                    ScanOpT,
                    InitValueT,
                    AccumT,
                    OffsetT>(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      num_segments,
      segments.begin,
      segments.end,
      output_begin_offsets(segments),
      scan_op,
      init_value,
      worker::block,
      stream,
      policy_selector,
      device_segmented_scan_kernel_source<
        PolicySelector,
        InputIteratorT,
        OutputIteratorT,
        BeginOffsetIteratorT,
        EndOffsetIteratorT,
        fallback_output_begin_t,
        OffsetT,
        ScanOpT,
        InitValueT,
        AccumT,
        EnforceInclusive>{},
      launcher_factory);
  };

  // Scans with the kernels of source, whose work type holds the total work.
  const auto dispatch_balanced = [&](auto source) -> cudaError_t {
    using source_t = decltype(source);
    using work_t   = typename source_t::work_t;

    // An accumulator too large for the load-balanced kernel under CUB's default policy takes one block per segment,
    // and the kernel is not instantiated for it.
    if constexpr (::cuda::std::is_same_v<LoadBalancedPolicySelector, load_balanced_policy_selector_from_types<AccumT>>
                  && !default_load_balanced_policy_fits<
                     InputIteratorT,
                     OutputIteratorT,
                     OffsetT,
                     work_t,
                     ScanOpT,
                     typename InitValueT::value_type,
                     AccumT,
                     EnforceInclusive == ForceInclusive::Yes,
                     typename source_t::addressing_t>)
    {
      return dispatch_blocks();
    }
    else
    {
      // The work index holds num_segments + 1 prefixes indexed by OffsetT, scanned in at most max_tile_state_tiles
      // tiles. Calls with more segments take one block per segment, which launches in pieces of at most INT32_MAX
      // segments.
      constexpr auto offset_max = ::cuda::std::numeric_limits<OffsetT>::max();
      const bool fits_load_balanced =
        ::cuda::std::cmp_less(num_segments, offset_max) && num_segments <= max_work_index_segments;
      if (!fits_load_balanced)
      {
        return dispatch_blocks();
      }

      ::cuda::compute_capability cc{};
      if (const auto error = CubDebug(launcher_factory.PtxComputeCap(cc)))
      {
        return error;
      }

      const SegmentedScanLoadBalancedPolicy active_policy = load_balanced_policy_selector(cc);
      log_dispatch("DeviceSegmentedScan", cc, active_policy);

      _CCCL_ASSERT((active_policy.load_modifier != CacheLoadModifier::LOAD_LDG),
                   "The memory consistency model does not apply to texture accesses");
      _CCCL_ASSERT(active_policy.threads_per_block > 0, "Policy value for threads_per_block is not positive");
      _CCCL_ASSERT(active_policy.items_per_thread > 0, "Policy value for items_per_thread is not positive");

      const int threads_per_block = active_policy.threads_per_block;

      int blocks_per_sm{};
      if (const auto error = CubDebug(launcher_factory.MaxSmOccupancy(
            blocks_per_sm, source.segmented_scan_load_balanced_kernel(), threads_per_block)))
      {
        return error;
      }
      int sm_count{};
      if (const auto error = CubDebug(launcher_factory.MultiProcessorCount(sm_count)))
      {
        return error;
      }

      if (blocks_per_sm <= 0)
      {
        return cudaErrorInvalidValue;
      }
      const int resident_blocks = blocks_per_sm * sm_count;
      int capacity              = resident_blocks * max_factor;

      if constexpr (has_num_items)
      {
        const work_t tile_size =
          static_cast<work_t>(threads_per_block) * static_cast<work_t>(active_policy.items_per_thread);
        capacity =
          (::cuda::std::max) (1, load_balanced_block_count(static_cast<work_t>(num_items), tile_size, resident_blocks));
      }

      using scan_tile_state_t = typename source_t::scan_tile_state_t;
      size_t tile_state_bytes{};
      if (const auto error = CubDebug(scan_tile_state_t::AllocationSize(capacity, tile_state_bytes)))
      {
        return error;
      }
      work_index_plan plan{};
      if (const auto error = plan_work_index<source_t>(num_segments, plan))
      {
        return error;
      }

      void* allocations[3]             = {};
      const size_t allocation_sizes[3] = {tile_state_bytes, plan.index_bytes, plan.scan_state_bytes};
      if (const auto error =
            CubDebug(detail::alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes)))
      {
        return error;
      }
      if (d_temp_storage == nullptr)
      {
        return cudaSuccess;
      }

      scan_tile_state_t tile_state;
      if (const auto error = CubDebug(tile_state.Init(capacity, allocations[0], allocation_sizes[0])))
      {
        return error;
      }
      if (const auto error = launch_work_index<OffsetT>(
            plan,
            segments,
            num_segments,
            allocations[1],
            allocations[2],
            tile_state,
            capacity,
            stream,
            source,
            launcher_factory))
      {
        return error;
      }

      auto launcher = launcher_factory(capacity, threads_per_block, 0, stream);
      if (const auto error = CubDebug(launcher.doit(
            source.segmented_scan_load_balanced_kernel(),
            d_in,
            d_out,
            static_cast<const work_t*>(allocations[1]),
            static_cast<OffsetT>(num_segments),
            scan_op,
            init_value,
            resident_blocks,
            capacity,
            tile_state,
            typename source_t::addressing_t{segments.begin, segments.output_begin})))
      {
        return error;
      }
      if (const auto error = CubDebug(cudaPeekAtLastError()))
      {
        return error;
      }
      return CubDebug(DebugSyncStream(stream));
    }
  };

  // A supplied total that the work type of kernel_source cannot hold takes the 64-bit work of wide_kernel_source.
  using work_t = typename KernelSource::work_t;
  if constexpr (has_num_items && !::cuda::std::is_same_v<work_t, typename WideKernelSource::work_t>)
  {
    if (::cuda::std::cmp_greater(num_items, ::cuda::std::numeric_limits<work_t>::max()))
    {
      return dispatch_balanced(wide_kernel_source);
    }
  }
  return dispatch_balanced(kernel_source);
}
} // namespace detail::segmented_scan

CUB_NAMESPACE_END
