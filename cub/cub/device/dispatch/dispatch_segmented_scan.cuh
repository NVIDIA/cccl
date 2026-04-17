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

#include <cub/detail/arch_dispatch.cuh>
#include <cub/detail/choose_offset.cuh>
#include <cub/detail/launcher/cuda_runtime.cuh>
#include <cub/detail/type_traits.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/device/dispatch/kernels/kernel_segmented_scan.cuh>
#include <cub/device/dispatch/tuning/tuning_segmented_scan.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_temporary_storage.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_unsigned.h>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_scan
{
enum class worker
{
  block,
  warp,
  thread
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
      EnforceInclusive == ForceInclusive::Yes>);

  CUB_DEFINE_KERNEL_GETTER(
    warp_segmented_scan_kernel,
    device_warp_segmented_scan_kernel<
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
      EnforceInclusive == ForceInclusive::Yes>);

  CUB_DEFINE_KERNEL_GETTER(
    thread_segmented_scan_kernel,
    device_thread_segmented_scan_kernel<
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
      EnforceInclusive == ForceInclusive::Yes>);

  CUB_RUNTIME_FUNCTION static constexpr size_t AccumSize()
  {
    return sizeof(AccumT);
  }
};

// select the accumulator type using an overload set, so __accumulator_t is not instantiated when an overriding
// accumulator type is present. This matches the reduce dispatch pattern and is needed by CCCL.C.
template <typename ScanOpT, typename InitValueT, typename InputValueT>
_CCCL_API auto select_accum_t(detail::use_default*) -> ::cuda::std::__accumulator_t<
  ScanOpT,
  InputValueT,
  ::cuda::std::_If<::cuda::std::is_same_v<InitValueT, NullType>, InputValueT, typename InitValueT::value_type>>;

template <typename ScanOpT,
          typename InitValueT,
          typename InputValueT,
          typename OverrideAccumT,
          ::cuda::std::enable_if_t<!::cuda::std::is_same_v<OverrideAccumT, detail::use_default>, int> = 0>
_CCCL_API auto select_accum_t(OverrideAccumT*) -> OverrideAccumT;

template <
  typename OverrideAccumT = detail::use_default,
  typename InputIteratorT,
  typename OutputIteratorT,
  typename BeginOffsetIteratorInputT,
  typename EndOffsetIteratorInputT,
  typename BeginOffsetIteratorOutputT,
  typename ScanOpT,
  typename InitValueT,
  typename AccumT = decltype(select_accum_t<ScanOpT, InitValueT, cub::detail::it_value_t<InputIteratorT>>(
    static_cast<OverrideAccumT*>(nullptr))),
  ForceInclusive EnforceInclusive = ForceInclusive::No,
  typename OffsetT                = typename detail::
    common_iterator_value_t<BeginOffsetIteratorInputT, EndOffsetIteratorInputT, BeginOffsetIteratorOutputT>,
  typename PolicySelector = detail::segmented_scan::policy_selector_from_types<AccumT>,
  typename KernelSource   = detail::segmented_scan::device_segmented_scan_kernel_source<
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
  int num_segments_per_worker,
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
    return cudaSuccess;
  }

  ::cuda::arch_id arch_id{};
  if (const auto error = CubDebug(launcher_factory.PtxArchId(arch_id)))
  {
    return error;
  }

  return detail::dispatch_arch(policy_selector, arch_id, [&](auto policy_getter) {
    const segmented_scan_policy active_policy = policy_getter();

    // Clamp to produce a positive integer
    num_segments_per_worker = (::cuda::std::max) (num_segments_per_worker, 1);

    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }

    active_policy.CheckLoadModifier();

    _CCCL_ASSERT(num_segments_per_worker > 0, "Number of segments per worker parameter must be positive");

    const auto [workers_per_block, block_size] = [&](worker selector) -> ::cuda::std::tuple<int, int> {
      switch (selector)
      {
        case worker::block: {
          const auto bw = active_policy.block;
          return {1, bw.block_threads};
        }
        case worker::warp: {
          const auto ww = active_policy.warp;
          const auto bt = ww.block_threads;
          _CCCL_ASSERT(0 == bt % detail::warp_threads,
                       "Warp worker requires block size to be a multiple of native warp-size");
          return {int(bt >> cub::detail::log2_warp_threads), ww.block_threads};
        }
        case worker::thread: {
          const auto tw = active_policy.thread;
          const auto bt = tw.block_threads;
          return {bt, bt};
        }
        default:
          _CCCL_UNREACHABLE();
      }
      _CCCL_UNREACHABLE();
    }(worker_choice);

    const auto segments_per_block = num_segments_per_worker * workers_per_block;
    _CCCL_ASSERT(segments_per_block > 0, "Number of segments to be processed by block must be positive");

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

      auto launcher = launcher_factory(grid_size, block_size, 0, stream);

      // Cast is safe, since OffsetT is integral with sizeof(OffsetT) >= 4, and num_segments_per_invocation
      // fits in int32_t by construction
      const auto segment_count = static_cast<OffsetT>(num_segments_per_invocation);

      switch (worker_choice)
      {
        case worker::block:
          launcher.doit(
            kernel_source.segmented_scan_kernel(),
            d_in,
            d_out,
            input_begin_offsets,
            input_end_offsets,
            output_begin_offsets,
            segment_count,
            scan_op,
            init_value,
            num_segments_per_worker);
          break;
        case worker::warp:
          launcher.doit(
            kernel_source.warp_segmented_scan_kernel(),
            d_in,
            d_out,
            input_begin_offsets,
            input_end_offsets,
            output_begin_offsets,
            segment_count,
            scan_op,
            init_value,
            num_segments_per_worker);
          break;
        case worker::thread:
          launcher.doit(
            kernel_source.thread_segmented_scan_kernel(),
            d_in,
            d_out,
            input_begin_offsets,
            input_end_offsets,
            output_begin_offsets,
            segment_count,
            scan_op,
            init_value,
            num_segments_per_worker);
          break;
        default:
          _CCCL_UNREACHABLE();
      }

      cudaError_t error = CubDebug(cudaPeekAtLastError());
      if (cudaSuccess != error)
      {
        return error;
      }

      if (invocation_index + 1 < num_invocations)
      {
        input_begin_offsets += num_segments_per_invocation;
        input_end_offsets += num_segments_per_invocation;
        output_begin_offsets += num_segments_per_invocation;
      }

      error = CubDebug(detail::DebugSyncStream(stream));
      if (cudaSuccess != error)
      {
        return error;
      }
    }

    return cudaSuccess;
  });
}
} // namespace detail::segmented_scan

CUB_NAMESPACE_END
