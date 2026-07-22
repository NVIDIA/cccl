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
#include <cub/device/dispatch/dispatch_streaming_reduce_by_key.cuh>
#include <cub/device/dispatch/kernels/kernel_rle_encode_lookahead.cuh>
#include <cub/device/dispatch/tuning/tuning_rle_encode.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>

#include <thrust/type_traits/is_contiguous_iterator.h>
#include <thrust/type_traits/unwrap_contiguous_iterator.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__device/compute_capability.h>
#include <cuda/__memory/align_up.h>
#include <cuda/__type_traits/is_trivially_copyable.h>
#include <cuda/iterator>
#include <cuda/std/cstdint>
#include <cuda/std/functional>

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
      policy.lookback.threads_per_block,
      policy.lookback.items_per_thread,
      policy.lookback.load_algorithm,
      policy.lookback.load_modifier,
      policy.lookback.scan_algorithm,
      policy.lookback.lookback_delay};
  }
};

template <class PolicySelector,
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
    policy_selector_adapter<PolicySelector>{});
}

// a preprocessor directive inside the NV_IF_TARGET argument list is undefined behavior (MSVC C5101),
// so the CUB_DEBUG_LOG guard has to live in this macro instead of around the _CubLog calls
#ifdef CUB_DEBUG_LOG
#  define CUB_DETAIL_RLE_ENCODE_LOG(...) _CubLog(__VA_ARGS__)
#else // ^^^ CUB_DEBUG_LOG ^^^ / vvv !CUB_DEBUG_LOG vvv
#  define CUB_DETAIL_RLE_ENCODE_LOG(...)
#endif // !CUB_DEBUG_LOG

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
  && ::cuda::std::is_signed_v<OffsetT> && (sizeof(OffsetT) == 4 || sizeof(OffsetT) == 8);

// Launches the lookahead init + main kernels. Callable from host and device: the host arm queries the
// device's opt-in shared memory and picks the tuned staged configuration when it fits, else the unstaged
// floor; device-side (CDP) callers cannot raise the dynamic shared memory limit, so they always launch
// the floor, which fits the default limit on every device. The kernels are passed in because their
// instantiation must stay in device-pass-visible text at the call site.
template <class KernelT,
          class InitKernelT,
          class InputIteratorT,
          class UniqueOutputIteratorT,
          class LengthsOutputIteratorT,
          class NumRunsOutputIteratorT,
          class OffsetT,
          class LauncherFactory>
CUB_RUNTIME_FUNCTION cudaError_t invoke_lookahead(
  KernelT kernel,
  InitKernelT init_kernel,
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

  const int num_tiles =
    static_cast<int>(::cuda::ceil_div(num_items, static_cast<OffsetT>(lookahead_policy.tile_size())));

  if (d_temp_storage == nullptr)
  {
    // + alignof: the tile states are aligned up inside the allocation, so any base pointer works
    temp_storage_bytes = static_cast<size_t>(num_tiles) * sizeof(TilePartialStateT) + alignof(TilePartialStateT);
    return cudaSuccess;
  }
  auto* tile_partial_states =
    static_cast<TilePartialStateT*>(::cuda::align_up(d_temp_storage, alignof(TilePartialStateT)));

  int key_ring_stages   = lookahead_policy.key_ring_stages;
  int pos_ring_stages   = lookahead_policy.pos_ring_stages;
  bool keys_staged      = true;
  size_t dyn_smem_bytes = lookahead_policy.dyn_smem_bytes(int{sizeof(key_t)}, int{alignof(key_t)});
  NV_IF_TARGET(NV_IS_HOST,
               ({
                 int device = 0, max_optin_smem = 0;
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
                   if (const auto error = CubDebug(
                         launcher_factory.set_max_dynamic_smem_size_for(kernel, static_cast<int>(dyn_smem_bytes))))
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
    key_ring_stages = lookahead_policy.floor_key_ring_stages();
    pos_ring_stages = lookahead_policy.floor_pos_ring_stages();
    dyn_smem_bytes  = lookahead_policy.floor_dyn_smem_bytes();
  }

  {
    constexpr int init_kernel_threads = 128;
    const auto init_grid_size         = ::cuda::ceil_div(num_tiles, init_kernel_threads);
    CUB_DETAIL_RLE_ENCODE_LOG(
      "Invoking DeviceRleEncodeLookaheadInitKernel<<<%d, %d, 0, %lld>>>()\n",
      init_grid_size,
      init_kernel_threads,
      (long long) stream);
    if (const auto error = CubDebug(
          launcher_factory(init_grid_size, init_kernel_threads, 0, stream, /* dependent_launch */ false)
            .doit(init_kernel, tile_partial_states, static_cast<::cuda::std::int64_t>(num_tiles))))
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
    CUB_DETAIL_RLE_ENCODE_LOG(
      "Invoking DeviceRleEncodeLookaheadKernel<<<%d, %d, %zu, %lld>>>()\n",
      num_tiles,
      block_dim,
      dyn_smem_bytes,
      (long long) stream);
    if (const auto error = CubDebug(
          launcher_factory(num_tiles,
                           block_dim,
                           static_cast<int>(dyn_smem_bytes),
                           stream,
                           /* dependent_launch */ false)
            .doit(kernel,
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
  [[maybe_unused]] LauncherFactory launcher_factory = {})
{
#if __cccl_ptx_isa >= 920
  if constexpr (lookahead_instantiable<InputIteratorT,
                                       UniqueOutputIteratorT,
                                       LengthsOutputIteratorT,
                                       NumRunsOutputIteratorT,
                                       OffsetT>)
  {
    using key_t      = it_value_t<InputIteratorT>;
    using length_t   = it_value_t<LengthsOutputIteratorT>;
    using num_runs_t = it_value_t<NumRunsOutputIteratorT>;

    // the kernel must be named OUTSIDE the host-only region
    [[maybe_unused]] auto kernel = DeviceRleEncodeLookaheadKernel<PolicySelector, key_t, length_t, num_runs_t, OffsetT>;
    [[maybe_unused]] auto init_kernel = DeviceRleEncodeLookaheadInitKernel<TilePartialStateT>;

    ::cuda::compute_capability cc{};
    if (const auto error = CubDebug(launcher_factory.PtxComputeCap(cc)))
    {
      return error;
    }
    return detail::dispatch_compute_cap(policy_selector, cc, [&](auto policy_getter) -> cudaError_t {
      if CUB_DETAIL_CONSTEXPR_ISH (policy_getter().algorithm == RleAlgorithm::lookahead)
      {
        return invoke_lookahead(
          kernel,
          init_kernel,
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
        return invoke_streaming<PolicySelector>(
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
