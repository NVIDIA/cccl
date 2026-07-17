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

#include <cub/detail/launcher/cuda_runtime.cuh>
#include <cub/device/dispatch/kernels/kernel_rle_encode_lookahead.cuh>
#include <cub/device/dispatch/tuning/tuning_rle_encode.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>

#include <thrust/type_traits/is_contiguous_iterator.h>
#include <thrust/type_traits/unwrap_contiguous_iterator.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__device/compute_capability.h>
#include <cuda/__memory/is_aligned.h>
#include <cuda/__type_traits/is_trivially_copyable.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail::rle::encode
{
// compile-time half of the lookahead viability: the kernel may only be instantiated for these types.
// The runtime half (device smem opt-in, tile count, temporary-storage alignment) lives in
// try_dispatch_lookahead and falls back to the streaming path.
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
  && (16 % sizeof(it_value_t<InputIteratorT>) == 0) && (alignof(it_value_t<InputIteratorT>) <= 16)
  && ::cuda::std::is_signed_v<OffsetT> && (sizeof(OffsetT) == 4 || sizeof(OffsetT) == 8);

// Dispatches the lookahead implementation when the tuning policy selects it and the runtime gates pass.
// `handled` reports whether this call satisfied the request (size query or launch); when false, the caller
// must dispatch the streaming implementation with the same arguments. The size query accounts for BOTH
// implementations: the runtime gates may route the same allocation to either path across calls.
template <class PolicySelector,
          class InputIteratorT,
          class UniqueOutputIteratorT,
          class LengthsOutputIteratorT,
          class NumRunsOutputIteratorT,
          class OffsetT,
          class StreamingSizeFnT,
          class LauncherFactory = detail::TripleChevronFactory>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t try_dispatch_lookahead(
  [[maybe_unused]] void* d_temp_storage,
  [[maybe_unused]] size_t& temp_storage_bytes,
  [[maybe_unused]] InputIteratorT d_in,
  [[maybe_unused]] UniqueOutputIteratorT d_unique_out,
  [[maybe_unused]] LengthsOutputIteratorT d_counts_out,
  [[maybe_unused]] NumRunsOutputIteratorT d_num_runs_out,
  [[maybe_unused]] OffsetT num_items,
  [[maybe_unused]] cudaStream_t stream,
  [[maybe_unused]] PolicySelector policy_selector,
  [[maybe_unused]] StreamingSizeFnT streaming_size_fn,
  bool& handled,
  [[maybe_unused]] LauncherFactory launcher_factory = {})
{
  handled = false;
  if constexpr (lookahead_instantiable<InputIteratorT,
                                       UniqueOutputIteratorT,
                                       LengthsOutputIteratorT,
                                       NumRunsOutputIteratorT,
                                       OffsetT>)
  {
    using key_t      = it_value_t<InputIteratorT>;
    using length_t   = it_value_t<LengthsOutputIteratorT>;
    using num_runs_t = it_value_t<NumRunsOutputIteratorT>;

    // the kernel must be named OUTSIDE the host-only region below: the device compilation pass has to see
    // this instantiation, or no device code is emitted for it (the host pass alone only creates launch stubs)
    [[maybe_unused]] auto kernel = DeviceRleEncodeLookaheadKernel<PolicySelector, key_t, length_t, num_runs_t, OffsetT>;
    [[maybe_unused]] auto init_kernel = DeviceRleEncodeLookaheadInitKernel<TilePartialStateT>;

    NV_IF_TARGET(
      NV_IS_HOST,
      (
        ::cuda::compute_capability cc{}; if (const auto error = CubDebug(ptx_compute_cap(cc))) {
          return error;
        } const RleEncodePolicy policy = policy_selector(cc);
        if (policy.algorithm != RleAlgorithm::lookahead) { return cudaSuccess; }

        const auto num_tiles = ::cuda::ceil_div(static_cast<::cuda::std::int64_t>(num_items),
                                                static_cast<::cuda::std::int64_t>(policy.lookahead.tile_size()));

        // the size query must cover both implementations: the same allocation may serve either across calls
        size_t streaming_bytes = 0;
        if (const auto error = CubDebug(streaming_size_fn(streaming_bytes))) { return error; } const size_t required =
          (::cuda::std::max) (streaming_bytes, static_cast<size_t>(num_tiles) * sizeof(TilePartialStateT));
        if (d_temp_storage == nullptr) {
          temp_storage_bytes = required;
          handled            = true;
          return cudaSuccess;
        } if (temp_storage_bytes < required) {
          handled = true;
          return CubDebug(cudaErrorInvalidValue);
        }

        // runtime gates: any failure falls back to the streaming implementation (the allocation covers it)
        if (num_items <= 0 || cc < ::cuda::compute_capability{10, 0} || num_tiles > 0x7fffffff
            || !::cuda::is_aligned(d_temp_storage, alignof(TilePartialStateT))) {
          return cudaSuccess;
        } int max_dynamic_smem_size = 0;
        if (const auto error = CubDebug(launcher_factory.max_dynamic_smem_size_for(max_dynamic_smem_size, kernel))) {
          return error;
        } const size_t dyn_smem_bytes = policy.lookahead.dyn_smem_bytes(int{sizeof(key_t)}, int{alignof(key_t)});
        if (static_cast<size_t>(max_dynamic_smem_size) < dyn_smem_bytes) {
          return cudaSuccess;
        } if (const auto error =
                CubDebug(launcher_factory.set_max_dynamic_smem_size_for(kernel, static_cast<int>(dyn_smem_bytes)))) {
          return error;
        } handled = true;

        auto* tile_partial_states = static_cast<TilePartialStateT*>(d_temp_storage);
        {
          constexpr int init_kernel_threads = 128;
          const auto init_grid_size         = static_cast<int>(::cuda::ceil_div(num_tiles, init_kernel_threads));
#ifdef CUB_DEBUG_LOG
          _CubLog("Invoking DeviceRleEncodeLookaheadInitKernel<<<%d, %d, 0, %lld>>>()\n",
                  init_grid_size,
                  init_kernel_threads,
                  (long long) stream);
#endif // CUB_DEBUG_LOG
          if (const auto error = CubDebug(
                launcher_factory(init_grid_size, init_kernel_threads, 0, stream, /* dependent_launch */ false)
                  .doit(init_kernel, tile_partial_states, static_cast<long long>(num_tiles))))
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
        } {
          const int block_dim = num_total_threads(policy.lookahead);
#ifdef CUB_DEBUG_LOG
          _CubLog("Invoking DeviceRleEncodeLookaheadKernel<<<%d, %d, %zu, %lld>>>()\n",
                  static_cast<int>(num_tiles),
                  block_dim,
                  dyn_smem_bytes,
                  (long long) stream);
#endif // CUB_DEBUG_LOG
          if (const auto error = CubDebug(
                launcher_factory(static_cast<int>(num_tiles),
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
                        static_cast<int>(num_tiles))))
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
        }))
  }
  return cudaSuccess;
}
} // namespace detail::rle::encode

CUB_NAMESPACE_END
