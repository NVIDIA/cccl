// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cub/detail/deferred_parameter.cuh>
#include <cub/detail/launcher/cuda_runtime.cuh>
#include <cub/device/dispatch/kernels/kernel_radix_sort_deferred.cuh>
#include <cub/util_device.cuh>
#include <cub/util_runs_on.cuh>

#include <cuda/std/limits>

CUB_NAMESPACE_BEGIN

namespace detail::radix_sort
{
template <
  SortOrder Order,
  typename KeyT,
  typename ValueT,
  typename NumItemsT,
  typename DecomposerT    = identity_decomposer_t,
  typename PolicySelector = policy_selector_from_types<KeyT, ValueT, detail::num_items_offset_t<NumItemsT>, true>,
  typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
CUB_RUNTIME_FUNCTION cudaError_t dispatch_deferred(
  void* storage,
  size_t& bytes,
  DoubleBuffer<KeyT>& keys,
  DoubleBuffer<ValueT>& values,
  NumItemsT num_items,
  int begin_bit,
  int end_bit,
  bool can_overwrite,
  cudaStream_t stream,
  const experimental::RunsOn& runs_on = {},
  DecomposerT decomposer              = {},
  PolicySelector policy_selector      = {},
  KernelLauncherFactory launcher      = {})
{
  using offset_t      = detail::num_items_offset_t<NumItemsT>;
  using count_t       = detail::parameter_from_host_t<offset_t, NumItemsT>;
  const count_t count = detail::make_num_items_kernel_arg(num_items);

  offset_t max_items{};
  if (!can_overwrite)
  {
    // Runtime bounds share a type with unbounded counts, so check the effective
    // host bound before sizing the intermediate arrays, without reading N.
    using element_t        = typename ::cuda::args::__traits<NumItemsT>::element_type;
    const auto upper_bound = ::cuda::args::__highest_(num_items);
    if (::cuda::std::cmp_less(upper_bound, 0) || upper_bound == (::cuda::std::numeric_limits<element_t>::max)())
    {
      return cudaErrorInvalidValue;
    }
    max_items = detail::num_items_upper_bound(num_items);
  }
  if (begin_bit < 0 || end_bit < begin_bit || end_bit > radix::traits_t<KeyT>::default_end_bit(decomposer))
  {
    return cudaErrorInvalidValue;
  }
  if (begin_bit == end_bit && can_overwrite)
  {
    if (storage == nullptr)
    {
      bytes = 1;
    }
    return cudaSuccess;
  }

  experimental::DeviceDescription description;
  if (const auto error = CubDebug(runs_on.concrete_description(launcher, storage, description)))
  {
    return error;
  }
  const int sms = static_cast<int>(description.__max_sm_count());
  if (begin_bit == end_bit)
  {
    if (storage == nullptr)
    {
      bytes = 1;
      return cudaSuccess;
    }
    if (const auto error = CubDebug(
          launcher(8 * sms, 256, 0, stream)
            .doit(deferred_radix_sort_copy_kernel<KeyT, ValueT, offset_t, count_t>,
                  keys.Alternate(),
                  keys.Current(),
                  values.Alternate(),
                  values.Current(),
                  count)))
    {
      return error;
    }
    return CubDebug(detail::DebugSyncStream(stream));
  }
  const auto cc = description.__compute_capability();
  return dispatch_compute_cap(policy_selector, cc, [&](auto policy_getter) -> cudaError_t {
    using policy_getter_t        = decltype(policy_getter);
    const RadixSortPolicy policy = policy_getter();

    static_assert(policy_getter_t{}().algorithm == RadixSortAlgorithm::onesweep,
                  "Deferred DeviceRadixSort counts require a tuning policy selecting onesweep.");
    constexpr int radix_bits = policy_getter_t{}().onesweep.radix_bits;
    constexpr int digits     = 1 << radix_bits;
    constexpr int tile_items =
      policy_getter_t{}().onesweep.threads_per_block * policy_getter_t{}().onesweep.items_per_thread;
    static_assert(policy_getter_t{}().histogram.radix_bits == radix_bits
                    && policy_getter_t{}().exclusive_sum.radix_bits == radix_bits,
                  "deferred radix sort requires matching histogram, scan, and Onesweep radix bits");
    // Keep all packed prefixes and signed batch-relative arithmetic below 2^28.
    static_assert(tile_items > 0 && tile_items < (1 << 28), "Onesweep tiles must fit the packed lookback counter");
    constexpr int batch_tiles =
      (::cuda::std::min) (policy_getter_t{}().onesweep.tiles_per_batch, ((1 << 28) - 1) / tile_items);
    static_assert(batch_tiles > 0, "Deferred radix sort batch size must be positive");
    const int passes                = ::cuda::ceil_div(end_bit - begin_bit, radix_bits);
    constexpr size_t value_size     = ::cuda::std::is_same_v<ValueT, NullType> ? 0 : sizeof(ValueT);
    const size_t intermediate_items = !can_overwrite && passes > 1 ? static_cast<size_t>(max_items) : 0;
    constexpr size_t bookkeeping_bytes =
      2 * size_t{batch_tiles} * digits * sizeof(int) + 2 * digits * sizeof(offset_t) + sizeof(radix_sort_batch_state);
    constexpr int num_allocations = 6;
    // Match alias_temporaries, including per-allocation and initial alignment slack.
    constexpr size_t storage_alignment = 256;
    const size_t fixed_bytes           = bookkeeping_bytes + static_cast<size_t>(passes) * digits * sizeof(offset_t)
                                       + (num_allocations + 1) * storage_alignment;
    if (intermediate_items > ((::cuda::std::numeric_limits<size_t>::max)() - fixed_bytes) / (sizeof(KeyT) + value_size))
    {
      return cudaErrorInvalidValue;
    }
    const size_t allocation_sizes[num_allocations] = {
      static_cast<size_t>(passes) * digits * sizeof(offset_t),
      2 * size_t{batch_tiles} * digits * sizeof(int),
      2 * digits * sizeof(offset_t),
      sizeof(radix_sort_batch_state),
      intermediate_items * sizeof(KeyT),
      intermediate_items * value_size};
    void* allocations[num_allocations]{};
    if (const auto error = detail::alias_temporaries(storage, bytes, allocations, allocation_sizes))
    {
      return error;
    }
    if (storage == nullptr)
    {
      return cudaSuccess;
    }
    auto* bins           = static_cast<offset_t*>(allocations[0]);
    auto* lookback       = static_cast<int*>(allocations[1]);
    auto* batch_bins_in  = static_cast<offset_t*>(allocations[2]);
    auto* batch_bins_out = batch_bins_in + digits;
    auto* state          = static_cast<radix_sort_batch_state*>(allocations[3]);
    // Dynamic, monotonic claiming gives resident CTAs the earliest unfinished
    // tiles. A future-batch waiter cannot strand an unclaimed predecessor in an
    // unscheduled CTA. No grid barrier depends on all launched CTAs being resident.
    const int grid = ::cuda::std::min(batch_tiles, 8 * sms);
    auto histogram = DeviceRadixSortHistogramKernel<PolicySelector, Order, KeyT, offset_t, DecomposerT, count_t>;
    int histogram_occupancy{};
    if (const auto error =
          CubDebug(launcher.MaxSmOccupancy(histogram_occupancy, histogram, policy.histogram.threads_per_block, 0)))
    {
      return error;
    }
    if (const auto error = CubDebug(cudaMemsetAsync(bins, 0, allocation_sizes[0], stream)))
    {
      return error;
    }
    if (const auto error = CubDebug(launcher(histogram_occupancy * sms, policy.histogram.threads_per_block, 0, stream)
                                      .doit(histogram, bins, keys.Current(), count, begin_bit, end_bit, decomposer)))
    {
      return error;
    }
    if (const auto error = CubDebug(launcher(passes, policy.exclusive_sum.threads_per_block, 0, stream)
                                      .doit(DeviceRadixSortExclusiveSumKernel<PolicySelector, offset_t>, bins)))
    {
      return error;
    }
    // As in immediate Onesweep, choose the first destination by pass parity so
    // the final pass lands in the caller's output without writing to the input.
    KeyT* keys_out     = keys.Alternate();
    ValueT* values_out = values.Alternate();
    auto* keys_tmp     = static_cast<KeyT*>(allocations[4]);
    auto* values_tmp   = static_cast<ValueT*>(allocations[5]);
    if (!can_overwrite && passes % 2 == 0)
    {
      keys.d_buffers[keys.selector ^ 1]     = keys_tmp;
      values.d_buffers[values.selector ^ 1] = values_tmp;
    }
    for (int pass = 0, bit = begin_bit; pass < passes; ++pass, bit += radix_bits)
    {
      constexpr int init_threads = 256;
      constexpr int num_states   = 2 * batch_tiles * digits;
      constexpr int init_blocks  = ::cuda::ceil_div(num_states, init_threads);
      if (const auto error = CubDebug(
            launcher(init_blocks, init_threads, 0, stream)
              .doit(deferred_radix_sort_init_kernel<offset_t>,
                    lookback,
                    num_states,
                    state,
                    batch_bins_in,
                    bins + pass * digits,
                    digits)))
      {
        return error;
      }
      if (const auto error = CubDebug(
            launcher(grid, policy.onesweep.threads_per_block, 0, stream)
              .doit(deferred_onesweep_kernel<PolicySelector, Order, KeyT, ValueT, offset_t, count_t, DecomposerT>,
                    lookback,
                    state,
                    batch_bins_in,
                    batch_bins_out,
                    keys.Alternate(),
                    keys.Current(),
                    values.Alternate(),
                    values.Current(),
                    count,
                    batch_tiles,
                    bit,
                    ::cuda::std::min(radix_bits, end_bit - bit),
                    decomposer)))
      {
        return error;
      }
      if (!can_overwrite && pass == 0)
      {
        keys = passes % 2 == 0 ? DoubleBuffer<KeyT>(keys_out, keys_tmp) : DoubleBuffer<KeyT>(keys_tmp, keys_out);
        values =
          passes % 2 == 0 ? DoubleBuffer<ValueT>(values_out, values_tmp) : DoubleBuffer<ValueT>(values_tmp, values_out);
      }
      keys.selector ^= 1;
      values.selector ^= 1;
    }
    return CubDebug(detail::DebugSyncStream(stream));
  });
}
} // namespace detail::radix_sort

CUB_NAMESPACE_END
