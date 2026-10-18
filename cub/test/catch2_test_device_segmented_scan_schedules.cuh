// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/device/device_segmented_scan.cuh>

#include <cuda/std/cstdint>

#include <c2h/catch2_test_helper.h>

// Bytes of temporary storage that DeviceSegmentedScan allocates under the schedule properties in schedule_env, for the
// expected_allocation_size that the environment launch wrappers check. The public interface has no size query that
// takes an environment, so this asks the dispatch that the public overloads forward to, as
// catch2_test_device_reduce_env.cu does for its determinism levels. init_value is cub::NullType{} or a
// cub::detail::InputValue. A two-offset overload passes its begin offsets as d_out_begins and ReuseInputBegin = true.
template <cub::ForceInclusive EnforceInclusive,
          bool ReuseInputBegin,
          typename ScheduleEnvT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename OutputBeginOffsetIteratorT,
          typename ScanOpT,
          typename InitValueT>
size_t schedule_allocation_size(
  const ScheduleEnvT& schedule_env,
  InputIteratorT d_in,
  OutputIteratorT d_out,
  cuda::std::int64_t num_segments,
  BeginOffsetIteratorT d_begins,
  EndOffsetIteratorT d_ends,
  OutputBeginOffsetIteratorT d_out_begins,
  ScanOpT scan_op,
  InitValueT init_value)
{
  using accum_t =
    cub::detail::segmented_scan::deduced_accum_t<ScanOpT, InitValueT, cub::detail::it_value_t<InputIteratorT>>;
  size_t bytes{};
  REQUIRE(
    cudaSuccess
    == cub::detail::segmented_scan::dispatch_from_env<EnforceInclusive, ReuseInputBegin>(
      schedule_env,
      nullptr,
      bytes,
      d_in,
      d_out,
      num_segments,
      d_begins,
      d_ends,
      d_out_begins,
      scan_op,
      init_value,
      cudaStream_t{},
      cub::detail::segmented_scan::policy_selector_from_types<accum_t>{}));
  return bytes;
}
