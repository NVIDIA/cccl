// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_radix_sort.cuh>

#include <c2h/catch2_test_helper.h>

// Guard tests: each public DeviceSegmentedRadixSort method must resolve unambiguously
// to the legacy temp-storage overload when called in its minimal form (no explicit
// stream, begin_bit/end_bit defaulted), even though the env overloads are also in
// scope. If env-overload SFINAE drifts in the future, these become "ambiguous
// overload" compile errors.

C2H_TEST("DeviceSegmentedRadixSort::SortPairs legacy size-query is unambiguous", "[segmented_radix_sort][device]")
{
  void* d_temp_storage        = nullptr;
  size_t temp_storage_bytes   = 0;
  const int* d_keys_in        = nullptr;
  int* d_keys_out             = nullptr;
  const int* d_values_in      = nullptr;
  int* d_values_out           = nullptr;
  ::cuda::std::int64_t n      = 0;
  ::cuda::std::int64_t n_segs = 0;
  int* d_offsets              = nullptr;

  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedRadixSort::SortPairs(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_keys_out,
      d_values_in,
      d_values_out,
      n,
      n_segs,
      d_offsets,
      d_offsets));
}

C2H_TEST("DeviceSegmentedRadixSort::SortPairsDescending legacy size-query is unambiguous",
         "[segmented_radix_sort][device]")
{
  void* d_temp_storage        = nullptr;
  size_t temp_storage_bytes   = 0;
  const int* d_keys_in        = nullptr;
  int* d_keys_out             = nullptr;
  const int* d_values_in      = nullptr;
  int* d_values_out           = nullptr;
  ::cuda::std::int64_t n      = 0;
  ::cuda::std::int64_t n_segs = 0;
  int* d_offsets              = nullptr;

  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedRadixSort::SortPairsDescending(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_keys_out,
      d_values_in,
      d_values_out,
      n,
      n_segs,
      d_offsets,
      d_offsets));
}

C2H_TEST("DeviceSegmentedRadixSort::SortKeys legacy size-query is unambiguous", "[segmented_radix_sort][device]")
{
  void* d_temp_storage        = nullptr;
  size_t temp_storage_bytes   = 0;
  const int* d_keys_in        = nullptr;
  int* d_keys_out             = nullptr;
  ::cuda::std::int64_t n      = 0;
  ::cuda::std::int64_t n_segs = 0;
  int* d_offsets              = nullptr;

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedRadixSort::SortKeys(
            d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, n, n_segs, d_offsets, d_offsets));
}

C2H_TEST("DeviceSegmentedRadixSort::SortKeysDescending legacy size-query is unambiguous",
         "[segmented_radix_sort][device]")
{
  void* d_temp_storage        = nullptr;
  size_t temp_storage_bytes   = 0;
  const int* d_keys_in        = nullptr;
  int* d_keys_out             = nullptr;
  ::cuda::std::int64_t n      = 0;
  ::cuda::std::int64_t n_segs = 0;
  int* d_offsets              = nullptr;

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedRadixSort::SortKeysDescending(
            d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, n, n_segs, d_offsets, d_offsets));
}
