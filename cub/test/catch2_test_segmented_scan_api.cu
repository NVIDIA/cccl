// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "insert_nested_NVTX_range_guard.h"
// above header needs to be included first

#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_scan.cuh>

#include <thrust/device_vector.h>

#include <cuda/cmath>

#include <iostream>
#include <vector>

#include <c2h/catch2_test_helper.h>

C2H_TEST("cub::DeviceSegmentedScan::ExclusiveSegmentedSum API with two offsets works",
         "[segmented][exclusive_sum][two_offsets]")
{
  // example-begin exclusive-segmented-sum-two-offsets
  auto input   = thrust::device_vector<int>{1, 2, 3, 4, 5, 6, 7, 8};
  auto offsets = thrust::device_vector<size_t>{0, 3, 5, 8};
  auto output  = thrust::device_vector<int>(input.size());

  void* temp_storage        = nullptr;
  size_t temp_storage_bytes = 0;

  auto begin_offsets = offsets.begin();
  auto end_offsets   = begin_offsets + 1;
  auto num_segments  = offsets.size() - 1;

  auto d_in  = input.begin();
  auto d_out = output.begin();

  auto status = cub::DeviceSegmentedScan::ExclusiveSegmentedSum(
    temp_storage, temp_storage_bytes, d_in, d_out, begin_offsets, end_offsets, num_segments);
  if (status != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedScan::ExclusiveSegmentedSum failed with status: " << status << std::endl;
  }

  status = cudaMalloc(&temp_storage, temp_storage_bytes);
  if (status != cudaSuccess)
  {
    std::cerr << "cudaMalloc failed with status: " << status << std::endl;
  }

  status = cub::DeviceSegmentedScan::ExclusiveSegmentedSum(
    temp_storage, temp_storage_bytes, d_in, d_out, begin_offsets, end_offsets, num_segments);
  if (status != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedScan::ExclusiveSegmentedSum failed with status: " << status << std::endl;
  }

  thrust::device_vector<int> expected{0, 1, 3, 0, 4, 0, 6, 13};
  // example-end exclusive-segmented-sum-two-offsets

  REQUIRE(status == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceSegmentedScan::ExclusiveSegmentedSum API with three offsets works",
         "[segmented][exclusive_sum][three_offsets]")
{
  // example-begin exclusive-segmented-sum-three-offsets
  // Sequence of 16 values, representing 4x4 matrix in row-major layout
  auto input = thrust::device_vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

  // offsets to starts of each of 4 rows
  size_t row_size       = 4;
  auto in_begin_offsets = thrust::device_vector<size_t>{0, row_size, 2 * row_size};
  auto num_segments     = in_begin_offsets.size();
  // Perform row-wise sum for 3-by-3 principal sub-matrix
  size_t segment_size = 3;

  auto in_end_offsets =
    thrust::device_vector<size_t>{0 + segment_size, row_size + segment_size, 2 * row_size + segment_size};

  auto output            = thrust::device_vector<int>(num_segments * segment_size);
  auto out_begin_offsets = thrust::device_vector<size_t>{0, segment_size, 2 * segment_size};

  void* temp_storage        = nullptr;
  size_t temp_storage_bytes = 0;

  auto d_in_beg_offsets  = in_begin_offsets.begin();
  auto d_in_end_offsets  = in_end_offsets.begin();
  auto d_out_beg_offsets = out_begin_offsets.begin();

  auto d_in  = input.begin();
  auto d_out = output.begin();

  auto status = cub::DeviceSegmentedScan::ExclusiveSegmentedSum(
    temp_storage, temp_storage_bytes, d_in, d_out, d_in_beg_offsets, d_in_end_offsets, d_out_beg_offsets, num_segments);
  if (status != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedScan::ExclusiveSegmentedSum failed with status: " << status << std::endl;
  }

  status = cudaMalloc(&temp_storage, temp_storage_bytes);
  if (status != cudaSuccess)
  {
    std::cerr << "cudaMalloc failed with status: " << status << std::endl;
  }

  status = cub::DeviceSegmentedScan::ExclusiveSegmentedSum(
    temp_storage, temp_storage_bytes, d_in, d_out, d_in_beg_offsets, d_in_end_offsets, d_out_beg_offsets, num_segments);
  if (status != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedScan::ExclusiveSegmentedSum failed with status: " << status << std::endl;
  }

  thrust::device_vector<int> expected{0, 1, 3, 0, 5, 11, 0, 9, 19};
  // example-end exclusive-segmented-sum-three-offsets

  REQUIRE(status == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceSegmentedScan::InclusiveSegmentedSum API with two offsets works inplace",
         "[segmented][inclusive_sum][two_offsets]")
{
  // example-begin inclusive-segmented-sum-two-offsets
  auto input   = thrust::device_vector<int>{2, 1, 1, 2, 1, 2, 1, 1};
  auto offsets = thrust::device_vector<size_t>{0, 3, 5, 8};

  void* temp_storage        = nullptr;
  size_t temp_storage_bytes = 0;

  auto begin_offsets = offsets.begin();
  auto end_offsets   = begin_offsets + 1;
  auto num_segments  = offsets.size() - 1;

  auto d_in = input.begin();

  auto status = cub::DeviceSegmentedScan::InclusiveSegmentedSum(
    temp_storage, temp_storage_bytes, d_in, d_in, begin_offsets, end_offsets, num_segments);
  if (status != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedScan::InclusiveSegmentedSum failed with status: " << status << std::endl;
  }

  status = cudaMalloc(&temp_storage, temp_storage_bytes);
  if (status != cudaSuccess)
  {
    std::cerr << "cudaMalloc failed with status: " << status << std::endl;
  }

  status = cub::DeviceSegmentedScan::InclusiveSegmentedSum(
    temp_storage, temp_storage_bytes, d_in, d_in, begin_offsets, end_offsets, num_segments);
  if (status != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedScan::InclusiveSegmentedSum failed with status: " << status << std::endl;
  }

  thrust::device_vector<int> expected{2, 3, 4, 2, 3, 2, 3, 4};
  // example-end inclusive-segmented-sum-two-offsets

  REQUIRE(status == cudaSuccess);
  // input was modified inplace
  REQUIRE(input == expected);
}

C2H_TEST("cub::DeviceSegmentedScan::InclusiveSegmentedSum API with three offsets works",
         "[segmented][inclusive_sum][three_offsets]")
{
  // example-begin inclusive-segmented-sum-three-offsets
  // Sequence of 16 values, representing 4x4 matrix in row-major layout
  auto input = thrust::device_vector<int>{1, 1, 1, 1, -1, -1, -1, -1, 2, 2, 2, 2, -2, -2, -2, -2};

  // begin offsets for each of 4 rows
  size_t m          = 4;
  auto row_offsets  = thrust::device_vector<size_t>{0, m, 2 * m, 3 * m, 4 * m};
  auto num_segments = row_offsets.size() - 1;

  // Allocate m rows of m + 1 filled with zero-initialized values
  auto output = thrust::device_vector<int>((m + 1) * m, 0);
  // begin offsets to second element of each row
  size_t lda             = m + 1;
  auto out_begin_offsets = thrust::device_vector<size_t>{1, lda + 1, 2 * lda + 1, 3 * lda + 1};

  void* temp_storage        = nullptr;
  size_t temp_storage_bytes = 0;

  auto d_in_beg_offsets  = row_offsets.begin();
  auto d_in_end_offsets  = row_offsets.begin() + 1;
  auto d_out_beg_offsets = out_begin_offsets.begin();

  auto d_in  = input.begin();
  auto d_out = output.begin();

  auto status = cub::DeviceSegmentedScan::InclusiveSegmentedSum(
    temp_storage, temp_storage_bytes, d_in, d_out, d_in_beg_offsets, d_in_end_offsets, d_out_beg_offsets, num_segments);
  if (status != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedScan::InclusiveSegmentedSum failed with status: " << status << std::endl;
  }

  status = cudaMalloc(&temp_storage, temp_storage_bytes);
  if (status != cudaSuccess)
  {
    std::cerr << "cudaMalloc failed with status: " << status << std::endl;
  }

  // Compute inclusive sum for each row prepended with 0
  status = cub::DeviceSegmentedScan::InclusiveSegmentedSum(
    temp_storage, temp_storage_bytes, d_in, d_out, d_in_beg_offsets, d_in_end_offsets, d_out_beg_offsets, num_segments);
  if (status != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedScan::InclusiveSegmentedSum failed with status: " << status << std::endl;
  }

  // thrust::device_vector<int> expected{0, 1, 2, 3, 4, 0, -1, -2, -3, -4, 0, 2, 4, 6, 8, 0, -2, -4, -6, -8};

  std::vector<int> h_expected{};
  h_expected.reserve(output.size());
  std::vector<std::vector<int>> expected_rows{
    {0, 1, 2, 3, 4}, {0, -1, -2, -3, -4}, {0, 2, 4, 6, 8}, {0, -2, -4, -6, -8}};
  for (const auto& row : expected_rows)
  {
    h_expected.insert(h_expected.end(), row.begin(), row.end());
  }

  auto expected = thrust::device_vector<int>{h_expected};

  // example-end inclusive-segmented-sum-three-offsets

  REQUIRE(status == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceSegmentedScan::InclusiveSegmentedScanInit API with two offsets works",
         "[segmented][inclusive_scan_init][two_offsets]")
{
  // example-begin inclusive-segmented-scan-init-two-offsets
  int prime  = 7;
  auto input = thrust::device_vector<int>{
    2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6};

  auto row_size    = static_cast<size_t>(prime);
  auto row_offsets = thrust::device_vector<size_t>{0, row_size, 2 * row_size, 3 * row_size, 4 * row_size, 5 * row_size};
  size_t num_segments = row_offsets.size() - 1;

  auto output = thrust::device_vector<int>(input.size());

  auto m_p = cuda::fast_mod_div(prime);
  // Binary operator to multiply arguments using modular arithmetic
  auto scan_op = [=] __host__ __device__(int v1, int v2) -> int {
    const auto proj_v1 = (v1 % m_p);
    const auto proj_v2 = (v2 % m_p);
    return (proj_v1 * proj_v2) % m_p;
  };
  int init_value = 1;

  auto d_in  = input.begin();
  auto d_out = output.begin();

  auto d_in_beg_offsets = row_offsets.begin();
  auto d_in_end_offsets = row_offsets.begin() + 1;

  void* temp_storage        = nullptr;
  size_t temp_storage_bytes = 0;

  auto status = cub::DeviceSegmentedScan::InclusiveSegmentedScanInit(
    temp_storage, temp_storage_bytes, d_in, d_out, d_in_beg_offsets, d_in_end_offsets, num_segments, scan_op, init_value);
  if (status != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedScan::InclusiveSegmentedScanInit failed with status: " << status << std::endl;
  }

  status = cudaMalloc(&temp_storage, temp_storage_bytes);
  if (status != cudaSuccess)
  {
    std::cerr << "cudaMalloc failed with status: " << status << std::endl;
  }

  status = cub::DeviceSegmentedScan::InclusiveSegmentedScanInit(
    temp_storage, temp_storage_bytes, d_in, d_out, d_in_beg_offsets, d_in_end_offsets, num_segments, scan_op, init_value);
  if (status != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedScan::InclusiveSegmentedScanInit failed with status: " << status << std::endl;
  }

  std::vector<int> h_expected{};
  h_expected.reserve(output.size());
  std::vector<std::vector<int>> expected_rows{
    {2, 4, 1, 2, 4, 1, 2}, {3, 2, 6, 4, 5, 1, 3}, {4, 2, 1, 4, 2, 1, 4}, {5, 4, 6, 2, 3, 1, 5}, {6, 1, 6, 1, 6, 1, 6}};
  for (const auto& row : expected_rows)
  {
    h_expected.insert(h_expected.end(), row.begin(), row.end());
  }

  auto expected = thrust::device_vector<int>{h_expected};
  // example-end inclusive-segmented-scan-init-two-offsets

  REQUIRE(status == cudaSuccess);
  REQUIRE(expected == output);
}

C2H_TEST("cub::DeviceSegmentedScan::ExclusiveSegmentedScan API with two offsets works",
         "[segmented][exclusive_scan][two-offsets]")
{
  // example-begin exclusive-segmented-scan-two-offsets
  // example-end exclusive-segmented-scan-two-offsets
}
