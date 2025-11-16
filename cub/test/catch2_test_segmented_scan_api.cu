// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"
// above header needs to be included first

#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_scan.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/cmath>
#include <cuda/iterator>

#include <iostream> // std::cerr
#include <string>
#include <vector>

#include "catch2_test_device_scan.cuh"
#include <c2h/catch2_test_helper.h>

void check_execution_status(cudaError_t status, const std::string& algo_name)
{
  if (status != cudaSuccess)
  {
    std::cerr << algo_name << " failed with status: " << status << "\n";
  }
}

C2H_TEST("cub::DeviceSegmentedScan::ExclusiveSegmentedSum API with two offsets works",
         "[segmented][exclusive_sum][two_offsets]")
{
  const std::string& algo_name = "cub::DeviceSegmentedScan::ExclusiveSegmentedSum[2 offsets]";

  // example-begin exclusive-segmented-sum-two-offsets
  auto input        = thrust::device_vector<int>{1, 2, 3, 4, 5, 6, 7, 8};
  auto offsets      = thrust::device_vector<size_t>{0, 3, 5, 8};
  auto output       = thrust::device_vector<int>(input.size(), thrust::no_init);
  auto num_segments = offsets.size() - 1;

  void* temp_storage        = nullptr;
  size_t temp_storage_bytes = 0;

  auto d_in          = input.begin();
  auto d_out         = output.begin();
  auto begin_offsets = offsets.begin();
  auto end_offsets   = begin_offsets + 1;

  // get size of required temporary storage and allocate
  auto status = cub::DeviceSegmentedScan::ExclusiveSegmentedSum(
    temp_storage, temp_storage_bytes, d_in, d_out, begin_offsets, end_offsets, num_segments);
  check_execution_status(status, algo_name);

  status = cudaMalloc(&temp_storage, temp_storage_bytes);
  check_execution_status(status, "cudaMalloc");

  // run the algorithm
  status = cub::DeviceSegmentedScan::ExclusiveSegmentedSum(
    temp_storage, temp_storage_bytes, d_in, d_out, begin_offsets, end_offsets, num_segments);
  check_execution_status(status, algo_name);

  thrust::device_vector<int> expected{0, 1, 3, 0, 4, 0, 6, 13};
  // example-end exclusive-segmented-sum-two-offsets

  REQUIRE(status == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceSegmentedScan::ExclusiveSegmentedSum API with three offsets works",
         "[segmented][exclusive_sum][three_offsets]")
{
  const std::string& algo_name = "cub::DeviceSegmentedScan::ExclusiveSegmentedSum[3 offsets]";

  // example-begin exclusive-segmented-sum-three-offsets
  // Sequence of 16 values, representing 4x4 matrix in row-major layout
  auto input = thrust::device_vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

  // offsets to starts of each of 4 rows
  size_t row_size       = 4;
  auto in_begin_offsets = thrust::device_vector<size_t>{0, row_size, 2 * row_size};
  auto num_segments     = in_begin_offsets.size();
  // Perform row-wise sum for 3-by-3 principal sub-matrix
  size_t segment_size = 3;

  auto in_end_offsets = thrust::device_vector<size_t>{
    0 * row_size + segment_size, 1 * row_size + segment_size, 2 * row_size + segment_size};

  auto output            = thrust::device_vector<int>(num_segments * segment_size, thrust::no_init);
  auto out_begin_offsets = thrust::device_vector<size_t>{0, segment_size, 2 * segment_size};

  void* temp_storage        = nullptr;
  size_t temp_storage_bytes = 0;

  auto d_in_beg_offsets  = in_begin_offsets.begin();
  auto d_in_end_offsets  = in_end_offsets.begin();
  auto d_out_beg_offsets = out_begin_offsets.begin();

  auto d_in  = input.begin();
  auto d_out = output.begin();

  // get size of required storage and allocate
  auto status = cub::DeviceSegmentedScan::ExclusiveSegmentedSum(
    temp_storage, temp_storage_bytes, d_in, d_out, d_in_beg_offsets, d_in_end_offsets, d_out_beg_offsets, num_segments);
  check_execution_status(status, algo_name);

  status = cudaMalloc(&temp_storage, temp_storage_bytes);
  check_execution_status(status, "cudaMalloc");

  // run the algorithm
  status = cub::DeviceSegmentedScan::ExclusiveSegmentedSum(
    temp_storage, temp_storage_bytes, d_in, d_out, d_in_beg_offsets, d_in_end_offsets, d_out_beg_offsets, num_segments);
  check_execution_status(status, algo_name);

  thrust::device_vector<int> expected{0, 1, 3, 0, 5, 11, 0, 9, 19};
  // example-end exclusive-segmented-sum-three-offsets

  REQUIRE(status == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceSegmentedScan::InclusiveSegmentedSum API with two offsets works inplace",
         "[segmented][inclusive_sum][two_offsets]")
{
  const std::string& algo_name = "cub::DeviceSegmentedScan::InclusiveSegmentedSum[2 offsets]";
  // example-begin inclusive-segmented-sum-two-offsets
  auto input   = thrust::device_vector<int>{2, 1, 1, 2, 1, 2, 1, 1};
  auto offsets = thrust::device_vector<size_t>{0, 3, 5, 8};

  void* temp_storage        = nullptr;
  size_t temp_storage_bytes = 0;

  auto begin_offsets = offsets.begin();
  auto end_offsets   = begin_offsets + 1;
  auto num_segments  = offsets.size() - 1;

  auto d_in = input.begin();

  // get size of requires storage and allocate
  auto status = cub::DeviceSegmentedScan::InclusiveSegmentedSum(
    temp_storage, temp_storage_bytes, d_in, d_in, begin_offsets, end_offsets, num_segments);
  check_execution_status(status, algo_name);

  status = cudaMalloc(&temp_storage, temp_storage_bytes);
  check_execution_status(status, "cudaMalloc");

  // execute the algorithm
  status = cub::DeviceSegmentedScan::InclusiveSegmentedSum(
    temp_storage, temp_storage_bytes, d_in, d_in, begin_offsets, end_offsets, num_segments);
  check_execution_status(status, algo_name);

  thrust::device_vector<int> expected{2, 3, 4, 2, 3, 2, 3, 4};
  // example-end inclusive-segmented-sum-two-offsets

  REQUIRE(status == cudaSuccess);
  // input was modified inplace
  REQUIRE(input == expected);
}

C2H_TEST("cub::DeviceSegmentedScan::InclusiveSegmentedSum API with three offsets works",
         "[segmented][inclusive_sum][three_offsets]")
{
  const std::string& algo_name = "cub::DeviceSegmentedScan::InclusiveSegmentedSum[3 offsets]";
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

  // get size of temporary storage and allocate
  auto status = cub::DeviceSegmentedScan::InclusiveSegmentedSum(
    temp_storage, temp_storage_bytes, d_in, d_out, d_in_beg_offsets, d_in_end_offsets, d_out_beg_offsets, num_segments);
  check_execution_status(status, algo_name);

  status = cudaMalloc(&temp_storage, temp_storage_bytes);
  check_execution_status(status, algo_name);

  // Compute inclusive sum for each row prepended with 0
  status = cub::DeviceSegmentedScan::InclusiveSegmentedSum(
    temp_storage, temp_storage_bytes, d_in, d_out, d_in_beg_offsets, d_in_end_offsets, d_out_beg_offsets, num_segments);
  check_execution_status(status, algo_name);

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
  const std::string& algo_name = "cub::DeviceSegmentedScan::InclusiveSegmentedScanInit[2 offsets]";
  // example-begin inclusive-segmented-scan-init-two-offsets
  int prime  = 7;
  auto input = thrust::device_vector<int>{
    2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6};

  auto row_size    = static_cast<size_t>(prime);
  auto row_offsets = thrust::device_vector<size_t>{0, row_size, 2 * row_size, 3 * row_size, 4 * row_size, 5 * row_size};
  size_t num_segments = row_offsets.size() - 1;

  thrust::device_vector<int> output(input.size(), thrust::no_init);

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

  // get size of temporary storage and allocate
  auto status = cub::DeviceSegmentedScan::InclusiveSegmentedScanInit(
    temp_storage, temp_storage_bytes, d_in, d_out, d_in_beg_offsets, d_in_end_offsets, num_segments, scan_op, init_value);
  check_execution_status(status, algo_name);

  status = cudaMalloc(&temp_storage, temp_storage_bytes);
  check_execution_status(status, "cudaMalloc");

  // run the algorithm
  status = cub::DeviceSegmentedScan::InclusiveSegmentedScanInit(
    temp_storage, temp_storage_bytes, d_in, d_out, d_in_beg_offsets, d_in_end_offsets, num_segments, scan_op, init_value);
  check_execution_status(status, algo_name);

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
  const std::string& algo_name = "cub::DeviceSegmentedScan::ExclusiveSegmentedScan[2 offsets]";
  auto input                   = thrust::device_vector<unsigned>{
    0x64b40b1b, 0x7bf23c0c, 0xaa982e07, 0x652cae12, 0xb5411ce2, 0x6b1b7da4, 0xa04a0320, 0xd81dd7c0, 0x88a92288,
    0xd26bc5ab, 0x006645f6, 0xfdf477a4, 0x9afbfc45, 0x563d312d, 0x732653a5, 0x424fa289, 0x572b509a, 0xb2b1eb5e,
    0x03cb95ee, 0xdfd59321, 0x5defeea3, 0x0bdf0744, 0xc9895d47, 0x57c652d6, 0xc4a9f967, 0x665eb273, 0xcbd5aa63,
    0x7059b1b0, 0x75d72e04, 0x2d2577c7, 0x0a8f5913, 0x7113b3d7, 0x6ebd37c3, 0x738ec95f, 0xebea2011, 0x40c89a64,
    0xae8aca44, 0x600620da, 0x83456476, 0x3fb8b857, 0x070b9bb3, 0x4816bd31, 0xbfc3741e, 0x79dea106, 0x915c6344,
    0x18b29de0, 0x43acca5e, 0x92a02255, 0x2ddd27d2, 0x3b99996b, 0xde125f8b, 0x864d5acd, 0x79d40c79, 0xc5f60fbf,
    0xd1a71410, 0xd9cb4af9, 0xfdd9b196, 0xb09af168, 0x49f492af, 0xdfe6a0b1, 0x3b2ac7c1, 0xdd833839, 0x3425c3c2,
    0xde52cf5d, 0x61993a11, 0xb031c2a2, 0xa8209f30, 0xa03344bd, 0x03924ec3, 0xab55c23e, 0xb58286f8, 0x39668eda,
    0x7fb87b9c, 0x48b6eb6b, 0x99520666, 0x45d922e4, 0x2eacd69e, 0x5c67a6f1, 0xc037688f, 0xe01fe6d5, 0x909bfcab,
    0xd81a9b3b, 0xc6538ec6, 0xa1f8e3b5, 0xe4fd6301, 0xc947df61, 0x1fb54d95, 0xd2fa61e1, 0xd828fe18, 0xde128980,
    0x46d559c2, 0x10ee7bcf, 0x96559474, 0x32581ef0, 0xaab3dd8d, 0xb0e153e7, 0x24152e51, 0xc3c04f6d, 0x02c10c65,
    0x6a060a23, 0x5cc30312, 0x5d803563, 0x412eadc8, 0xf2c9af0a, 0xe4bc978a, 0xf30b8541, 0x10cd5dd8, 0xd1b720c8,
    0x6cc02e30, 0xd4a0e86f, 0x9db5bd3d, 0x17497b1f, 0x29e479ef, 0xdcd198aa, 0xb9bad1b6, 0x7e8a3128, 0xc808df66,
    0x83e8eb06, 0xa82782b5, 0x779045ae, 0x1ff3d860, 0x7492bc8c, 0xa8f95df2, 0x340b73ed, 0x14481ee0, 0xa673ddc3,
    0x292a814a, 0x39de627e};

  // example-begin exclusive-segmented-scan-two-offsets

  /* Compute exclusive scan using addition of GF(2) field represented
   * over boolean values stored as bits in unsigned integer, where addition is bitwise XOR.
   * Each unsigned integer represents 32-long tuple of GF(2) values
   */
  auto scan_op        = [] __host__ __device__(unsigned v1, unsigned v2) -> unsigned { return v1 ^ v2; };
  unsigned init_value = 0u;

  // 128 input elements
  // auto input = thrust::device_vector<unsigned>{0x64b40b1b, 0x7bf23c0c, 0xaa982e07, ... };

  // 4 segments
  auto offsets = thrust::device_vector<unsigned>{0, 40, 77, 101, 128};
  auto output  = thrust::device_vector<unsigned>(input.size(), thrust::no_init);

  void* temp_storage        = nullptr;
  size_t temp_storage_bytes = 0;

  auto d_in           = input.begin();
  auto d_out          = output.begin();
  auto begin_offsets  = offsets.begin();
  auto end_offsets    = offsets.begin() + 1;
  size_t num_segments = offsets.size() - 1;

  // inquire size of needed temporary storage and allocate
  auto status = cub::DeviceSegmentedScan::ExclusiveSegmentedScan(
    temp_storage, temp_storage_bytes, d_in, d_out, begin_offsets, end_offsets, num_segments, scan_op, init_value);
  check_execution_status(status, algo_name);

  status = cudaMalloc(&temp_storage, temp_storage_bytes);
  check_execution_status(status, "cudaMalloc");

  // run the algorithm
  status = cub::DeviceSegmentedScan::ExclusiveSegmentedScan(
    temp_storage, temp_storage_bytes, d_in, d_out, begin_offsets, end_offsets, num_segments, scan_op, init_value);
  check_execution_status(status, algo_name);
  // example-end exclusive-segmented-scan-two-offsets

  // verify correctness
  thrust::host_vector<unsigned> h_input(input);

  thrust::host_vector<unsigned> h_offsets(offsets);
  for (size_t id = 0; id < num_segments; ++id)
  {
    auto inp_b = h_input.begin() + h_offsets[id];
    auto inp_e = h_input.begin() + h_offsets[id + 1];
    auto out_b = h_input.begin() + h_offsets[id];
    compute_exclusive_scan_reference(inp_b, inp_e, out_b, init_value, scan_op);
  }
  REQUIRE(status == cudaSuccess);
  REQUIRE(output == h_input);
}

C2H_TEST("cub::DeviceSegmentedScan::InclusiveSegmentedScan API with three offsets works",
         "[segmented][exclusive_scan][three-offsets]")

{
  /*
  Example uses ExclusiveSegmentedScan to compute running maximum of a sequence of length n
  and writes the result to upper diagonal portion of n-by-n matrix.
  */
  const std::string& algo_name = "cub::DeviceSegmentedScan::InclusiveSegmentedScan[3-offsets]";
  // example-begin inclusive-segmented-scan-three-offsets
  size_t n = 8;
  thrust::device_vector<float> input{0.21f, 0.33f, 0.17f, 0.56f, 0.31f, 0.25f, 1.0f, 0.72f};

  constexpr unsigned _zero{0};
  auto _n               = static_cast<unsigned>(n);
  auto counting_it      = cuda::counting_iterator(_zero);
  auto in_begin_offsets = counting_it;
  auto in_end_offsets   = cuda::constant_iterator(_n);

  // use stride n + 1 is the distance between consecutive diagonal elements in C-contiguous layout
  auto out_begin_offsets = cuda::strided_iterator(counting_it, _n + 1);

  // allocate and zero-initialize output matrix in C-contiguous layout
  auto output = thrust::device_vector<float>(n * n, 0.0f);

  auto d_in  = input.begin();
  auto d_out = output.begin();

  auto scan_op = [] __host__ __device__(float v1, float v2) noexcept -> float { return cuda::maximum<>{}(v1, v2); };

  void* temp_storage = nullptr;
  size_t temp_storage_bytes;

  // determine size of required temporary storage and allocate
  auto status = cub::DeviceSegmentedScan::InclusiveSegmentedScan(
    temp_storage, temp_storage_bytes, d_in, d_out, in_begin_offsets, in_end_offsets, out_begin_offsets, n, scan_op);
  check_execution_status(status, algo_name);

  status = cudaMalloc(&temp_storage, temp_storage_bytes);
  check_execution_status(status, "cudaMalloc");

  // run the algorithm
  status = cub::DeviceSegmentedScan::InclusiveSegmentedScan(
    temp_storage, temp_storage_bytes, d_in, d_out, in_begin_offsets, in_end_offsets, out_begin_offsets, n, scan_op);
  check_execution_status(status, algo_name);

  thrust::device_vector<float> expected{
    0.21f, 0.33f, 0.33f, 0.56f, 0.56f, 0.56f, 1.00f, 1.00f, // row 0
    0.00f, 0.33f, 0.33f, 0.56f, 0.56f, 0.56f, 1.00f, 1.00f, // row 1
    0.00f, 0.00f, 0.17f, 0.56f, 0.56f, 0.56f, 1.00f, 1.00f, // row 2
    0.00f, 0.00f, 0.00f, 0.56f, 0.56f, 0.56f, 1.00f, 1.00f, // row 3
    0.00f, 0.00f, 0.00f, 0.00f, 0.31f, 0.31f, 1.00f, 1.00f, // row 4
    0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.25f, 1.00f, 1.00f, // row 5
    0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 1.00f, 1.00f, // row 6
    0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.72f // row 7
  };

  // example-end inclusive-segmented-scan-three-offsets
  REQUIRE(output == expected);
  REQUIRE(status == cudaSuccess);
}
