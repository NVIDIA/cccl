// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_run_length_decode.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>

#include <iostream>

#include "cub_test_macros.h"

CUB_TEST("cub::DeviceRunLengthDecode::Decode works with int data elements", "[run_length_decode][device]", CUB_SMALL)
{
  // example-begin decode-run-lengths
  constexpr int num_runs                       = 5;
  const thrust::device_vector<int> run_values  = {0, 2, 9, 5, 8};
  const thrust::device_vector<int> run_lengths = {1, 2, 0, 3, 1};

  // The decoded sequence has as many items as the sum of the run lengths
  thrust::device_vector<int> decoded(7, thrust::no_init);

  // Determine the temporary device storage requirements
  size_t temp_storage_bytes = 0;
  auto error                = cub::DeviceRunLengthDecode::Decode(
    nullptr, temp_storage_bytes, run_values.begin(), run_lengths.begin(), decoded.begin(), num_runs);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceRunLengthDecode::Decode failed with status: " << error << '\n';
  }

  thrust::device_vector<char> temp_storage(temp_storage_bytes, thrust::no_init);

  // Run the decoding
  error = cub::DeviceRunLengthDecode::Decode(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_storage_bytes,
    run_values.begin(),
    run_lengths.begin(),
    decoded.begin(),
    num_runs);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceRunLengthDecode::Decode failed with status: " << error << '\n';
  }

  const thrust::device_vector<int> expected = {0, 2, 2, 5, 5, 5, 8};
  // example-end decode-run-lengths

  REQUIRE(error == cudaSuccess);
  REQUIRE(decoded == expected);
}

CUB_TEST("cub::DeviceRunLengthDecode::DecodeFromOffsets works with int data elements",
         "[run_length_decode][device]",
         CUB_SMALL)
{
  // example-begin decode-run-offsets
  constexpr int num_runs                      = 5;
  const thrust::device_vector<int> run_values = {0, 2, 9, 5, 8};

  // The i-th run occupies the positions [run_offsets[i], run_offsets[i + 1]) of the decoded sequence
  const thrust::device_vector<int> run_offsets = {0, 1, 3, 3, 6, 7};

  thrust::device_vector<int> decoded(7, thrust::no_init);

  // Determine the temporary device storage requirements
  size_t temp_storage_bytes = 0;
  auto error                = cub::DeviceRunLengthDecode::DecodeFromOffsets(
    nullptr, temp_storage_bytes, run_values.begin(), run_offsets.begin(), decoded.begin(), num_runs);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceRunLengthDecode::DecodeFromOffsets failed with status: " << error << '\n';
  }

  thrust::device_vector<char> temp_storage(temp_storage_bytes, thrust::no_init);

  // Run the decoding
  error = cub::DeviceRunLengthDecode::DecodeFromOffsets(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_storage_bytes,
    run_values.begin(),
    run_offsets.begin(),
    decoded.begin(),
    num_runs);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceRunLengthDecode::DecodeFromOffsets failed with status: " << error << '\n';
  }

  const thrust::device_vector<int> expected = {0, 2, 2, 5, 5, 5, 8};
  // example-end decode-run-offsets

  REQUIRE(error == cudaSuccess);
  REQUIRE(decoded == expected);
}
