// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/experimental/mgmn/sort>

#include <algorithm>
#include <vector>

#include <testing.cuh>

#include <c2h/catch2_test_helper.h>

#if defined(_CUDAX_SORT_HAS_NCCL_DEVICE)
namespace sort_copy_test
{
constexpr int stride = 8256;

__global__ void copy_bytes(char* destination, const char* source, int count)
{
  const int source_offset      = blockIdx.x / 16;
  const int destination_offset = blockIdx.x % 16;
  cudax::mgmn::__detail::__sort::__copy_bytes(
    destination + blockIdx.x * stride + 16 + destination_offset,
    source + blockIdx.x * stride + 16 + source_offset,
    count);
}
} // namespace sort_copy_test

TEST_CASE("sort copy, all alignments and guarded tails", "[sort]")
{
  constexpr int bytes = 256 * sort_copy_test::stride;
  std::vector<char> source(bytes), expected(bytes), actual(bytes);
  for (int index = 0; index < bytes; ++index)
  {
    source[index] = static_cast<char>(index % 127);
  }
  char* device_source;
  char* device_destination;
  REQUIRE(cudaMalloc(&device_source, bytes) == cudaSuccess);
  REQUIRE(cudaMalloc(&device_destination, bytes) == cudaSuccess);
  REQUIRE(cudaMemcpy(device_source, source.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess);
  std::vector<int> counts;
  for (int count = 0; count <= 33; ++count)
  {
    counts.push_back(count);
  }
  for (int count : {255, 256, 257, 4095, 4096, 4097, 8191, 8192, 8193})
  {
    counts.push_back(count);
  }
  for (int count : counts)
  {
    CAPTURE(count);
    std::fill(expected.begin(), expected.end(), static_cast<char>(0xff));
    for (int block = 0; block < 256; ++block)
    {
      std::copy_n(source.begin() + block * sort_copy_test::stride + 16 + block / 16,
                  count,
                  expected.begin() + block * sort_copy_test::stride + 16 + block % 16);
    }
    REQUIRE(cudaMemset(device_destination, 0xff, bytes) == cudaSuccess);
    sort_copy_test::copy_bytes<<<256, 256>>>(device_destination, device_source, count);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaMemcpy(actual.data(), device_destination, bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(actual == expected);
  }
  REQUIRE(cudaMemcpy(actual.data(), device_source, bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
  REQUIRE(actual == source);
  REQUIRE(cudaFree(device_destination) == cudaSuccess);
  REQUIRE(cudaFree(device_source) == cudaSuccess);
}
#endif // _CUDAX_SORT_HAS_NCCL_DEVICE
