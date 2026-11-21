// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/execution>

#include <cuda/experimental/mgmn/__algorithm/sort/local.h>

#include <algorithm>
#include <vector>

#include <cuda_runtime_api.h>
#include <testing.cuh>

#include <c2h/catch2_test_helper.h>

namespace gated_merge_test
{
namespace impl = cuda::experimental::mgmn::__detail::__sort;

struct unchanged_tuning
{
  __host__ __device__ constexpr cub::MergePolicy operator()(cuda::compute_capability, cub::MergePolicy policy) const
  {
    return policy;
  }
};

template <class Compare>
void check(int run_count, cuda::std::int64_t size, Compare compare, bool duplicates, bool tuned, cudaStream_t stream)
{
  CAPTURE(run_count, size, duplicates, tuned);
  constexpr int guard = -123456789;
  std::vector<int> input(size + 4 * run_count + 3, guard), actual(size + 2, guard), expected;
  std::vector<impl::__run> runs;
  cuda::std::int64_t cursor = 3;
  for (int rank = 0; rank < run_count; ++rank)
  {
    const auto count = size / run_count + (rank < size % run_count);
    if (count == 0)
    {
      continue; // As in the sort, the merge receives only nonempty runs.
    }
    runs.push_back({cursor, count});
    for (cuda::std::int64_t i = 0; i < count; ++i)
    {
      input[cursor + i] = duplicates ? 7 : static_cast<int>(i * run_count + rank);
    }
    std::sort(input.begin() + cursor, input.begin() + cursor + count, compare);
    expected.insert(expected.end(), input.begin() + cursor, input.begin() + cursor + count);
    cursor += count + 1 + rank % 3;
  }
  std::sort(expected.begin(), expected.end(), compare);
  int* source;
  int* destination;
  REQUIRE(cudaMalloc(&source, input.size() * sizeof(int)) == cudaSuccess);
  REQUIRE(cudaMalloc(&destination, actual.size() * sizeof(int)) == cudaSuccess);
  REQUIRE(
    cudaMemcpyAsync(source, input.data(), input.size() * sizeof(int), cudaMemcpyHostToDevice, stream) == cudaSuccess);
  REQUIRE(cudaMemcpyAsync(destination, actual.data(), actual.size() * sizeof(int), cudaMemcpyHostToDevice, stream)
          == cudaSuccess);
  if (tuned)
  {
    const auto env = cuda::std::execution::env{cuda::stream_ref(stream), cuda::execution::tune(unchanged_tuning{})};
    impl::__merge_received(source, size, runs, destination + 1, compare, env);
  }
  else
  {
    const auto env = cuda::std::execution::env{
      cuda::stream_ref(stream), cuda::std::execution::prop{cuda::allocation_alignment, cuda::std::size_t{256}}};
    impl::__merge_received(source, size, runs, destination + 1, compare, env);
  }
  REQUIRE(cudaMemcpyAsync(actual.data(), destination, actual.size() * sizeof(int), cudaMemcpyDeviceToHost, stream)
          == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
  REQUIRE(actual.front() == guard);
  REQUIRE(actual.back() == guard);
  REQUIRE(std::equal(expected.begin(), expected.end(), actual.begin() + 1));
  // The received buffer is private workspace. The CUB fallback may reuse it
  // between merge rounds; only the caller's output and its guards are fixed.
  REQUIRE(cudaFree(destination) == cudaSuccess);
  REQUIRE(cudaFree(source) == cudaSuccess);
}
} // namespace gated_merge_test

TEST_CASE("sort merge, size and fan-in gate boundaries", "[sort]")
{
  using gated_merge_test::impl::__small_tiled_merge_size_supported;
  STATIC_REQUIRE(__small_tiled_merge_size_supported(3, 3 * 2097152));
  STATIC_REQUIRE_FALSE(__small_tiled_merge_size_supported(4, 16777216));
  STATIC_REQUIRE(__small_tiled_merge_size_supported(32, 16777216));
  for (int runs = 0; runs <= 64; ++runs)
  {
    CAPTURE(runs);
    CHECK_FALSE(__small_tiled_merge_size_supported(runs, -1));
    CHECK_FALSE(__small_tiled_merge_size_supported(runs, 0));
    if (runs >= 3 && runs <= 32)
    {
      const auto limit = std::min<cuda::std::int64_t>(16777216, cuda::std::int64_t{runs} * 2097152);
      CHECK(__small_tiled_merge_size_supported(runs, limit - 1));
      CHECK(__small_tiled_merge_size_supported(runs, limit));
      CHECK_FALSE(__small_tiled_merge_size_supported(runs, limit + 1));
    }
    else
    {
      CHECK_FALSE(__small_tiled_merge_size_supported(runs, 1));
    }
  }
}

TEST_CASE("sort merge, guarded tiles, fallback, and nondefault device", "[sort]")
{
  int device_count, original_device;
  REQUIRE(cudaGetDeviceCount(&device_count) == cudaSuccess);
  REQUIRE(cudaGetDevice(&original_device) == cudaSuccess);
  REQUIRE(cudaSetDevice(device_count > 1 ? 1 : 0) == cudaSuccess);
  cudaStream_t stream;
  REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  for (int runs : {0, 1, 2, 3, 7, 8, 16, 31, 32, 33})
  {
    for (int size : {0, 1, 257, 8703, 8704, 8705, 65537})
    {
      if (runs == 0 && size != 0)
      {
        continue;
      }
      for (bool duplicates : {false, true})
      {
        gated_merge_test::check(runs, size, cuda::std::less<int>{}, duplicates, false, stream);
        gated_merge_test::check(runs, size, cuda::std::greater<>{}, duplicates, false, stream);
      }
    }
  }
  // Exercise both sides of the average-fragment and total-output limits on
  // real buffers; the host test above covers the complete policy truth table.
  for (int runs : {3, 32})
  {
    const auto limit = std::min<cuda::std::int64_t>(16777216, cuda::std::int64_t{runs} * 2097152);
    for (auto size : {limit - 1, limit, limit + 1})
    {
      gated_merge_test::check(runs, size, cuda::std::less<int>{}, false, false, stream);
      gated_merge_test::check(runs, size, cuda::std::greater<>{}, false, false, stream);
    }
  }
  // An explicit CUB tuning query must continue to select the CUB merge path.
  gated_merge_test::check(7, 4097, cuda::std::less<int>{}, false, true, stream);
  gated_merge_test::check(7, 4097, cuda::std::greater<>{}, true, true, stream);
  REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
  REQUIRE(cudaSetDevice(original_device) == cudaSuccess);
}
