// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstddef>

#include <unittest/unittest.h>

namespace
{
struct is_even
{
  bool operator()(int x) const
  {
    return x % 2 == 0;
  }
};
} // namespace

TEST_CASE("TestOmpCopyIfInsideParallelRegion", "[copy_if]")
{
  // Large enough that copy_if splits the work between threads when it is called outside of a parallel region
  const int n = 1 << 20;

  const thrust::host_vector<int> h_input   = unittest::random_integers<int>(n);
  const thrust::device_vector<int> d_input = h_input;

  thrust::host_vector<int> h_result(n, thrust::no_init);
  h_result.resize(thrust::copy_if(h_input.begin(), h_input.end(), h_result.begin(), is_even{}) - h_result.begin());

  thrust::device_vector<int> d_results[2] = {
    thrust::device_vector<int>(n, thrust::no_init), thrust::device_vector<int>(n, thrust::no_init)};
  std::ptrdiff_t d_result_sizes[2] = {};

  // Nested parallelism is disabled by default, so the parallel regions started by copy_if get a single thread each,
  // although omp_get_max_threads() still reports the full number of threads
#pragma omp parallel for num_threads(2)
  for (int i = 0; i < 2; ++i)
  {
    d_result_sizes[i] =
      thrust::copy_if(d_input.begin(), d_input.end(), d_results[i].begin(), is_even{}) - d_results[i].begin();
  }

  for (int i = 0; i < 2; ++i)
  {
    d_results[i].resize(d_result_sizes[i]);
    REQUIRE(d_results[i] == h_result);
  }
}
