// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// %PARAM% TEST_ERR err 0:1:2

#include <cub/thread/thread_reduce.cuh>

#include <cuda/std/array>
#include <cuda/std/functional>
#include <cuda/std/mdspan>
#include <cuda/std/span>

// Verifies that cub::ThreadReduce without a prefix rejects zero-length inputs at compile time, since there is no
// element to return. The overload taking a prefix supports empty inputs.

__global__ void kernel(int* d_out)
{
  int data[1] = {};
#if TEST_ERR == 0
  const cuda::std::span<int, 0> input(data, 0);
  // expected-error-0 {{"ThreadReduce without a prefix requires a non-empty input"}}
#elif TEST_ERR == 1
  const cuda::std::mdspan<int, cuda::std::extents<int, 0>> input(data);
  // expected-error-1 {{"ThreadReduce without a prefix requires a non-empty input"}}
#elif TEST_ERR == 2
  const cuda::std::array<int, 0> input{};
  // expected-error-2 {{"ThreadReduce without a prefix requires a non-empty input"}}
#endif
  *d_out = cub::ThreadReduce(input, cuda::std::plus<>{});
}

int main()
{
  kernel<<<1, 1>>>(nullptr);
}
