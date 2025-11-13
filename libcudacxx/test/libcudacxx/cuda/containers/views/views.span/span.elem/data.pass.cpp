//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/span>

// pointer data() const noexcept;
//

#include <cuda/span>
#include <cuda/std/bit>
#include <cuda/std/cassert>

#include "test_macros.h"

template <typename Span>
__device__ void testRuntimeSpan(Span sp, typename Span::pointer ptr)
{
  static_assert(noexcept(sp.data()));
  assert(sp.data() == ptr);
}

struct A
{};

template <typename T, ::cuda::std::size_t Extent>
using fixed_span_t = cuda::device::shared_memory_span<T, Extent>;

template <typename T>
using dynamic_span_t = cuda::device::shared_memory_span<T, ::cuda::std::dynamic_extent>;

__global__ void test()
{
  constexpr int array[] = {10, 11, 12, 13, 14, 15, 16};
  constexpr int size    = cuda::std::size(array);
  __shared__ int array_smem[10];
  for (int i = 0; i < size; ++i)
  {
    array_smem[i] = array[i];
  }
  auto float_array = cuda::std::bit_cast<float*>(array_smem);
  // dynamic size
  // IMPORTANT: nullptr is not a valid pointer in shared memory, the following case will not work
  // testRuntimeSpan(dynamic_span_t<int>(), nullptr);

  testRuntimeSpan(dynamic_span_t<int>(array_smem, 1), array_smem);
  testRuntimeSpan(dynamic_span_t<int>(array_smem, 2), array_smem);
  testRuntimeSpan(dynamic_span_t<int>(array_smem, 3), array_smem);
  testRuntimeSpan(dynamic_span_t<int>(array_smem, 4), array_smem);

  testRuntimeSpan(dynamic_span_t<int>(array_smem + 1, 1), array_smem + 1);
  testRuntimeSpan(dynamic_span_t<int>(array_smem + 2, 2), array_smem + 2);
  testRuntimeSpan(dynamic_span_t<int>(array_smem + 3, 3), array_smem + 3);
  testRuntimeSpan(dynamic_span_t<int>(array_smem + 4, 4), array_smem + 4);

  testRuntimeSpan(dynamic_span_t<int>(float_array + 2, 2), float_array + 2);

  //  static size
  testRuntimeSpan(fixed_span_t<int, 1>(array_smem, 1), array_smem);
  testRuntimeSpan(fixed_span_t<int, 2>(array_smem, 2), array_smem);
  testRuntimeSpan(fixed_span_t<int, 3>(array_smem, 3), array_smem);
  testRuntimeSpan(fixed_span_t<int, 4>(array_smem, 4), array_smem);

  testRuntimeSpan(fixed_span_t<int, 1>(array_smem + 1, 1), array_smem + 1);
  testRuntimeSpan(fixed_span_t<int, 2>(array_smem + 2, 2), array_smem + 2);
  testRuntimeSpan(fixed_span_t<int, 3>(array_smem + 3, 3), array_smem + 3);
  testRuntimeSpan(fixed_span_t<int, 4>(array_smem + 4, 4), array_smem + 4);

  testRuntimeSpan(fixed_span_t<int, 2>(float_array + 2, 2), float_array + 2);
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test<<<1, 1>>>(); assert(cudaDeviceSynchronize() == cudaSuccess);))
  return 0;
}

