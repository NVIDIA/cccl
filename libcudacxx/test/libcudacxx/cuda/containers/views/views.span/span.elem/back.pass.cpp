//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/span>

// constexpr reference back() const noexcept;
//   Expects: empty() is false.
//   Effects: Equivalent to: return *(data() + (size() - 1));
//
#include <cuda/span>
#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "test_macros.h"

template <typename T, ::cuda::std::size_t Extent>
using fixed_span_t = cuda::device::shared_memory_span<T, Extent>;

template <typename T>
using dynamic_span_t = cuda::device::shared_memory_span<T, ::cuda::std::dynamic_extent>;

template <typename T, typename Span>
__device__ void test_back(Span sp, T expected)
{
  static_assert(noexcept(sp.back()));
  static_assert(noexcept(cuda::std::as_const(sp).back()));
  static_assert(cuda::std::is_same_v<decltype(sp.back()), T&>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::as_const(sp).back()), T&>);
  assert(sp.size() > 0);

  auto& ref = sp.back();
  assert(ref == expected);

  auto& ref2 = cuda::std::as_const(sp).back();
  assert(ref2 == expected);
}

__global__ void test()
{
  constexpr int array[] = {10, 11, 12, 13, 14, 15, 16, 17};
  constexpr int size    = sizeof(array) / sizeof(array[0]);
  __shared__ int array_smem[size];
  for (int i = 0; i < size; ++i)
  {
    array_smem[i] = array[i];
  }
  for (int i = 1; i < size; ++i)
  {
    test_back(dynamic_span_t<int>(array_smem, i), array_smem[i - 1]);
    test_back<const int>(dynamic_span_t<const int>(array_smem, i), array_smem[i - 1]);
  }
  test_back(fixed_span_t<int, 1>(array_smem, 1), array_smem[0]);
  test_back<const int>(fixed_span_t<const int, 1>(array_smem, 1), array_smem[0]);

  test_back(fixed_span_t<int, 3>(array_smem, 3), array_smem[2]);
  test_back<const int>(fixed_span_t<const int, 3>(array_smem, 3), array_smem[2]);
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test<<<1, 1>>>(); assert(cudaDeviceSynchronize() == cudaSuccess);))
  return 0;
}
