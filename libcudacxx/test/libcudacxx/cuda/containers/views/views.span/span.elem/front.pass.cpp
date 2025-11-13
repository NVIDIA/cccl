//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/span>

// reference front() const noexcept;
//   Expects: empty() is false.
//   Effects: Equivalent to: return *data();
//
#include <cuda/span>
#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "test_macros.h"

template <typename T, ::cuda::std::size_t Extent>
using fixed_span_t = cuda::device::shared_memory_span<T, Extent>;

template <typename T>
using dynamic_span_t = cuda::device::shared_memory_span<T, ::cuda::std::dynamic_extent>;

template <typename Span>
__device__ void test_front(Span sp, typename Span::pointer expected)
{
  static_assert(noexcept(sp.front()));
  static_assert(noexcept(cuda::std::as_const(sp).front()));

  auto& ref = sp.front();
  assert(&ref == expected);
  assert(ref == *expected);

  auto& ref2 = cuda::std::as_const(sp).front();
  assert(&ref2 == expected);
  assert(ref2 == *expected);
}

template <typename Span>
__device__ void test_empty_span(Span sp)
{
  static_assert(noexcept(sp.empty()));
  static_assert(noexcept(cuda::std::as_const(sp).empty()));
  static_assert(cuda::std::is_same_v<decltype(sp.empty()), bool>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::as_const(sp).empty()), bool>);
  if (!sp.empty())
  {
    static_cast<void>(sp.front());
  }
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
  for (int extent = 1; extent <= size; ++extent)
  {
    test_front(dynamic_span_t<int>(array_smem, extent), array_smem);
    test_front(dynamic_span_t<const int>(array_smem, extent), array_smem);
  }
  test_front(fixed_span_t<int, 1>(array_smem, 1), array_smem);
  test_front(fixed_span_t<const int, 1>(array_smem, 1), array_smem);

  test_front(fixed_span_t<int, 2>(array_smem, 2), array_smem);
  test_front(fixed_span_t<const int, 2>(array_smem, 2), array_smem);

  test_empty_span(dynamic_span_t<int>());
  test_empty_span(dynamic_span_t<const int>());
  test_empty_span(fixed_span_t<int, 0>());
  test_empty_span(fixed_span_t<const int, 0>());
  test_empty_span(dynamic_span_t<int>(array_smem, 0));
  test_empty_span(dynamic_span_t<const int>(array_smem, 0));
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test<<<1, 1>>>(); assert(cudaDeviceSynchronize() == cudaSuccess);))
  return 0;
}
