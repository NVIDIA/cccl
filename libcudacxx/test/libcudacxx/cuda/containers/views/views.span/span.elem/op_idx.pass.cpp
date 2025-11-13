//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <dynamic_span_t>

// reference operator[](size_type idx) const;
//

#include <cuda/span>
#include <cuda/std/cassert>
#include <cuda/std/utility>
#include <cuda/utility>

#include "test_macros.h"

template <typename T, typename Span>
__host__ __device__ void test_acces_operator(Span sp, size_t idx, T expected)
{
  static_assert(cuda::std::is_same_v<typename Span::reference, T&>);
  {
    static_assert(cuda::std::is_same_v<decltype(sp[idx]), T&>);
    static_assert(cuda::std::is_same_v<decltype(sp.at(idx)), T&>);
    static_assert(noexcept(sp[idx]));
    static_assert(!noexcept(sp.at(idx)));
    assert(sp[idx] == expected);
  }
  {
    static_assert(cuda::std::is_same_v<decltype(cuda::std::as_const(sp)[idx]), T&>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::as_const(sp).at(idx)), T&>);
    static_assert(noexcept(cuda::std::as_const(sp)[idx]));
    static_assert(!noexcept(cuda::std::as_const(sp).at(idx)));
    assert(cuda::std::as_const(sp)[idx] == expected);
  }
}

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
  for (int i = 1; i <= size; ++i)
  {
    for (int j = 0; j < i; ++j)
    {
      test_acces_operator(dynamic_span_t<int>(array_smem, i), j, array[j]);
      test_acces_operator<const int>(dynamic_span_t<const int>(array_smem, i), j, array[j]);
    }
  }
  test_acces_operator(fixed_span_t<int, 1>(array_smem, 1), 0, array[0]);
  test_acces_operator<const int>(fixed_span_t<const int, 1>(array_smem, 1), 0, array[0]);

  for (int j = 0; j < 3; ++j)
  {
    test_acces_operator(fixed_span_t<int, 3>(array_smem, 3), j, array[j]);
    test_acces_operator<const int>(fixed_span_t<const int, 3>(array_smem, 3), j, array[j]);
  }
  for (int j = 0; j < size; ++j)
  {
    test_acces_operator(fixed_span_t<int, size>(array_smem, size), j, array[j]);
    test_acces_operator<const int>(fixed_span_t<const int, size>(array_smem, size), j, array[j]);
  }
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test<<<1, 1>>>(); assert(cudaDeviceSynchronize() == cudaSuccess);))
  return 0;
}
