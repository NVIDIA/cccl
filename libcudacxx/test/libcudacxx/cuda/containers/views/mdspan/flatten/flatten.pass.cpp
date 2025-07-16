//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/mdspan>
#include <cuda/std/complex>

#include "test_macros.h"

template <typename MDSpan, typename T>
__host__ __device__ void test_basic(MDSpan md, T value)
{
  for (auto&& v : cuda::flatten(md))
  {
    assert(v == value);
  }

  // The 2-step is needed here because we want to make sure that each iteration accesses a
  // unique element. If these ever overlap, then v = v + N * one where N is the number of
  // overlaps.
  const auto one = T{1};

  for (auto&& v : cuda::flatten(md))
  {
    v += one;
  }

  for (auto&& v : cuda::flatten(md))
  {
    assert(v == value + one);
  }

  for (auto&& v : cuda::flatten(md))
  {
    v -= one;
  }

  for (auto&& v : cuda::flatten(md))
  {
    assert(v == value);
  }
}

template <typename MDSpan>
__host__ __device__ void test_iter(MDSpan md)
{
  const auto size = md.size();

  {
    auto i    = std::size_t{0};
    auto view = cuda::flatten(md);

    for (auto it = view.begin(); it != view.end(); ++it)
    {
      ++i;
    }
    assert(i == size);
  }

  {
    auto i    = std::size_t{0};
    auto view = cuda::flatten(md);

    for (auto&& _ : view)
    {
      ++i;
    }
    assert(i == size);
  }

  {
    auto i = std::size_t{0};

    for (auto&& _ : cuda::flatten(md))
    {
      ++i;
    }
    assert(i == size);
  }
}

template <typename T>
__host__ __device__ void test_body()
{
  {
    auto md = cuda::std::mdspan<T>{};

    test_iter(md);
    test_basic(md, T{});
  }
  {
    auto data = T{42};
    auto md   = cuda::std::mdspan{&data};

    test_iter(md);
    test_basic(md, data);
  }
}

template <typename T>
__global__ void test_kernel()
{
  test_body<T>();
}

template <typename T>
void test_dispatch()
{
  test_body<T>();
  test_kernel<T><<<1, 1>>>();
}

void test()
{
  test_dispatch<int>();
  test_disptach<uint64_t>();
  test_dispatch<cuda::std::complex<double>>();
}

int main()
{
  NV_IF_TARGET(NV_IS_HOST, (test();));
}
