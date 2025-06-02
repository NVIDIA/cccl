//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc
// UNSUPPORTED: msvc
// See nvbug5272086

#include <cuda/std/cassert>
#include <cuda/std/mdspan>

#include "test_macros.h"

template <class Mdspan>
__host__ __device__ void test(Mdspan md, size_t expected_size, int* data_handle)
{
  assert(md.data_handle() == data_handle);
  assert(md.extents().extent(0) == 42);
  assert(md.extents().extent(1) == 1337);
  assert(md.extents().extent(2) == 7);
  assert(sizeof(Mdspan) == expected_size);
}

template <class Mdspan>
__global__ void test_kernel(Mdspan md, size_t expected_size, int* data_handle)
{
  test(md, expected_size, data_handle);
}

template <template <class> class Mapping>
void test()
{
  int data = 42;
  { // all dynamic
    using extents =
      cuda::std::extents<size_t, cuda::std::dynamic_extent, cuda::std::dynamic_extent, cuda::std::dynamic_extent>;
    cuda::std::mdspan mdspan{&data, Mapping<extents>{extents{42, 1337, 7}}};
    test(mdspan, sizeof(decltype(mdspan)), &data);
    test_kernel<<<1, 1>>>(mdspan, sizeof(decltype(mdspan)), &data);
  }

  { // middle static
    using extents = cuda::std::extents<size_t, cuda::std::dynamic_extent, 1337, cuda::std::dynamic_extent>;
    cuda::std::mdspan mdspan{&data, Mapping<extents>{extents{42, 7}}};
    test(mdspan, sizeof(decltype(mdspan)), &data);
    test_kernel<<<1, 1>>>(mdspan, sizeof(decltype(mdspan)), &data);
  }

  { // middle dynamic
    using extents = cuda::std::extents<size_t, 42, cuda::std::dynamic_extent, 7>;
    cuda::std::mdspan mdspan{&data, Mapping<extents>{extents{1337}}};
    test(mdspan, sizeof(decltype(mdspan)), &data);
    test_kernel<<<1, 1>>>(mdspan, sizeof(decltype(mdspan)), &data);
  }

  { // all static
    using extents = cuda::std::extents<size_t, 42, 1337, 7>;
    cuda::std::mdspan mdspan{&data, Mapping<extents>{extents{}}};
    test(mdspan, sizeof(decltype(mdspan)), &data);
    test_kernel<<<1, 1>>>(mdspan, sizeof(decltype(mdspan)), &data);
  }
}

void test()
{
  test<cuda::std::layout_left::mapping>();
  test<cuda::std::layout_right::mapping>();
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
