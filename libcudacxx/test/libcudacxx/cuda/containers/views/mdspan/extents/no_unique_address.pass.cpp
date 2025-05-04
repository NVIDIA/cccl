//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

#include <cuda/std/cassert>
#include <cuda/std/mdspan>

#include "test_macros.h"

template <size_t First, size_t Second, size_t Third>
__host__ __device__ void test(cuda::std::extents<size_t, First, Second, Third> ext, size_t expected_size)
{
  using extents = cuda::std::extents<size_t, First, Second, Third>;
  assert(ext.extent(0) == 42);
  assert(ext.extent(1) == 1337);
  assert(ext.extent(2) == 7);
  assert(sizeof(extents) == expected_size);
}

template <size_t First, size_t Second, size_t Third>
__global__ void test_kernel(cuda::std::extents<size_t, First, Second, Third> ext, size_t expected_size)
{
  test(ext, expected_size);
}

void test()
{
  { // all dynamic
    using extents =
      cuda::std::extents<size_t, cuda::std::dynamic_extent, cuda::std::dynamic_extent, cuda::std::dynamic_extent>;
    extents ext{42, 1337, 7};
    test(ext, sizeof(extents));
    test_kernel<<<1, 1>>>(ext, sizeof(extents));
  }

  { // middle static
    using extents = cuda::std::extents<size_t, cuda::std::dynamic_extent, 1337, cuda::std::dynamic_extent>;
    extents ext{42, 7};
    test(ext, sizeof(extents));
    test_kernel<<<1, 1>>>(ext, sizeof(extents));
  }

  { // middle dynamic
    using extents = cuda::std::extents<size_t, 42, cuda::std::dynamic_extent, 7>;
    extents ext{1337};
    test(ext, sizeof(extents));
    test_kernel<<<1, 1>>>(ext, sizeof(extents));
  }

  { // all dynamic
    using extents = cuda::std::extents<size_t, 42, 1337, 7>;
    extents ext{};
    test(ext, sizeof(extents));
    test_kernel<<<1, 1>>>(ext, sizeof(extents));
  }
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
