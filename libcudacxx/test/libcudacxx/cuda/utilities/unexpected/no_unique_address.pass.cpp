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
#include <cuda/std/expected>

#include "test_macros.h"

template <class Error>
__host__ __device__ void test(cuda::std::unexpected<Error> with_error, size_t expected_size)
{
  assert(with_error.error() == 1337);
  assert(sizeof(cuda::std::unexpected<Error>) == expected_size);
}

template <class Error>
__global__ void test_kernel(cuda::std::unexpected<Error> with_error, size_t expected_size)
{
  test(with_error, expected_size);
}

template <int Expected>
struct empty
{
  constexpr empty() = default;
  __host__ __device__ constexpr empty(const int val) noexcept
  {
    assert(val == Expected);
  }

  __host__ __device__ friend constexpr bool operator==(const empty&, int val)
  {
    return val == Expected;
  }
};

void test()
{
  { // non-empty payload, non-empty error
    using unexpect = cuda::std::unexpected<int>;
    unexpect with_error{cuda::std::in_place, 1337};
    test(with_error, sizeof(unexpect));
    test_kernel<<<1, 1>>>(with_error, sizeof(unexpect));
  }

  { // non-empty payload, empty error
    using unexpect = cuda::std::unexpected<empty<1337>>;
    unexpect with_error{cuda::std::in_place, 1337};
    test(with_error, sizeof(unexpect));
    test_kernel<<<1, 1>>>(with_error, sizeof(unexpect));
  }
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
