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

template <class Value, class Error>
__host__ __device__ void
test(cuda::std::expected<Value, Error> with_value, cuda::std::expected<Value, Error> with_error, size_t expected_size)
{
  assert(with_value.value() == 42);
  assert(with_error.error() == 1337);
  assert(sizeof(cuda::std::expected<Value, Error>) == expected_size);
}

template <class Value, class Error>
__global__ void test_kernel(
  cuda::std::expected<Value, Error> with_value, cuda::std::expected<Value, Error> with_error, size_t expected_size)
{
  test(with_value, with_error, expected_size);
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
    using expect = cuda::std::expected<int, int>;
    expect with_value{cuda::std::in_place, 42};
    expect with_error{cuda::std::unexpect, 1337};
    test(with_value, with_error, sizeof(expect));
    test_kernel<<<1, 1>>>(with_value, with_error, sizeof(expect));
  }

  { // empty payload, non-empty error
    using expect = cuda::std::expected<empty<42>, int>;
    expect with_value{cuda::std::in_place, 42};
    expect with_error{cuda::std::unexpect, 1337};
    test(with_value, with_error, sizeof(expect));
    test_kernel<<<1, 1>>>(with_value, with_error, sizeof(expect));
  }

  { // non-empty payload, empty error
    using expect = cuda::std::expected<int, empty<1337>>;
    expect with_value{cuda::std::in_place, 42};
    expect with_error{cuda::std::unexpect, 1337};
    test(with_value, with_error, sizeof(expect));
    test_kernel<<<1, 1>>>(with_value, with_error, sizeof(expect));
  }

  { // empty payload, non-empty error
    using expect = cuda::std::expected<empty<42>, empty<1337>>;
    expect with_value{cuda::std::in_place, 42};
    expect with_error{cuda::std::unexpect, 1337};
    test(with_value, with_error, sizeof(expect));
    test_kernel<<<1, 1>>>(with_value, with_error, sizeof(cuda::std::expected<empty<42>, empty<1337>>));
  }
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
