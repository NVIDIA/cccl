//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function from a __host__ __device__ __tile__ function is not allowed

#include <cuda/std/cassert>
#include <cuda/std/charconv>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include "test_macros.h"

template <class T, cuda::std::size_t Size>
TEST_HOST_DEVICE_FUNC constexpr void test_out_of_range(const char (&input)[Size], int base)
{
  T value     = static_cast<T>(17);
  auto result = cuda::std::from_chars(input, input + Size - 1, value, base);
  assert(result.ptr == input + Size - 1);
  assert(result.ec == cuda::std::errc::result_out_of_range);
  assert(value == static_cast<T>(17));
}

TEST_HOST_DEVICE_FUNC constexpr bool test()
{
  // 755 wraps upward to 243 in uint8_t.
  test_out_of_range<cuda::std::uint8_t>("kz", 36);

  // 383 wraps back to 127 in int8_t.
  test_out_of_range<cuda::std::int8_t>("112012", 3);

  // The magnitude of -384 wraps to 128, which looks like a valid int8_t minimum.
  test_out_of_range<cuda::std::int8_t>("-112020", 3);
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
