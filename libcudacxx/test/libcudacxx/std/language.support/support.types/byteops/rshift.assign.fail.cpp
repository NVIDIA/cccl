//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cstddef>

#include <test_macros.h>

// template <class IntegerType>
//   constexpr byte operator>>(byte& b, IntegerType shift) noexcept;
// This function shall not participate in overload resolution unless
//   is_integral_v<IntegerType> is true.

__host__ __device__ constexpr cuda::std::byte test(cuda::std::byte b)
{
  return b >>= 2.0;
}

int main(int, char**)
{
  constexpr cuda::std::byte b1 = test(static_cast<cuda::std::byte>(1));

  return 0;
}
