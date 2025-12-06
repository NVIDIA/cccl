//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cassert>
#include <cuda/std/charconv>
#include <cuda/std/type_traits>

__host__ __device__ constexpr bool test()
{
  using cf = cuda::std::chars_format;
  using ut = cuda::std::underlying_type<cf>::type;

  {
    cf x = cf::scientific;
    x |= cf::fixed;
    assert(x == cf::general);
  }
  {
    cf x = cf::general;
    x &= cf::fixed;
    assert(x == cf::fixed);
  }
  {
    cf x = cf::general;
    x ^= cf::fixed;
    assert(x == cf::scientific);
  }

  assert(static_cast<ut>(cf::scientific & (cf::fixed | cf::hex)) == 0);
  assert(static_cast<ut>(cf::fixed & (cf::scientific | cf::hex)) == 0);
  assert(static_cast<ut>(cf::hex & (cf::scientific | cf::fixed)) == 0);

  assert((cf::scientific | cf::fixed) == cf::general);

  assert(static_cast<ut>(cf::scientific & cf::fixed) == 0);

  assert((cf::general ^ cf::fixed) == cf::scientific);

  assert((~cf::hex & cf::general) == cf::general);

  {
    [[maybe_unused]] cuda::std::chars_format x{};
    static_assert(cuda::std::is_same_v<cuda::std::chars_format, decltype(~x)>);
    static_assert(cuda::std::is_same_v<cuda::std::chars_format, decltype(x & x)>);
    static_assert(cuda::std::is_same_v<cuda::std::chars_format, decltype(x | x)>);
    static_assert(cuda::std::is_same_v<cuda::std::chars_format, decltype(x ^ x)>);
    static_assert(cuda::std::is_same_v<cuda::std::chars_format&, decltype(x &= x)>);
    static_assert(cuda::std::is_same_v<cuda::std::chars_format&, decltype(x |= x)>);
    static_assert(cuda::std::is_same_v<cuda::std::chars_format&, decltype(x ^= x)>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
