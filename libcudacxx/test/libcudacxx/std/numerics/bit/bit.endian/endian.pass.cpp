//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// enum class endian;
// <cuda/std/bit>

#include <cuda/std/bit>
// #include <cuda/std/cstring>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

#include "test_macros.h"

int main(int, char**)
{
  static_assert(cuda::std::is_enum<cuda::std::endian>::value, "");

  // Check that E is a scoped enum by checking for conversions.
  typedef cuda::std::underlying_type<cuda::std::endian>::type UT;
  static_assert(!cuda::std::is_convertible<cuda::std::endian, UT>::value, "");

  // test that the enumeration values exist
  static_assert(cuda::std::endian::little == cuda::std::endian::little, "");
  static_assert(cuda::std::endian::big == cuda::std::endian::big, "");
  static_assert(cuda::std::endian::native == cuda::std::endian::native, "");
  static_assert(cuda::std::endian::little != cuda::std::endian::big, "");

  //  Technically not required, but true on all existing machines
  static_assert(
    cuda::std::endian::native == cuda::std::endian::little || cuda::std::endian::native == cuda::std::endian::big, "");

  //  Try to check at runtime
  {
    uint32_t i = 0x01020304;
    char c[4];
    static_assert(sizeof(i) == sizeof(c), "");
    memcpy(c, &i, sizeof(c));

    assert((c[0] == 1) == (cuda::std::endian::native == cuda::std::endian::big));
  }

  return 0;
}
