//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cstddef>
#include <test_macros.h>

// UNSUPPORTED: c++98, c++03, c++11, c++14

// template <class IntegerType>
//   constexpr byte& operator<<=(byte& b, IntegerType shift) noexcept;
// This function shall not participate in overload resolution unless
//   is_integral_v<IntegerType> is true.


__host__ __device__
constexpr cuda::std::byte test(cuda::std::byte b) {
    return b <<= 2;
    }


int main(int, char**) {
    cuda::std::byte b;  // not constexpr, just used in noexcept check
    constexpr cuda::std::byte b2{static_cast<cuda::std::byte>(2)};
    constexpr cuda::std::byte b3{static_cast<cuda::std::byte>(3)};

    static_assert(noexcept(b <<= 2), "" );

    static_assert(cuda::std::to_integer<int>(test(b2)) ==  8, "" );
    static_assert(cuda::std::to_integer<int>(test(b3)) == 12, "" );


  return 0;
}
