//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The CI "Apple back-deployment with assertions enabled" needs a higher value
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=12712420

// bitset<N> operator<<(size_t pos) const; // constexpr since C++23

#include <cuda/std/bitset>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>


#include "../bitset_test_cases.h"
#include "test_macros.h"

template <cuda::std::size_t N>
__host__ __device__
TEST_CONSTEXPR_CXX23 void test_left_shift() {
    span_stub<const char *> const cases = get_test_cases<N>();
    for (cuda::std::size_t c = 0; c != cases.size(); ++c) {
        for (cuda::std::size_t s = 0; s <= N+1; ++s) {
            cuda::std::bitset<N> v1(cases[c]);
            cuda::std::bitset<N> v2 = v1;
            assert((v1 <<= s) == (v2 << s));
        }
    }
}

__host__ __device__
TEST_CONSTEXPR_CXX23 bool test() {
  test_left_shift<0>();
  test_left_shift<1>();
  test_left_shift<31>();
  test_left_shift<32>();
  test_left_shift<33>();
  test_left_shift<63>();
  test_left_shift<64>();
  test_left_shift<65>();

  return true;
}

int main(int, char**) {
  test();
  test_left_shift<1000>(); // not in constexpr because of constexpr evaluation step limits
#if TEST_STD_VER > 2020
  static_assert(test());
#endif

  return 0;
}
