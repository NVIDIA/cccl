//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// UNSUPPORTED: nvrtc

// bool operator==(array<T, N> const&, array<T, N> const&);
// bool operator!=(array<T, N> const&, array<T, N> const&);
// bool operator<(array<T, N> const&, array<T, N> const&);
// bool operator<=(array<T, N> const&, array<T, N> const&);
// bool operator>(array<T, N> const&, array<T, N> const&);
// bool operator>=(array<T, N> const&, array<T, N> const&);


#include <cuda/std/array>
#include <cuda/std/vector>
#include <cuda/std/cassert>

#include "test_macros.h"

// cuda::std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "disable_missing_braces_warning.h"

template <class Array>
__host__ __device__
void test_compare(const Array& LHS, const Array& RHS) {
  typedef cuda::std::vector<typename Array::value_type> Vector;
  const Vector LHSV(LHS.begin(), LHS.end());
  const Vector RHSV(RHS.begin(), RHS.end());
  assert((LHS == RHS) == (LHSV == RHSV));
  assert((LHS != RHS) == (LHSV != RHSV));
  assert((LHS < RHS) == (LHSV < RHSV));
  assert((LHS <= RHS) == (LHSV <= RHSV));
  assert((LHS > RHS) == (LHSV > RHSV));
  assert((LHS >= RHS) == (LHSV >= RHSV));
}

template <int Dummy> struct NoCompare {};

int main(int, char**)
{
  {
    typedef NoCompare<0> T;
    typedef cuda::std::array<T, 3> C;
    C c1 = {{}};
    // expected-error@algorithm:* 2 {{invalid operands to binary expression}}
    TEST_IGNORE_NODISCARD (c1 == c1);
    TEST_IGNORE_NODISCARD (c1 < c1);
  }
  {
    typedef NoCompare<1> T;
    typedef cuda::std::array<T, 3> C;
    C c1 = {{}};
    // expected-error@algorithm:* 2 {{invalid operands to binary expression}}
    TEST_IGNORE_NODISCARD (c1 != c1);
    TEST_IGNORE_NODISCARD (c1 > c1);
  }
  {
    typedef NoCompare<2> T;
    typedef cuda::std::array<T, 0> C;
    C c1 = {{}};
    // expected-error@algorithm:* 2 {{invalid operands to binary expression}}
    TEST_IGNORE_NODISCARD (c1 == c1);
    TEST_IGNORE_NODISCARD (c1 < c1);
  }

  return 0;
}
