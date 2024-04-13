//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// integral_constant

#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  typedef cuda::std::integral_constant<int, 5> _5;
  static_assert(_5::value == 5, "");
  static_assert((cuda::std::is_same<_5::value_type, int>::value), "");
  static_assert((cuda::std::is_same<_5::type, _5>::value), "");
  static_assert((_5() == 5), "");
  assert(_5() == 5);

#if TEST_STD_VER > 2011
  static_assert(_5{}() == 5, "");
  static_assert(cuda::std::true_type{}(), "");
#endif

  static_assert(cuda::std::false_type::value == false, "");
  static_assert((cuda::std::is_same<cuda::std::false_type::value_type, bool>::value), "");
  static_assert((cuda::std::is_same<cuda::std::false_type::type, cuda::std::false_type>::value), "");

  static_assert(cuda::std::true_type::value == true, "");
  static_assert((cuda::std::is_same<cuda::std::true_type::value_type, bool>::value), "");
  static_assert((cuda::std::is_same<cuda::std::true_type::type, cuda::std::true_type>::value), "");

  cuda::std::false_type f1;
  cuda::std::false_type f2 = f1;
  assert(!f2);

  cuda::std::true_type t1;
  cuda::std::true_type t2 = t1;
  assert(t2);

  return 0;
}
