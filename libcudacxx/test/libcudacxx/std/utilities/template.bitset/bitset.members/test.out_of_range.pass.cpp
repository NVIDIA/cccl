//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-no-exceptions

// constexpr bool test(size_t pos) const;

// Make sure we throw cuda::std::out_of_range when calling test() on an OOB index.

#include <cuda/std/bitset>
#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(
    NV_IS_HOST,
    {
      cuda::std::bitset<0> v;
      try
      {
        (void) v.test(0);
        assert(false);
      }
      catch (::std::out_of_range const&)
      {}
    } {
      cuda::std::bitset<1> v("0");
      try
      {
        (void) v.test(2);
        assert(false);
      }
      catch (::std::out_of_range const&)
      {}
    } {
      cuda::std::bitset<10> v("0000000000");
      try
      {
        (void) v.test(10);
        assert(false);
      }
      catch (::std::out_of_range const&)
      {}
    })
#endif // TEST_HAS_EXCEPTIONS()

  return 0;
}
