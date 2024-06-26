//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-no-exceptions

// bitset<N>& set(size_t pos, bool val = true); // constexpr since C++23

// Make sure we throw ::std::out_of_range when calling set() on an OOB index.

#include <cuda/std/bitset>
#include <cuda/std/cassert>

int main(int, char**)
{
  NV_IF_TARGET(
    NV_IS_HOST,
    {
      cuda::std::bitset<0> v;
      try
      {
        v.set(0);
        assert(false);
      }
      catch (::std::out_of_range const&)
      {}
    } {
      cuda::std::bitset<1> v("0");
      try
      {
        v.set(2);
        assert(false);
      }
      catch (::std::out_of_range const&)
      {}
    } {
      cuda::std::bitset<10> v("0000000000");
      try
      {
        v.set(10);
        assert(false);
      }
      catch (::std::out_of_range const&)
      {}
    })

  return 0;
}
