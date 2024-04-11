//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: c++98 || c++03

// <utility>

// template <class T1, class T2> struct pair

// pair& operator=(pair const& p);

#include <cuda/std/utility>
// cuda/std/memory not supported
// #include <cuda/std/memory>
#include <cuda/std/cassert>

#include "test_macros.h"

struct NonAssignable
{
  __host__ __device__ NonAssignable() {}

private:
  __host__ __device__ NonAssignable& operator=(NonAssignable const&);
};

struct Incomplete;
extern Incomplete inc_obj;

int main(int, char**)
{
  {
    // Test that we don't constrain the assignment operator in C++03 mode.
    // Since we don't have access control SFINAE having pair evaluate SFINAE
    // may cause a hard error.
    typedef cuda::std::pair<int, NonAssignable> P;
    static_assert(cuda::std::is_copy_assignable<P>::value, "");
  }
  {
    typedef cuda::std::pair<int, Incomplete&> P;
    static_assert(cuda::std::is_copy_assignable<P>::value, "");
    P p(42, inc_obj);
    assert(&p.second == &inc_obj);
  }

  return 0;
}

struct Incomplete
{};
Incomplete inc_obj;
