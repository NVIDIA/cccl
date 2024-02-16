//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// explicit tuple(const T&...);

// UNSUPPORTED: c++98, c++03


#include <cuda/std/tuple>
#include <cuda/std/cassert>

struct ExplicitCopy {
  TEST_HOST_DEVICE ExplicitCopy(int) {}
  TEST_HOST_DEVICE explicit ExplicitCopy(ExplicitCopy const&) {}
};

TEST_HOST_DEVICE std::tuple<ExplicitCopy> const_explicit_copy() {
    const ExplicitCopy e(42);
    return {e};
    // expected-error@-1 {{chosen constructor is explicit in copy-initialization}}
}


TEST_HOST_DEVICE std::tuple<ExplicitCopy> non_const_explicit_copy() {
    ExplicitCopy e(42);
    return {e};
    // expected-error@-1 {{chosen constructor is explicit in copy-initialization}}
}

TEST_HOST_DEVICE std::tuple<ExplicitCopy> const_explicit_copy_no_brace() {
    const ExplicitCopy e(42);
    return e;
    // expected-error@-1 {{no viable conversion}}
}

int main(int, char**)
{

  return 0;
}
