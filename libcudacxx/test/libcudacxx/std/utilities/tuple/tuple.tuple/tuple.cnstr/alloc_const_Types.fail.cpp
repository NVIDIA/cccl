//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class Alloc>
//   EXPLICIT tuple(allocator_arg_t, const Alloc& a, const Types&...);

// UNSUPPORTED: c++98, c++03


#include <cuda/std/tuple>
#include <cuda/std/cassert>

struct ExplicitCopy {
  TEST_HOST_DEVICE explicit ExplicitCopy(ExplicitCopy const&) {}
  TEST_HOST_DEVICE explicit ExplicitCopy(int) {}
};

TEST_HOST_DEVICE cuda::std::tuple<ExplicitCopy> const_explicit_copy_test() {
    const ExplicitCopy e(42);
    return {cuda::std::allocator_arg, cuda::std::allocator<void>{}, e};
    // expected-error@-1 {{chosen constructor is explicit in copy-initialization}}
}

TEST_HOST_DEVICE cuda::std::tuple<ExplicitCopy> non_const_explicity_copy_test() {
    ExplicitCopy e(42);
    return {cuda::std::allocator_arg, cuda::std::allocator<void>{}, e};
    // expected-error@-1 {{chosen constructor is explicit in copy-initialization}}
}

int main(int, char**)
{
    const_explicit_copy_test();
    non_const_explicity_copy_test();

  return 0;
}
