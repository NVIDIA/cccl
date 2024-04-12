//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class Alloc, class ...UTypes>
//   tuple(allocator_arg_t, const Alloc& a, tuple<UTypes...>&&);

// UNSUPPORTED: c++98, c++03

#include <cuda/std/tuple>

struct ExplicitCopy
{
  __host__ __device__ explicit ExplicitCopy(int) {}
  __host__ __device__ explicit ExplicitCopy(ExplicitCopy const&) {}
};

__host__ __device__ cuda::std::tuple<ExplicitCopy> explicit_move_test()
{
  cuda::std::tuple<int> t1(42);
  return {cuda::std::allocator_arg, cuda::std::allocator<void>{}, cuda::std::move(t1)};
  // expected-error@-1 {{chosen constructor is explicit in copy-initialization}}
}

int main(int, char**)
{
  return 0;
}
