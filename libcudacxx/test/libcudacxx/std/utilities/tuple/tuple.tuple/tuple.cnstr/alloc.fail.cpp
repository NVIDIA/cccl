//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class Alloc>
//   explicit(see-below) tuple(allocator_arg_t, const Alloc& a);

// Make sure we get the explicit-ness of the constructor right.
// This is LWG 3158.

#include <cuda/std/tuple>

struct ExplicitDefault
{
  __host__ __device__ explicit ExplicitDefault() {}
};

__host__ __device__ std::tuple<ExplicitDefault> explicit_default_test()
{
  return {cuda::std::allocator_arg, cuda::std::allocator<int>()}; // expected-error {{chosen constructor is explicit in
                                                                  // copy-initialization}}
}

int main(int, char**)
{
  return 0;
}
