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

#include <cuda/std/cassert>
#include <cuda/std/tuple>

struct ExplicitCopy
{
  __host__ __device__ ExplicitCopy(int) {}
  __host__ __device__ explicit ExplicitCopy(ExplicitCopy const&) {}
};

__host__ __device__ std::tuple<ExplicitCopy> const_explicit_copy()
{
  const ExplicitCopy e(42);
  return {e};
  // expected-error@-1 {{chosen constructor is explicit in copy-initialization}}
}

__host__ __device__ std::tuple<ExplicitCopy> non_const_explicit_copy()
{
  ExplicitCopy e(42);
  return {e};
  // expected-error@-1 {{chosen constructor is explicit in copy-initialization}}
}

__host__ __device__ std::tuple<ExplicitCopy> const_explicit_copy_no_brace()
{
  const ExplicitCopy e(42);
  return e;
  // expected-error@-1 {{no viable conversion}}
}

int main(int, char**)
{
  return 0;
}
