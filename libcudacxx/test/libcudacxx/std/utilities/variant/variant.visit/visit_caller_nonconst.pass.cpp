//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: clang-7, clang-8

// <cuda/std/variant>
// template <class Visitor, class... Variants>
// constexpr see below visit(Visitor&& vis, Variants&&... vars);

#include <cuda/std/cassert>
// #include <cuda/std/memory>
// #include <cuda/std/string>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/std/variant>

#include "test_macros.h"
#include "variant_test_helpers.h"

// See https://llvm.org/PR31916
__host__ __device__ void test_caller_accepts_nonconst()
{
  struct A
  {};
  struct Visitor
  {
    __host__ __device__ void operator()(A&) {}
  };
  cuda::std::variant<A> v;
  cuda::std::visit(Visitor{}, v);
}

int main(int, char**)
{
  test_caller_accepts_nonconst();

  return 0;
}
