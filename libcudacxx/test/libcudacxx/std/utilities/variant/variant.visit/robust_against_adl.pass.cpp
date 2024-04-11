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

#include <cuda/std/variant>

#include "test_macros.h"

struct Incomplete;
template <class T>
struct Holder
{
  T t;
};

struct empty_visitor
{
  template <class T>
  __host__ __device__ constexpr void operator()(T) const noexcept
  {}
};

struct holder_visitor
{
  template <class T>
  __host__ __device__ constexpr Holder<Incomplete>* operator()(T) const noexcept
  {
    return nullptr;
  }
};

__host__ __device__ constexpr bool test(bool do_it)
{
  if (do_it)
  {
    cuda::std::variant<Holder<Incomplete>*, int> v = nullptr;
    cuda::std::visit(empty_visitor{}, v);
    cuda::std::visit(holder_visitor{}, v);
    cuda::std::visit<void>(empty_visitor{}, v);
    cuda::std::visit<void*>(holder_visitor{}, v);
  }
  return true;
}

int main(int, char**)
{
  test(true);
  static_assert(test(true), "");
  return 0;
}
