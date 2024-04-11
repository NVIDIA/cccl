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
// template <class R, class Visitor, class... Variants>
// constexpr R visit(Visitor&& vis, Variants&&... vars);

#include <cuda/std/cassert>
// #include <cuda/std/memory>
// #include <cuda/std/string>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/std/variant>

#include "test_macros.h"
#include "variant_test_helpers.h"

struct visitor_42
{
  template <class T>
  __host__ __device__ constexpr bool operator()(T x) const noexcept
  {
    assert(x == 42);
    return true;
  }
};
struct visitor_42_3
{
  template <class T>
  __host__ __device__ constexpr bool operator()(T x) const noexcept
  {
    assert(x == 42.3);
    return true;
  }
};
struct visitor_float
{
  template <class T>
  __host__ __device__ constexpr bool operator()(T x) const noexcept
  {
    assert(x == -1.3f);
    return true;
  }
};

struct MyVariant : cuda::std::variant<short, long, float>
{
  using cuda::std::variant<short, long, float>::variant;
};

// Check that visit does not take index nor valueless_by_exception members from the base class.
struct EvilVariantBase
{
  int index{};
  char valueless_by_exception{};
};

struct EvilVariant1
    : cuda::std::variant<int, long, double>
    , cuda::std::tuple<int>
    , EvilVariantBase
{
  using cuda::std::variant<int, long, double>::variant;
};

// Check that visit unambiguously picks the variant, even if the other base has __impl member.
struct ImplVariantBase
{
  struct Callable
  {
    __host__ __device__ bool operator()() const
    {
      assert(false);
      return false;
    }
  };

  Callable __impl;
};

struct EvilVariant2
    : cuda::std::variant<int, long, double>
    , ImplVariantBase
{
  using cuda::std::variant<int, long, double>::variant;
};

__host__ __device__ void test_derived_from_variant()
{
  cuda::std::visit<bool>(visitor_42{}, MyVariant{42});
  cuda::std::visit<bool>(visitor_float{}, MyVariant{-1.3f});

  cuda::std::visit<bool>(visitor_42{}, EvilVariant1{42});
  cuda::std::visit<bool>(visitor_42_3{}, EvilVariant1{42.3});

  cuda::std::visit<bool>(visitor_42{}, EvilVariant2{42});
  cuda::std::visit<bool>(visitor_42_3{}, EvilVariant2{42.3});
}

int main(int, char**)
{
  test_derived_from_variant();

  return 0;
}
