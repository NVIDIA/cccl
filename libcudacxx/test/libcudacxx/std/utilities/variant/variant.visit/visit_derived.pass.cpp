//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

struct almost_string
{
  const char* ptr;

  __host__ __device__ almost_string(const char* ptr)
      : ptr(ptr)
  {}

  __host__ __device__ friend bool operator==(const almost_string& lhs, const almost_string& rhs)
  {
    return lhs.ptr == rhs.ptr;
  }
};

struct MyVariant : cuda::std::variant<short, long, float>
{
  using cuda::std::variant<short, long, float>::variant;
};

namespace cuda::std
{
template <size_t Index>
__host__ __device__ void get(const MyVariant&)
{
  assert(false);
}
} // namespace cuda::std

struct visitor_42
{
  template <class T>
  __host__ __device__ constexpr bool operator()(T x) const noexcept
  {
    assert(x == 42);
    return true;
  }
};
struct visitor_142
{
  template <class T>
  __host__ __device__ constexpr bool operator()(T x) const noexcept
  {
    assert(x == 142);
    return true;
  }
};
struct visitor_float
{
  template <class T>
  __host__ __device__ constexpr bool operator()(T x) const noexcept
  {
    assert(x == -1.25f);
    return true;
  }
};
struct visitor_double
{
  template <class T>
  __host__ __device__ constexpr bool operator()(T x) const noexcept
  {
    assert(x == 42.3);
    return true;
  }
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
  auto v1        = MyVariant{short(42)};
  const auto cv1 = MyVariant{long(142)};

  cuda::std::visit(visitor_42{}, v1);
  cuda::std::visit(visitor_142{}, cv1);
  cuda::std::visit(visitor_float{}, MyVariant{-1.25f});
  cuda::std::visit(visitor_42{}, cuda::std::move(v1));
  cuda::std::visit(visitor_142{}, cuda::std::move(cv1));

  cuda::std::visit(visitor_42{}, EvilVariant1{42});
  cuda::std::visit(visitor_double{}, EvilVariant1{42.3});

  cuda::std::visit(visitor_42{}, EvilVariant2{42});
  cuda::std::visit(visitor_double{}, EvilVariant2{42.3});
}

int main(int, char**)
{
  test_derived_from_variant();

  return 0;
}
