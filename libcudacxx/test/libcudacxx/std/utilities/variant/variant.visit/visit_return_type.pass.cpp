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

  TEST_FUNC almost_string(const char* ptr)
      : ptr(ptr)
  {}

  TEST_FUNC friend bool operator==(const almost_string& lhs, const almost_string& rhs)
  {
    return lhs.ptr == rhs.ptr;
  }
};

TEST_FUNC void test_return_type()
{
  using Fn = ForwardingCallObject;
  Fn obj{};
  [[maybe_unused]] const Fn& cobj = obj;
  { // test call operator forwarding - no variant
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(obj)), Fn&>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(cobj)), const Fn&>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(cuda::std::move(obj))), Fn&&>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(cuda::std::move(cobj))), const Fn&&>);
  }
  { // test call operator forwarding - single variant, single arg
    using V = cuda::std::variant<int>;
    [[maybe_unused]] V v(42);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(obj, v)), Fn&>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(cobj, v)), const Fn&>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(cuda::std::move(obj), v)), Fn&&>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(cuda::std::move(cobj), v)), const Fn&&>);
  }
  { // test call operator forwarding - single variant, multi arg
    using V = cuda::std::variant<int, long, double>;
    [[maybe_unused]] V v(42l);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(obj, v)), Fn&>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(cobj, v)), const Fn&>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(cuda::std::move(obj), v)), Fn&&>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(cuda::std::move(cobj), v)), const Fn&&>);
  }
  { // test call operator forwarding - multi variant, multi arg
    using V  = cuda::std::variant<int, long, double>;
    using V2 = cuda::std::variant<int*, almost_string>;
    [[maybe_unused]] V v(42l);
    [[maybe_unused]] V2 v2("hello");
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(obj, v, v2)), Fn&>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(cobj, v, v2)), const Fn&>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(cuda::std::move(obj), v, v2)), Fn&&>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(cuda::std::move(cobj), v, v2)), const Fn&&>);
  }
  {
    using V = cuda::std::variant<int, long, double, almost_string>;
    [[maybe_unused]] V v1(42l), v2("hello"), v3(101), v4(1.1);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(obj, v1, v2, v3, v4)), Fn&>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(cobj, v1, v2, v3, v4)), const Fn&>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(cuda::std::move(obj), v1, v2, v3, v4)), Fn&&>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(cuda::std::move(cobj), v1, v2, v3, v4)), const Fn&&>);
  }
  {
    using V = cuda::std::variant<int, long, double, int*, almost_string>;
    [[maybe_unused]] V v1(42l), v2("hello"), v3(nullptr), v4(1.1);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(obj, v1, v2, v3, v4)), Fn&>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(cobj, v1, v2, v3, v4)), const Fn&>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(cuda::std::move(obj), v1, v2, v3, v4)), Fn&&>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(cuda::std::move(cobj), v1, v2, v3, v4)), const Fn&&>);
  }
}

int main(int, char**)
{
  test_return_type();

  return 0;
}
